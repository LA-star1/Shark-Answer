"""Topic frequency analyser and next-session predictor.

Reads question papers via the manifest, extracts topics per question using
Claude Haiku, builds a topic × session frequency matrix, then predicts
high-probability topics for the next exam session.

Usage:
    # Predict for Economics Paper 2, target 2026:
    python -m shark_answer.knowledge_base.topic_predictor \\
        --subject economics --paper 2 --target-year 2026

    # All papers for a subject:
    python -m shark_answer.knowledge_base.topic_predictor --subject economics --target-year 2026

    # All subjects:
    python -m shark_answer.knowledge_base.topic_predictor --all --target-year 2026

Output file: shark_answer/knowledge_base/predictions/{subject_key}_{target_year}_paper{paper}_predictions.json

The predictions are also surfaced via get_topic_predictions() for display in the UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DEFAULT_KB_DIR = Path.home() / "Desktop/BTC/AI/shark_answer_kb"
KB_DIR: Path = Path(os.getenv("SHARK_ANSWER_KB_DIR", str(_DEFAULT_KB_DIR))).expanduser()

_SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = _SCRIPT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

# ── Subject map ───────────────────────────────────────────────────────────────
_SUBJECT_MAP: dict[str, str] = {
    "physics":             "physics_9702",
    "chemistry":           "chemistry_9701",
    "math":                "math_9709",
    "mathematics":         "math_9709",
    "further math":        "further_math_9231",
    "further mathematics": "further_math_9231",
    "economics":           "economics_9708",
    "computer science":    "cs_9618",
    "cs":                  "cs_9618",
    "physics_9702":        "physics_9702",
    "chemistry_9701":      "chemistry_9701",
    "math_9709":           "math_9709",
    "further_math_9231":   "further_math_9231",
    "economics_9708":      "economics_9708",
    "cs_9618":             "cs_9618",
}

ALL_SUBJECT_KEYS = list(dict.fromkeys(_SUBJECT_MAP.values()))

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 4096


# ── Manifest loading ───────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    mpath = KB_DIR / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"manifest.json not found at {mpath}")
    return json.loads(mpath.read_text(encoding="utf-8"))


def _read_txt(subject_key: str, category: str, filename: str) -> str | None:
    txt_path = KB_DIR / subject_key / category / Path(filename).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="replace")
    return None


def _matches_paper(entry_paper: int | None, paper_number: int | None) -> bool:
    if paper_number is None:
        return True
    if entry_paper is None:
        return False
    if paper_number < 10:
        return entry_paper // 10 == paper_number
    return entry_paper == paper_number


# ── Step 1: Topic extraction ───────────────────────────────────────────────────

_TOPIC_EXTRACT_PROMPT = """\
You are a CIE A-Level curriculum expert. Extract ALL topics covered by questions in this {subject} Paper {paper} question paper.

For each distinct question (including sub-parts), identify:
- question_id: e.g. "1(a)", "2(b)", "Section B Essay 3"
- marks: marks available (if shown)
- topics: list of 1-4 specific syllabus topics tested (e.g. "elasticity of demand", "monetary policy", "market failure")
- subtopics: more specific tags (e.g. "price elasticity", "quantitative easing", "externalities")

Return ONLY valid JSON (no markdown):
{{"paper":{paper},"year":{year},"session":"{session}","questions":[
  {{"question_id":"1(a)","marks":2,"topics":["topic1"],"subtopics":["subtopic1"]}}
]}}

Keep topics concise (2-5 words each). Use standard CIE syllabus terminology.

QUESTION PAPER:
{qp_text}
"""

# ── Step 2: Prediction logic ───────────────────────────────────────────────────

# Probability boost rules (applied after base frequency):
# - Topic not tested in ≥2 sessions → HIGH (+0.2)
# - Topic not tested in ≥3 sessions → VERY HIGH (+0.35)
# - Topic tested in every session → core_topic (no prediction boost, but flag as reliable)
# - Brand-new syllabus topic → HIGH

_GAP_BOOST = {
    1: 0.05,   # tested recently, slight recency discount
    2: 0.20,   # 2-session gap → likely due
    3: 0.35,   # 3-session gap → overdue
    4: 0.45,   # 4-session gap → very overdue
}


def _build_frequency_matrix(
    all_topic_data: list[dict],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build topic → list of session_keys where it appeared.

    Returns:
        topic_sessions: dict mapping topic_label → list of "2023_june" strings
        subtopic_map:   dict mapping topic_label → most specific subtopic seen
    """
    topic_sessions: dict[str, list[str]] = defaultdict(list)
    subtopic_map: dict[str, str] = {}

    for entry in all_topic_data:
        year = entry.get("year", 0)
        session = entry.get("session", "unknown")
        session_key = f"{year}_{session}"

        for q in entry.get("questions", []):
            for topic in q.get("topics", []):
                t = topic.strip().lower()
                if t and session_key not in topic_sessions[t]:
                    topic_sessions[t].append(session_key)
            for subtopic in q.get("subtopics", []):
                st = subtopic.strip().lower()
                # Keep subtopics paired with parent topics
                for topic in q.get("topics", []):
                    t = topic.strip().lower()
                    if t:
                        subtopic_map[t] = st

    return dict(topic_sessions), subtopic_map


def _predict_topics(
    topic_sessions: dict[str, list[str]],
    subtopic_map: dict[str, str],
    all_sessions_sorted: list[str],
    target_year: int,
    n_recent: int = 3,
) -> list[dict]:
    """Produce ranked predictions for the target year.

    Args:
        topic_sessions:     topic → list of sessions where it appeared
        subtopic_map:       topic → most specific subtopic
        all_sessions_sorted: all session keys, sorted chronologically
        target_year:        year to predict for
        n_recent:           number of recent sessions to check gap against

    Returns sorted list of prediction dicts.
    """
    recent_sessions = all_sessions_sorted[-n_recent:] if len(all_sessions_sorted) >= n_recent else all_sessions_sorted
    all_n = len(all_sessions_sorted)

    predictions: list[dict] = []
    for topic, sessions_present in topic_sessions.items():
        sessions_present_sorted = sorted(sessions_present)
        last_tested = sessions_present_sorted[-1] if sessions_present_sorted else None

        # Base frequency: fraction of all sessions where this topic appeared
        base_freq = len(sessions_present) / max(all_n, 1)

        # Gap: how many recent sessions is this topic absent?
        gap = sum(1 for s in recent_sessions if s not in sessions_present)

        # Frequency in recent sessions (last n_recent)
        recent_freq = sum(1 for s in recent_sessions if s in sessions_present) / max(len(recent_sessions), 1)

        # Compute probability
        boost = _GAP_BOOST.get(gap, 0.45 if gap > 4 else 0.0)
        probability = min(0.97, base_freq * 0.6 + recent_freq * 0.2 + boost)

        # Label classification
        if base_freq >= 0.9:
            label = "core_topic"       # appears in nearly every session
        elif gap >= 3:
            label = "VERY HIGH"
        elif gap == 2:
            label = "HIGH"
        elif gap == 1:
            label = "MEDIUM"
        else:
            label = "LOW"              # tested very recently

        predictions.append({
            "topic":             topic,
            "subtopic":          subtopic_map.get(topic, ""),
            "probability":       round(probability, 3),
            "label":             label,
            "last_tested":       last_tested,
            "sessions_appeared": len(sessions_present),
            "total_sessions":    all_n,
            "gap_sessions":      gap,
            "base_freq":         round(base_freq, 3),
        })

    # Sort: VERY HIGH first, then HIGH, then by probability desc
    _label_order = {"VERY HIGH": 0, "HIGH": 1, "core_topic": 2, "MEDIUM": 3, "LOW": 4}
    predictions.sort(
        key=lambda x: (_label_order.get(x["label"], 5), -x["probability"])
    )
    return predictions


async def _call_haiku(client: anthropic.AsyncAnthropic, prompt: str, label: str) -> str:
    logger.info("  Haiku → %s", label)
    msg = await client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else ""


def _safe_parse_json(text: str, label: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error for '%s': %s", label, e)
        return None


# ── Main analysis function ─────────────────────────────────────────────────────

async def build_topic_predictions(
    subject: str,
    paper_number: int,
    target_year: int = 2026,
    preview_only: bool = False,
) -> dict | None:
    """Run full topic-frequency analysis and generate predictions.

    Args:
        subject:      Subject name or manifest key
        paper_number: CIE paper component (e.g. 2 for Paper 2x)
        target_year:  Year to predict topics for
        preview_only: If True, print results without saving to disk

    Returns prediction dict, or None on failure.
    """
    subject_key = _SUBJECT_MAP.get(subject.lower().strip(), subject)
    manifest = _load_manifest()
    if subject_key not in manifest:
        logger.error("Subject '%s' not found in manifest", subject_key)
        return None

    data = manifest[subject_key]
    qp_entries = [
        e for e in data.get("question_papers", [])
        if _matches_paper(e.get("paper"), paper_number)
        and e.get("session") not in ("specimen",)
    ]

    if not qp_entries:
        logger.warning("No question papers for %s paper %s", subject_key, paper_number)
        return None

    qp_entries.sort(key=lambda e: (e.get("year", 0), e.get("session", "")))
    logger.info(
        "Found %d question papers for %s paper %s (%d–%d)",
        len(qp_entries),
        subject_key,
        paper_number,
        qp_entries[0].get("year", 0),
        qp_entries[-1].get("year", 0),
    )

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.AsyncAnthropic(api_key=api_key)
    subject_display = subject_key.replace("_", " ").title().replace("9702", "").strip()

    # ── Step 1: Extract topics from each question paper ────────────────────
    all_topic_data: list[dict] = []
    all_sessions_sorted: list[str] = []

    for entry in qp_entries:
        year = entry.get("year", "?")
        session = entry.get("session", "unknown")
        paper = entry.get("paper", "?")
        fname = entry.get("renamed") or entry.get("original", "")
        text = _read_txt(subject_key, "question_papers", fname)
        if not text:
            logger.debug("No .txt for %s %s paper %s — skipping", year, session, paper)
            continue

        session_key = f"{year}_{session}"
        if session_key not in all_sessions_sorted:
            all_sessions_sorted.append(session_key)

        label = f"{year} {session} p{paper}"
        # Truncate long papers
        if len(text) > 20_000:
            text = text[:20_000] + "\n[...truncated]"

        prompt = _TOPIC_EXTRACT_PROMPT.format(
            subject=subject_display,
            paper=paper,
            year=year,
            session=session,
            qp_text=text,
        )

        raw = await _call_haiku(client, prompt, label)
        parsed = _safe_parse_json(raw, label)
        if parsed:
            parsed["session"] = session
            parsed["session_key"] = session_key
            all_topic_data.append(parsed)
            n_q = len(parsed.get("questions", []))
            all_topics = {t for q in parsed.get("questions", []) for t in q.get("topics", [])}
            logger.info("    ✓ %s → %d questions, %d distinct topics", label, n_q, len(all_topics))
        else:
            logger.warning("    ✗ Failed to parse topics from %s", label)

    if not all_topic_data:
        logger.error("No topic data extracted")
        return None

    # Sort sessions chronologically
    all_sessions_sorted = sorted(set(all_sessions_sorted))

    # ── Step 2: Build frequency matrix ────────────────────────────────────
    logger.info("Building topic frequency matrix...")
    topic_sessions, subtopic_map = _build_frequency_matrix(all_topic_data)
    logger.info("  %d distinct topics identified across %d sessions",
                len(topic_sessions), len(all_sessions_sorted))

    # ── Step 3: Predict ────────────────────────────────────────────────────
    logger.info("Computing predictions for %s...", target_year)
    predictions = _predict_topics(
        topic_sessions,
        subtopic_map,
        all_sessions_sorted,
        target_year,
    )

    # ── Step 4: Assemble output ────────────────────────────────────────────
    output = {
        "subject":          subject_key,
        "subject_display":  subject_display,
        "paper":            paper_number,
        "target_year":      target_year,
        "sessions_analysed": all_sessions_sorted,
        "total_sessions":   len(all_sessions_sorted),
        "total_topics":     len(topic_sessions),
        "generated_at":     "2026-03-05",
        "predictions":      predictions,
        "_raw_by_session":  all_topic_data,  # full breakdown, useful for debugging
    }

    # ── Step 5: Save ──────────────────────────────────────────────────────
    out_path = PREDICTIONS_DIR / f"{subject_key}_{target_year}_paper{paper_number}_predictions.json"
    if not preview_only:
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("✅ Saved predictions → %s", out_path)

    return output


# ── Public query function ──────────────────────────────────────────────────────

def get_topic_predictions(
    subject: str,
    paper: int,
    target_year: int = 2026,
    min_label: str = "MEDIUM",
    top_n: int = 20,
) -> list[dict]:
    """Load and return cached topic predictions for a subject+paper.

    Falls back to empty list if no predictions file exists yet.
    (Run build_topic_predictions() first to generate the file.)

    Args:
        subject:     Subject name or key
        paper:       Paper component number
        target_year: Year of predictions to load
        min_label:   Minimum probability label to include ("VERY HIGH"|"HIGH"|"MEDIUM"|"LOW"|"core_topic")
        top_n:       Maximum number of predictions to return

    Returns sorted list of prediction dicts.
    """
    subject_key = _SUBJECT_MAP.get(subject.lower().strip(), subject)
    pred_path = PREDICTIONS_DIR / f"{subject_key}_{target_year}_paper{paper}_predictions.json"

    if not pred_path.exists():
        return []

    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        predictions = data.get("predictions", [])

        _label_priority = {"VERY HIGH": 0, "HIGH": 1, "core_topic": 2, "MEDIUM": 3, "LOW": 4}
        _min_priority = _label_priority.get(min_label, 3)

        filtered = [
            p for p in predictions
            if _label_priority.get(p.get("label", "LOW"), 4) <= _min_priority
        ]
        return filtered[:top_n]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load predictions from %s: %s", pred_path, exc)
        return []


def format_predictions_for_prompt(
    subject: str,
    paper: int,
    target_year: int = 2026,
) -> str:
    """Return a ready-to-inject prediction block for model prompts.

    Returns empty string if no predictions available.
    """
    preds = get_topic_predictions(subject, paper, target_year, min_label="HIGH")
    if not preds:
        return ""

    lines = [
        f"=== TOPIC PREDICTIONS FOR {target_year} ===",
        f"Based on {subject} Paper {paper} topic frequency analysis:",
        "",
    ]

    for p in preds[:15]:
        label = p.get("label", "")
        topic = p.get("topic", "")
        subtopic = p.get("subtopic", "")
        gap = p.get("gap_sessions", 0)
        last = p.get("last_tested", "")

        subtopic_str = f" → {subtopic}" if subtopic and subtopic != topic else ""
        gap_str = f" (not tested in {gap} sessions)" if gap >= 2 else f" (last: {last})"
        lines.append(f"  [{label}] {topic}{subtopic_str}{gap_str}")

    lines.append("")
    return "\n".join(lines)


def _print_predictions(output: dict) -> None:
    """Print human-readable predictions summary."""
    print("\n" + "═" * 70)
    print(f"  TOPIC PREDICTIONS: {output['subject_display']} Paper {output['paper']}")
    print(f"  Target year: {output['target_year']}")
    print(f"  Sessions analysed: {output['total_sessions']} "
          f"({output['sessions_analysed'][0] if output['sessions_analysed'] else '?'}"
          f" — {output['sessions_analysed'][-1] if output['sessions_analysed'] else '?'})")
    print(f"  Distinct topics found: {output['total_topics']}")
    print("═" * 70)

    _label_order = {"VERY HIGH": 0, "HIGH": 1, "core_topic": 2, "MEDIUM": 3, "LOW": 4}
    preds = output.get("predictions", [])

    for label_group, label_name in [
        (0, "🔴 VERY HIGH probability"),
        (1, "🟠 HIGH probability"),
        (2, "🔵 CORE TOPIC (appears every session)"),
        (3, "🟡 MEDIUM probability"),
    ]:
        group = [p for p in preds if _label_order.get(p.get("label", ""), 5) == label_group]
        if not group:
            continue
        print(f"\n  {label_name}:")
        for p in group[:10]:
            topic = p.get("topic", "")
            subtopic = p.get("subtopic", "")
            gap = p.get("gap_sessions", 0)
            last = p.get("last_tested", "unknown")
            freq = p.get("base_freq", 0)
            prob = p.get("probability", 0)
            print(f"    • {topic:<40s}  gap={gap}  last={last}  freq={freq:.0%}  p={prob:.0%}")
            if subtopic and subtopic != topic:
                print(f"        └─ subtopic: {subtopic}")

    print("═" * 70 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build topic frequency matrix and generate next-session predictions."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process ALL subjects")
    group.add_argument("--subject", metavar="SUBJ", help="Subject name or key")
    parser.add_argument("--paper", type=int, metavar="N", help="Paper component")
    parser.add_argument("--target-year", type=int, default=2026, help="Prediction target year")
    parser.add_argument("--preview", action="store_true", help="Print without saving")
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()

    if args.all:
        manifest = _load_manifest()
        for sk in ALL_SUBJECT_KEYS:
            if sk not in manifest:
                continue
            qp_entries = manifest[sk].get("question_papers", [])
            components = sorted(set(
                e["paper"] // 10
                for e in qp_entries if e.get("paper") is not None
            ))
            for comp in components:
                out = await build_topic_predictions(sk, comp, args.target_year, args.preview)
                if out:
                    _print_predictions(out)
        return

    if not args.subject:
        print("Error: specify --subject or --all")
        sys.exit(1)

    subject_key = _SUBJECT_MAP.get(args.subject.lower().strip(), args.subject)

    if args.paper is not None:
        out = await build_topic_predictions(subject_key, args.paper, args.target_year, args.preview)
        if out:
            _print_predictions(out)
    else:
        manifest = _load_manifest()
        if subject_key not in manifest:
            logger.error("Subject not found: %s", subject_key)
            sys.exit(1)
        qp_entries = manifest[subject_key].get("question_papers", [])
        components = sorted(set(
            e["paper"] // 10 for e in qp_entries if e.get("paper") is not None
        ))
        for comp in components:
            out = await build_topic_predictions(subject_key, comp, args.target_year, args.preview)
            if out:
                _print_predictions(out)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
