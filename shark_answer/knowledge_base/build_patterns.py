"""Analyze ALL mark schemes for a subject+paper combo and extract CIE marking patterns.

Sends each year's mark scheme to Claude Haiku, then synthesises the patterns across years
into a single JSON file saved to:
    shark_answer/knowledge_base/patterns/{subject_key}_paper{paper}_patterns.json

Usage:
    # Preview only Economics Paper 2 (cheap sanity check, ~$0.10-0.20):
    python -m shark_answer.knowledge_base.build_patterns --subject economics --paper 2

    # Process ALL subjects and ALL papers:
    python -m shark_answer.knowledge_base.build_patterns --all

    # Specific subject, all its papers:
    python -m shark_answer.knowledge_base.build_patterns --subject physics

    # Skip cross-validation step:
    python -m shark_answer.knowledge_base.build_patterns --subject economics --paper 2 --no-validate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
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
PATTERNS_DIR = _SCRIPT_DIR / "patterns"
PATTERNS_DIR.mkdir(exist_ok=True)

# ── Subject map (mirrors retriever.py) ────────────────────────────────────────
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

ALL_SUBJECT_KEYS = list(dict.fromkeys(_SUBJECT_MAP.values()))  # unique, stable order

# Haiku model — cheapest Anthropic model with strong reading ability
_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS_PER_CALL = 8192   # mark scheme patterns can be long; 4096 was truncating

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


# ── Prompts ────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
CIE mark scheme extractor. For {label}, extract ALL questions as compact JSON.

Types: calculation|explain_single|explain_multi|assess_analyse|essay_knowledge|essay_evaluation|data_extract|diagram|define

Output ONLY valid JSON (no markdown fences). Keep text fields short (≤60 chars each).
Schema:
{{"year":{year},"session":"{session}","paper":{paper},"question_patterns":[
  {{"question_id":"1(a)","question_stem":"<60 chars>","type":"calculation","marks":2,
   "mark_structure":"2 pts","full_mark_formula":"method(1)+answer(1)",
   "mark_points":["correct ans 4.5%(1)","OR 60.8% with working(1)"],
   "common_penalties":[],"examiner_notes":""}}
]}}

MARK SCHEME:
{mark_scheme_text}
"""

_SYNTHESISE_PROMPT = """\
You are a CIE A-Level examiner expert. You have received pattern analysis from {n_years} years
of mark schemes for {subject} Paper {paper}. Your task: synthesise these into a single
"meta-pattern" document that describes what ALWAYS happens, what USUALLY happens, and
what VARIES across years.

For each recurring question position (e.g. 1(a), 1(b), essay parts), produce:
1. The CONSISTENT pattern (what never changes)
2. The TYPICAL variation range (marks, types, structures that rotate)
3. The FORMULA for full marks (generalised)
4. The EXAMINER TENDENCIES across years (what they reward, what they penalise)
5. A CONFIDENCE score 0-1 (1 = perfectly consistent across all years)
6. The YEARS ANALYSED

Input data (JSON array of per-year patterns):
{per_year_json}

Return a JSON object with this schema:
{{
  "subject": "{subject}",
  "subject_key": "{subject_key}",
  "paper": {paper},
  "years_analysed": [2021, 2022, 2023, 2024],
  "generated_at": "2026-03-05",
  "question_patterns": [
    {{
      "question_id": "1(a)",
      "type": "calculation",
      "typical_marks": 2,
      "marks_range": [2, 2],          // [min, max] seen across years
      "mark_structure": "...",
      "full_mark_formula": "...",
      "consistent_across_years": true,
      "common_mark_points": [         // points that appear in MOST years
        "correct numerical answer (1)",
        "working shown for percentage change (1)"
      ],
      "common_penalties": [],
      "examiner_tendencies": "",      // what examiners reward/penalise most
      "confidence": 0.95,
      "years_analysed": [2021, 2022, 2023, 2024, 2025]
    }}
  ]
}}

IMPORTANT: Output ONLY valid JSON, no markdown, no commentary.
"""

_VALIDATE_PROMPT = """\
You are a CIE A-Level examiner expert. You have a synthesised pattern document and a
held-out mark scheme (one that was NOT used to build the patterns).

Your task: check how well the pattern document PREDICTS the actual mark scheme structure.

For each question in the held-out mark scheme:
- Does the pattern correctly predict the question TYPE? (yes/partial/no)
- Does the pattern correctly predict the MARKS? (yes/within_1/off)
- Does the FULL MARK FORMULA match? (yes/partial/no)
- Are the predicted mark points present in the actual mark scheme?

Return a JSON validation report:
{{
  "holdout_label": "{holdout_label}",
  "overall_accuracy": 0.85,           // fraction of questions correctly predicted
  "question_validations": [
    {{
      "question_id": "1(a)",
      "type_correct": true,
      "marks_correct": true,
      "formula_match": "yes",
      "mark_points_found": 2,
      "mark_points_total": 2,
      "notes": ""
    }}
  ],
  "summary": "Pattern correctly predicted 8/10 questions. Misses: ..."
}}

PATTERN DOCUMENT:
{patterns_json}

HELD-OUT MARK SCHEME:
{holdout_ms_text}
"""


# ── Core functions ─────────────────────────────────────────────────────────────

async def _call_haiku(client: anthropic.AsyncAnthropic, prompt: str, label: str) -> str:
    """Call Claude Haiku and return the text response."""
    logger.info("  Calling Haiku for: %s", label)
    msg = await client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=_MAX_TOKENS_PER_CALL,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else ""


def _safe_parse_json(text: str, label: str) -> dict | list | None:
    """Attempt to parse JSON, stripping markdown fences if present.

    If the response was truncated (common with long mark schemes), tries to
    recover by closing open arrays/objects before parsing.
    """
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to salvage truncated JSON: close any open arrays/objects
        recovered = _try_recover_json(text)
        if recovered is not None:
            logger.info("  Recovered truncated JSON for '%s' (lost last %d chars)",
                        label, len(text) - len(text.rstrip()))
            return recovered
        logger.warning("JSON parse error for '%s': %s — raw:\n%s", label, e, text[:300])
        return None


def _try_recover_json(text: str) -> dict | list | None:
    """Attempt to close truncated JSON by counting brackets/braces."""
    # Remove any trailing incomplete value (up to last complete item)
    # Strategy: try progressively shorter truncation points
    for end_char in [",\n    {", ",\n  {", ",\n{"]:
        idx = text.rfind(end_char)
        if idx > 0:
            truncated = text[:idx]
            # Count unclosed brackets
            close = _close_json(truncated)
            if close:
                try:
                    return json.loads(close)
                except json.JSONDecodeError:
                    pass
    # Last resort: close from current position
    close = _close_json(text)
    if close:
        try:
            return json.loads(close)
        except json.JSONDecodeError:
            pass
    return None


def _close_json(text: str) -> str | None:
    """Add closing brackets/braces to make truncated JSON valid."""
    # Track open brackets
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if not stack:
        return text  # already balanced
    # Close open structures in reverse
    return text + "".join(reversed(stack))


async def analyse_subject_paper(
    subject: str,
    paper_number: int,
    validate: bool = True,
    preview_only: bool = False,
) -> dict | None:
    """Run full pattern analysis for one subject+paper combo.

    Args:
        subject:      Subject name or manifest key (e.g. "economics", "economics_9708")
        paper_number: CIE paper component number (e.g. 2 for Paper 2x)
        validate:     Whether to run cross-validation on held-out mark schemes
        preview_only: If True, only show a summary without saving to disk

    Returns the synthesised patterns dict, or None on failure.
    """
    subject_key = _SUBJECT_MAP.get(subject.lower().strip(), subject)
    manifest = _load_manifest()
    if subject_key not in manifest:
        logger.error("Subject '%s' not found in manifest", subject_key)
        return None

    data = manifest[subject_key]
    ms_entries = data.get("mark_schemes", [])
    ms_matching = [
        e for e in ms_entries
        if _matches_paper(e.get("paper"), paper_number)
        and e.get("session") != "specimen"  # skip specimen mark schemes
    ]

    if not ms_matching:
        logger.warning("No mark schemes found for %s paper=%s", subject_key, paper_number)
        return None

    # Sort by year ascending so we process chronologically
    ms_matching.sort(key=lambda e: (e.get("year", 0), e.get("session", "")))
    logger.info(
        "Found %d mark schemes for %s paper %s (%d–%d)",
        len(ms_matching),
        subject_key,
        paper_number,
        ms_matching[0].get("year", 0),
        ms_matching[-1].get("year", 0),
    )

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Step 1: Extract patterns from each mark scheme ─────────────────────
    # Hold out 2-3 recent mark schemes for validation if requested
    if validate and len(ms_matching) >= 4:
        n_holdout = min(3, len(ms_matching) // 4)
        holdout_entries = ms_matching[-n_holdout:]
        train_entries = ms_matching[:-n_holdout]
    else:
        holdout_entries = []
        train_entries = ms_matching

    logger.info(
        "Train: %d mark schemes | Holdout: %d",
        len(train_entries),
        len(holdout_entries),
    )

    per_year_results: list[dict] = []
    for entry in train_entries:
        year = entry.get("year", "?")
        session = entry.get("session", "unknown")
        paper = entry.get("paper", "?")
        fname = entry.get("renamed") or entry.get("original", "")
        text = _read_txt(subject_key, "mark_schemes", fname)
        if not text:
            logger.warning("Could not read .txt for %s %s paper %s — skipping", year, session, paper)
            continue

        label = f"{year} {session} paper {paper}"
        # Truncate very long mark schemes (keep first 30k chars ~ 7.5k tokens)
        if len(text) > 30_000:
            text = text[:30_000] + "\n[...truncated for token budget]"

        prompt = _EXTRACT_PROMPT.format(
            label=label,
            year=year,
            session=session,
            paper=paper,
            mark_scheme_text=text,
        )

        raw = await _call_haiku(client, prompt, label)
        parsed = _safe_parse_json(raw, label)
        if parsed:
            per_year_results.append(parsed)
            logger.info("    ✓ Extracted %d question patterns from %s",
                        len(parsed.get("question_patterns", [])), label)
        else:
            logger.warning("    ✗ Failed to parse patterns from %s", label)

    if not per_year_results:
        logger.error("No patterns extracted — aborting")
        return None

    # ── Step 2: Synthesise across years ───────────────────────────────────
    logger.info("Synthesising %d year-patterns into meta-pattern...", len(per_year_results))
    synth_prompt = _SYNTHESISE_PROMPT.format(
        n_years=len(per_year_results),
        subject=subject_key.replace("_", " ").title(),
        subject_key=subject_key,
        paper=paper_number,
        per_year_json=json.dumps(per_year_results, indent=2)[:40_000],  # cap at 40k
    )

    raw_synth = await _call_haiku(client, synth_prompt, f"synthesis {subject_key} p{paper_number}")
    patterns = _safe_parse_json(raw_synth, "synthesis")
    if not patterns:
        logger.error("Failed to synthesise patterns")
        return None

    logger.info("  ✓ Synthesised %d question patterns", len(patterns.get("question_patterns", [])))

    # ── Step 3: Cross-validation ───────────────────────────────────────────
    validation_results: list[dict] = []
    if validate and holdout_entries:
        logger.info("Running cross-validation on %d held-out mark schemes...", len(holdout_entries))
        for entry in holdout_entries:
            year = entry.get("year", "?")
            session = entry.get("session", "unknown")
            paper = entry.get("paper", "?")
            fname = entry.get("renamed") or entry.get("original", "")
            text = _read_txt(subject_key, "mark_schemes", fname)
            if not text:
                continue

            holdout_label = f"{year} {session} paper {paper}"
            if len(text) > 20_000:
                text = text[:20_000] + "\n[...truncated]"

            val_prompt = _VALIDATE_PROMPT.format(
                holdout_label=holdout_label,
                patterns_json=json.dumps(patterns, indent=2)[:20_000],
                holdout_ms_text=text,
            )

            raw_val = await _call_haiku(client, val_prompt, f"validate {holdout_label}")
            parsed_val = _safe_parse_json(raw_val, f"validate {holdout_label}")
            if parsed_val:
                validation_results.append(parsed_val)
                acc = parsed_val.get("overall_accuracy", 0)
                logger.info("    Holdout %s → accuracy %.0f%%", holdout_label, acc * 100)
                logger.info("    %s", parsed_val.get("summary", ""))

    # ── Attach validation to output ────────────────────────────────────────
    if validation_results:
        patterns["validation"] = {
            "holdout_count": len(holdout_entries),
            "holdout_labels": [
                f"{e.get('year')} {e.get('session')} paper {e.get('paper')}"
                for e in holdout_entries
            ],
            "results": validation_results,
            "mean_accuracy": round(
                sum(r.get("overall_accuracy", 0) for r in validation_results)
                / len(validation_results),
                3,
            ) if validation_results else None,
        }

    patterns["_meta"] = {
        "train_count": len(train_entries),
        "holdout_count": len(holdout_entries),
        "total_ms_available": len(ms_matching),
    }

    # ── Step 4: Save ──────────────────────────────────────────────────────
    out_path = PATTERNS_DIR / f"{subject_key}_paper{paper_number}_patterns.json"
    if not preview_only:
        out_path.write_text(json.dumps(patterns, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("✅ Saved patterns → %s", out_path)
    else:
        logger.info("(preview mode — not saving to disk)")

    return patterns


def _print_preview(patterns: dict) -> None:
    """Print a human-readable summary of the patterns."""
    print("\n" + "═" * 70)
    print(f"  PATTERN ANALYSIS: {patterns.get('subject', '?')} Paper {patterns.get('paper', '?')}")
    print(f"  Years analysed: {patterns.get('years_analysed', [])}")
    meta = patterns.get("_meta", {})
    print(f"  Train: {meta.get('train_count',0)} MS | Holdout: {meta.get('holdout_count',0)} MS")
    if "validation" in patterns:
        val = patterns["validation"]
        acc = val.get("mean_accuracy")
        if acc is not None:
            print(f"  Cross-validation accuracy: {acc:.0%}")
    print("═" * 70)
    for qp in patterns.get("question_patterns", []):
        qid = qp.get("question_id", "?")
        qtype = qp.get("type", "?")
        marks = qp.get("typical_marks", "?")
        marks_range = qp.get("marks_range", [])
        consistent = qp.get("consistent_across_years", False)
        confidence = qp.get("confidence", 0)
        formula = qp.get("full_mark_formula", "")
        print(f"\n  [{qid}]  type={qtype}  marks={marks}  range={marks_range}  "
              f"consistent={'✓' if consistent else '~'}  conf={confidence:.2f}")
        print(f"    formula: {formula}")
        for pt in qp.get("common_mark_points", [])[:3]:
            print(f"    • {pt}")
        if qp.get("examiner_tendencies"):
            print(f"    tendency: {qp['examiner_tendencies'][:100]}")
    print()

    if "validation" in patterns:
        print("\n  CROSS-VALIDATION RESULTS:")
        for vr in patterns["validation"].get("results", []):
            print(f"    {vr.get('holdout_label','?')}: "
                  f"accuracy={vr.get('overall_accuracy',0):.0%}")
            print(f"    → {vr.get('summary','')[:120]}")
    print("═" * 70 + "\n")


async def run_all(validate: bool = True) -> None:
    """Run pattern analysis for ALL subjects and ALL paper components."""
    manifest = _load_manifest()
    tasks = []
    for subject_key in ALL_SUBJECT_KEYS:
        if subject_key not in manifest:
            continue
        ms_entries = manifest[subject_key].get("mark_schemes", [])
        # Collect unique paper components (first digit only)
        components = set()
        for e in ms_entries:
            p = e.get("paper")
            if p is not None:
                components.add(p // 10)
        for comp in sorted(components):
            tasks.append((subject_key, comp))

    logger.info("Will process %d subject+paper combos", len(tasks))
    for subject_key, paper in tasks:
        logger.info("── Processing %s paper %s ──", subject_key, paper)
        await analyse_subject_paper(subject_key, paper, validate=validate)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CIE mark-scheme pattern files from historical data."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Process ALL subjects and ALL papers",
    )
    group.add_argument(
        "--subject",
        metavar="SUBJ",
        help='Subject name or key (e.g. "economics", "physics_9702")',
    )
    parser.add_argument(
        "--paper",
        type=int,
        metavar="N",
        help="Paper component number (e.g. 2 for Paper 2x). Omit to process all papers for the subject.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip cross-validation step (faster, cheaper)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print results to stdout without saving to disk",
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    validate = not args.no_validate

    if args.all:
        await run_all(validate=validate)
        return

    if not args.subject:
        print("Error: specify --subject or --all")
        sys.exit(1)

    subject_key = _SUBJECT_MAP.get(args.subject.lower().strip(), args.subject)

    if args.paper is not None:
        # Single subject + paper
        patterns = await analyse_subject_paper(
            subject=subject_key,
            paper_number=args.paper,
            validate=validate,
            preview_only=args.preview,
        )
        if patterns:
            _print_preview(patterns)
    else:
        # All papers for this subject
        manifest = _load_manifest()
        if subject_key not in manifest:
            logger.error("Subject '%s' not found in manifest", subject_key)
            sys.exit(1)
        ms_entries = manifest[subject_key].get("mark_schemes", [])
        components = sorted(set(
            e["paper"] // 10
            for e in ms_entries
            if e.get("paper") is not None
        ))
        logger.info("Processing %s papers: %s", subject_key, components)
        for comp in components:
            patterns = await analyse_subject_paper(
                subject=subject_key,
                paper_number=comp,
                validate=validate,
                preview_only=args.preview,
            )
            if patterns:
                _print_preview(patterns)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
