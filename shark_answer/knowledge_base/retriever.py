"""Query-time knowledge base retriever.

Reads manifest.json once (cached), then loads pre-extracted .txt files on demand.
No vector DB, no preprocessing, no API calls at retrieval time.

Typical usage in a pipeline:
    from shark_answer.knowledge_base.retriever import get_context

    result = get_context(subject="Physics", paper_number=2)
    kb_context = result["context"]   # inject into model system prompt
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── KB root directory ──────────────────────────────────────────────────────
_DEFAULT_KB_DIR = Path.home() / "Desktop/BTC/AI/shark_answer_kb"
KB_DIR: Path = Path(
    os.getenv("SHARK_ANSWER_KB_DIR", str(_DEFAULT_KB_DIR))
).expanduser()

# ── Subject rules directory (ships with the package, not in KB_DIR) ────────
# Located at shark_answer/knowledge_base/subject_rules/{subject_key}_rules.txt
_RULES_DIR: Path = Path(__file__).parent / "subject_rules"

# ── Subject normalizer: any common name / code → manifest key ─────────────
_SUBJECT_MAP: dict[str, str] = {
    # Display names (lowercase)
    "physics":              "physics_9702",
    "chemistry":            "chemistry_9701",
    "math":                 "math_9709",
    "mathematics":          "math_9709",
    "further math":         "further_math_9231",
    "further mathematics":  "further_math_9231",
    "economics":            "economics_9708",
    "computer science":     "cs_9618",
    "cs":                   "cs_9618",
    "chinese":              "chinese_8238",
    # CIE syllabus codes
    "9702":  "physics_9702",
    "9701":  "chemistry_9701",
    "9709":  "math_9709",
    "9231":  "further_math_9231",
    "9708":  "economics_9708",
    "9618":  "cs_9618",
    "8238":  "chinese_8238",
    # Manifest keys (identity)
    "physics_9702":       "physics_9702",
    "chemistry_9701":     "chemistry_9701",
    "math_9709":          "math_9709",
    "further_math_9231":  "further_math_9231",
    "economics_9708":     "economics_9708",
    "cs_9618":            "cs_9618",
    "chinese_8238":       "chinese_8238",
}

# Subjects that carry confidential instructions (practical papers)
_CI_SUBJECTS: frozenset[str] = frozenset({"physics_9702", "chemistry_9701"})

# ── Token budget ───────────────────────────────────────────────────────────
# Rough estimate: 1 token ≈ 4 chars → 30 000 tokens ≈ 120 000 chars
_TOKEN_BUDGET_CHARS: int = 120_000
# Minimum budget per section so we don't add empty slices
_MIN_SECTION_CHARS: int = 500

# Per-item character caps — balances budget across all section types.
# Rough allocations (120 k total):
#   5 MS  × 12 k = 60 k
#   3 ER  × 12 k = 36 k
#   3 GT  ×  3 k =  9 k
#   1 syl         ≤ 15 k   (capped separately in syllabus block)
#   Total target  ≤ 120 k ✓
_MAX_MS_CHARS:  int = 12_000   # per mark scheme  (~3 k tokens)
_MAX_ER_CHARS:  int = 12_000   # per examiner report
_MAX_GT_CHARS:  int =  3_000   # per grade threshold doc

# ── Manifest (lazy-loaded, module-level singleton) ─────────────────────────
_manifest: dict | None = None


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        mpath = KB_DIR / "manifest.json"
        if mpath.exists():
            try:
                _manifest = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load manifest.json: %s", exc)
                _manifest = {}
        else:
            logger.warning("manifest.json not found at %s — KB disabled", mpath)
            _manifest = {}
    return _manifest


def _normalize_subject(subject: str) -> str | None:
    """Map any subject string to its manifest key, or None if unknown."""
    return _SUBJECT_MAP.get(subject.strip().lower())


def _read_txt(subject_key: str, category: str, filename: str) -> str | None:
    """Read the .txt companion of a PDF file.

    Falls back gracefully: returns None if the .txt hasn't been extracted yet
    (run extract_text.py to generate them).
    """
    txt_path = (
        KB_DIR / subject_key / category / Path(filename).with_suffix(".txt")
    )
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", txt_path, exc)
    else:
        # Only warn once per missing file to avoid log spam
        logger.debug("No .txt for %s/%s/%s — run extract_text.py", subject_key, category, filename)
    return None


def _matches_paper(entry_paper: int | None, paper_number: int | None) -> bool:
    """True if an entry's paper number belongs to the requested paper component.

    Rules:
        paper_number=None  → match everything (no filter)
        paper_number=2     → match 21, 22, 23, 24, 25 (any variant of component 2)
        paper_number=21    → exact match only
    """
    if paper_number is None:
        return True
    if entry_paper is None:
        return False
    if paper_number < 10:
        # Component-level: first digit must match
        return entry_paper // 10 == paper_number
    # Exact match (caller already knows the specific variant)
    return entry_paper == paper_number


def get_context(
    subject: str,
    paper_number: Optional[int] = None,
    question_text: str = "",
    exclude_year: Optional[int] = None,
) -> dict:
    """Retrieve knowledge base context for injection into a model prompt.

    Args:
        subject:       Subject name or code, e.g. "Physics", "9702", "physics_9702"
        paper_number:  CIE paper component (1–5). None = match all papers.
        question_text: The question being answered (reserved for future topic filtering).
        exclude_year:  If set, mark schemes/examiner reports from this year are excluded.
                       Use this for fair benchmarking: set exclude_year=paper_year so the
                       model cannot see the mark scheme it is being tested against.

    Returns a dict with:
        context  (str):  Ready-to-inject context block with section headers.
        included (dict): Metadata listing what was included.
        char_count (int): Approximate size in characters (~chars/4 = tokens).
    """
    manifest = _load_manifest()

    subject_key = _normalize_subject(subject)
    if not subject_key or subject_key not in manifest:
        logger.info("No KB data for subject '%s' — returning empty context", subject)
        return {"context": "", "included": {}, "char_count": 0}

    data = manifest[subject_key]
    budget = _TOKEN_BUDGET_CHARS
    sections: list[str] = []
    included: dict = {
        "subject_rules":      False,
        "mark_schemes":       [],
        "examiner_reports":   [],
        "grade_thresholds":   [],
        "syllabus":           [],
        "ci":                 [],
        "subject_summary":    False,
    }

    # ── -1. Subject rules (highest priority — must appear first in context) ──
    # Loaded from shark_answer/knowledge_base/subject_rules/{subject_key}_rules.txt
    # These encode exact CIE terminology that overrides any model defaults.
    rules_path = _RULES_DIR / f"{subject_key}_rules.txt"
    if rules_path.exists():
        try:
            rules_text = rules_path.read_text(encoding="utf-8", errors="replace").strip()
            if rules_text and budget > _MIN_SECTION_CHARS:
                block = f"=== SUBJECT TERMINOLOGY RULES (HIGHEST PRIORITY) ===\n{rules_text}"
                sections.append(block)
                budget -= len(block)
                included["subject_rules"] = True
                logger.debug("Loaded subject rules from %s (%d chars)", rules_path, len(rules_text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read subject rules %s: %s", rules_path, exc)

    # ── 0. Subject-level examiner summary (pre-built by build_summaries.py) ─
    summary_path = KB_DIR / subject_key / "subject_summary.txt"
    if summary_path.exists():
        try:
            summary_text = summary_path.read_text(encoding="utf-8", errors="replace").strip()
            if summary_text and budget > _MIN_SECTION_CHARS:
                block = f"=== EXAMINER INSIGHT SUMMARY ===\n{summary_text}"
                sections.append(block)
                budget -= len(block)
                included["subject_summary"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read subject_summary.txt: %s", exc)

    # ── 1. Mark schemes: matching paper, most recent 5 years ──────────────
    # Reserve budget slots so ERs / GTs can always be included:
    #   mark schemes  → up to 60 000 chars (5 × _MAX_MS_CHARS)
    #   examiner rpts → up to 30 000 chars (2 × _MAX_ER_CHARS)
    #   grade thresh  → up to  9 000 chars (3 × _MAX_GT_CHARS)
    #   syllabus      → remainder
    ms_entries = data.get("mark_schemes", [])
    ms_matching = [
        e for e in ms_entries
        if _matches_paper(e.get("paper"), paper_number)
        and (exclude_year is None or e.get("year") != exclude_year)
    ]
    # Sort newest-first, take up to 5
    ms_recent = sorted(ms_matching, key=lambda e: e.get("year", 0), reverse=True)[:5]

    ms_loaded: list[tuple[int, str, str]] = []  # (year, section_label, text)
    for entry in ms_recent:
        year    = entry.get("year", "?")
        session = (entry.get("session") or "unknown").title()
        paper   = entry.get("paper", "?")
        fname   = entry.get("renamed") or entry.get("original", "")
        text    = _read_txt(subject_key, "mark_schemes", fname)
        if text:
            # Cap each mark scheme to avoid a single doc eating all budget
            if len(text) > _MAX_MS_CHARS:
                text = text[:_MAX_MS_CHARS] + "\n[...mark scheme truncated]"
            label = f"[Year {year} {session}, Paper {paper} Mark Scheme]"
            ms_loaded.append((year, label, text))

    if ms_loaded and budget > _MIN_SECTION_CHARS:
        # Try all loaded, then trim to 3 if over remaining budget
        for keep in (len(ms_loaded), min(3, len(ms_loaded))):
            subset = ms_loaded[:keep]
            body = "\n\n".join(f"{lbl}\n{txt}" for _, lbl, txt in subset)
            ms_section = f"=== MARK SCHEME REFERENCE ===\n{body}"
            if len(ms_section) <= budget:
                sections.append(ms_section)
                budget -= len(ms_section)
                included["mark_schemes"] = [lbl for _, lbl, _ in subset]
                break

    # ── 2. Examiner reports: 3 most recent (respects exclude_year) ────────
    er_entries = sorted(
        [
            e for e in data.get("examiner_reports", [])
            if exclude_year is None or e.get("year") != exclude_year
        ],
        key=lambda e: e.get("year", 0),
        reverse=True,
    )[:3]

    er_loaded: list[tuple[int, str, str]] = []
    for entry in er_entries:
        year    = entry.get("year", "?")
        session = (entry.get("session") or "unknown").title()
        fname   = entry.get("renamed") or entry.get("original", "")
        text    = _read_txt(subject_key, "examiner_reports", fname)
        if text:
            if len(text) > _MAX_ER_CHARS:
                text = text[:_MAX_ER_CHARS] + "\n[...examiner report truncated]"
            label = f"[Year {year} {session} Examiner Report]"
            er_loaded.append((year, label, text))

    if er_loaded and budget > _MIN_SECTION_CHARS:
        for keep in (len(er_loaded), min(2, len(er_loaded))):
            subset = er_loaded[:keep]
            body = "\n\n".join(f"{lbl}\n{txt}" for _, lbl, txt in subset)
            er_section = f"=== EXAMINER REPORT EXCERPTS ===\n{body}"
            if len(er_section) <= budget:
                sections.append(er_section)
                budget -= len(er_section)
                included["examiner_reports"] = [lbl for _, lbl, _ in subset]
                break

    # ── 3. Grade thresholds: 3 most recent (respects exclude_year) ────────
    gt_entries = sorted(
        [
            e for e in data.get("grade_thresholds", [])
            if exclude_year is None or e.get("year") != exclude_year
        ],
        key=lambda e: e.get("year", 0),
        reverse=True,
    )[:3]

    gt_loaded: list[tuple[int, str, str]] = []
    for entry in gt_entries:
        year    = entry.get("year", "?")
        session = (entry.get("session") or "unknown").title()
        fname   = entry.get("renamed") or entry.get("original", "")
        text    = _read_txt(subject_key, "grade_thresholds", fname)
        if text:
            if len(text) > _MAX_GT_CHARS:
                text = text[:_MAX_GT_CHARS] + "\n[...thresholds truncated]"
            label = f"[Year {year} {session} Grade Thresholds]"
            gt_loaded.append((year, label, text))

    if gt_loaded and budget > _MIN_SECTION_CHARS:
        body = "\n\n".join(f"{lbl}\n{txt}" for _, lbl, txt in gt_loaded)
        gt_section = f"=== GRADE THRESHOLDS ===\n{body}"
        if len(gt_section) <= budget:
            sections.append(gt_section)
            budget -= len(gt_section)
            included["grade_thresholds"] = [lbl for _, lbl, _ in gt_loaded]

    # ── 4. Syllabus: most recent document ─────────────────────────────────
    syl_entries = sorted(
        data.get("syllabus", []),
        key=lambda e: e.get("year", 0),
        reverse=True,
    )[:1]

    for entry in syl_entries:
        year  = entry.get("year", "?")
        fname = entry.get("renamed") or entry.get("original", "")
        text  = _read_txt(subject_key, "syllabus", fname)
        if text and budget > _MIN_SECTION_CHARS:
            # Truncate syllabus if it would eat all remaining budget
            max_syl = max(_MIN_SECTION_CHARS, budget - 4_000)
            if len(text) > max_syl:
                text = text[:max_syl] + "\n[...syllabus truncated for token budget]"
            block = f"=== CURRENT SYLLABUS ===\n[Year {year} Syllabus]\n{text}"
            if len(block) <= budget:
                sections.append(block)
                budget -= len(block)
                included["syllabus"] = [str(year)]

    # ── 5. Confidential instructions (Physics/Chemistry practical papers) ──
    if subject_key in _CI_SUBJECTS and budget > _MIN_SECTION_CHARS:
        ci_entries = [
            e for e in data.get("confidential_instructions", [])
            if _matches_paper(e.get("paper"), paper_number)
        ]
        ci_recent = sorted(ci_entries, key=lambda e: e.get("year", 0), reverse=True)[:2]

        ci_loaded: list[tuple[int, str, str]] = []
        for entry in ci_recent:
            year    = entry.get("year", "?")
            session = (entry.get("session") or "unknown").title()
            paper   = entry.get("paper", "?")
            fname   = entry.get("renamed") or entry.get("original", "")
            text    = _read_txt(subject_key, "confidential_instructions", fname)
            if text:
                label = f"[Year {year} {session}, Paper {paper} Confidential Instructions]"
                ci_loaded.append((year, label, text))

        if ci_loaded:
            body = "\n\n".join(f"{lbl}\n{txt}" for _, lbl, txt in ci_loaded)
            ci_section = f"=== CONFIDENTIAL INSTRUCTIONS ===\n{body}"
            if len(ci_section) <= budget:
                sections.append(ci_section)
                budget -= len(ci_section)
                included["ci"] = [lbl for _, lbl, _ in ci_loaded]

    # ── Assemble final context ─────────────────────────────────────────────
    context = "\n\n".join(sections)

    included_keys = [k for k, v in included.items() if v]
    logger.info(
        "KB context for '%s' paper=%s: %d chars (~%d tokens), sections: %s",
        subject_key,
        paper_number,
        len(context),
        len(context) // 4,
        included_keys,
    )

    return {
        "context":    context,
        "included":   included,
        "char_count": len(context),
    }


# ── Convenience wrapper used in pipeline prompts ───────────────────────────
_KB_CONTEXT_HEADER = """\
You have access to official CIE mark schemes, examiner reports, and grade \
thresholds below. Use these to ensure your answer hits EVERY required mark \
point. Pay close attention to what examiners explicitly reward and penalise.

{context}

Use the mark scheme reference above to guide your answer. \
Write ONLY what earns marks — nothing more.\
"""


def build_prompt_context(
    subject: str,
    paper_number: Optional[int] = None,
    exclude_year: Optional[int] = None,
) -> str:
    """Return a ready-to-prepend context block for model system prompts.

    Args:
        subject:      Subject name or code.
        paper_number: CIE paper component. None = all papers.
        exclude_year: If set, exclude this year's mark schemes from context.
                      Pass the paper year for fair (non-leaky) evaluation.

    Returns an empty string if no KB data is available (safe to inject regardless).
    """
    result = get_context(subject=subject, paper_number=paper_number, exclude_year=exclude_year)
    if not result["context"]:
        return ""
    return _KB_CONTEXT_HEADER.format(context=result["context"])
