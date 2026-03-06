"""Mark scheme predictor — predicts the likely mark scheme for a new question
at query time, using pre-built pattern files from build_patterns.py.

Usage:
    from shark_answer.knowledge_base.predictor import predict_mark_scheme

    prediction = predict_mark_scheme(
        subject="economics",
        paper=2,
        question_text="Assess the extent to which a fall in interest rates ...",
        marks=8,
    )
    print(prediction["predicted_ms_block"])  # ready-to-inject text
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
PATTERNS_DIR = _SCRIPT_DIR / "patterns"

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

# ── Type → prompt skeleton ────────────────────────────────────────────────────
# These are fill-in-the-blank templates that get enriched with pattern data.
_TYPE_SKELETONS: dict[str, str] = {
    "calculation": (
        "Award marks for: correct method/formula (if shown) + correct numerical answer.\n"
        "Accept alternative valid methods if they arrive at the correct answer.\n"
        "Penalise: wrong units, missing %, wrong rounding (only if explicit)."
    ),
    "data_extract": (
        "Credit: accurate reading of the stated figure from the data, with correct units.\n"
        "Comparative questions require both data points + direction of change.\n"
        "1 mark per valid accurate description unless stated otherwise."
    ),
    "define": (
        "Award marks for: correct definition of the key term.\n"
        "Do NOT credit: just listing characteristics without defining the term.\n"
        "Accept equivalent wording if the meaning is clear."
    ),
    "explain_single": (
        "Award marks for: identifying the correct mechanism + brief explanation of link.\n"
        "Penalise: vague answers that do not show understanding (e.g. 'supply and demand').\n"
        "Each separate valid explanation earns 1 mark; further development may earn the 2nd mark."
    ),
    "explain_multi": (
        "Award marks for: up to [N] valid explanations, each earning 1-2 marks.\n"
        "Full marks require covering distinct points (not restatements).\n"
        "Penalise: repetition of the same point in different words."
    ),
    "assess_analyse": (
        "Up to [analysis_marks] for analysis: at least 2 distinct explained reasons/effects.\n"
        "Up to [eval_marks] for evaluation: a reasoned judgement/conclusion.\n"
        "Award evaluation marks ONLY if analysis marks have been earned.\n"
        "Penalise: missing conclusion for evaluation marks."
    ),
    "essay_knowledge": (
        "Level-marked (AO1 Knowledge + AO2 Analysis).\n"
        "Level 3 (top marks): detailed knowledge, fully developed explanations, accurate diagrams.\n"
        "Level 2: some knowledge with limited development.\n"
        "Level 1: limited knowledge, largely descriptive.\n"
        "Diagrams: must be correctly labelled and explained to earn diagram marks."
    ),
    "essay_evaluation": (
        "Level-marked (AO3 Evaluation).\n"
        "Top level: justified conclusion, developed reasoning, considers counter-arguments.\n"
        "Middle level: simple evaluative comment with little support.\n"
        "Bottom level: no valid conclusion or purely descriptive evaluation.\n"
        "Must reach a clear, reasoned judgement to score top level marks."
    ),
    "diagram": (
        "Award marks for: correct axes labels, correct shape/direction of curve(s), "
        "correct equilibrium point(s), correct shift if required.\n"
        "Each correct element earns 1 mark. Penalise: missing labels, wrong direction."
    ),
}

# ── Pattern file loading (cached) ─────────────────────────────────────────────
_pattern_cache: dict[str, dict] = {}


def _load_patterns(subject_key: str, paper: int) -> dict | None:
    """Load the pre-built pattern file for a subject+paper.  Cached after first read."""
    cache_key = f"{subject_key}_paper{paper}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]

    pattern_path = PATTERNS_DIR / f"{subject_key}_paper{paper}_patterns.json"
    if not pattern_path.exists():
        logger.debug("No pattern file for %s paper %s at %s", subject_key, paper, pattern_path)
        _pattern_cache[cache_key] = None
        return None

    try:
        data = json.loads(pattern_path.read_text(encoding="utf-8"))
        _pattern_cache[cache_key] = data
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load pattern file %s: %s", pattern_path, exc)
        _pattern_cache[cache_key] = None
        return None


# ── Question type classification ──────────────────────────────────────────────

def _classify_question(question_text: str, marks: int) -> str:
    """Heuristic question-type classifier.

    Used when no pattern match is available (fallback).
    """
    q = question_text.lower().strip()

    # Explicit keywords
    if re.search(r"\bcalculat\b", q):
        return "calculation"
    if re.search(r"\bdefin\b|\bwhat is meant by\b|\bstate the meaning\b", q):
        return "define"
    if re.search(r"\bdiagram\b|\bdraw\b|\bsketch\b", q):
        return "diagram"

    # Mark-count heuristics
    if marks <= 2:
        if re.search(r"\bidentif\b|\bstate\b|\bname\b|\blist\b|\bfrom\b.*\bfig\b", q):
            return "data_extract"
        return "explain_single"

    if marks <= 4:
        if re.search(r"\bexplain\b|\bdescribe\b", q):
            return "explain_multi"
        return "assess_analyse"

    if marks <= 8:
        if re.search(r"\bassess\b|\banalyse\b|\bevaluate\b|\bdiscuss\b|\bconsider\b", q):
            return "assess_analyse"
        return "essay_knowledge"

    # High marks (9+)
    if re.search(r"\bto what extent\b|\bdo you agree\b|\bjustif\b", q):
        return "essay_evaluation"
    return "essay_knowledge"


def _find_best_pattern_match(
    question_text: str,
    marks: int,
    q_type: str,
    question_patterns: list[dict],
) -> dict | None:
    """Find the most relevant pattern from the library for this question."""
    if not question_patterns:
        return None

    # First try: exact type + exact marks
    exact = [
        p for p in question_patterns
        if p.get("type") == q_type and p.get("typical_marks") == marks
    ]
    if exact:
        return exact[0]

    # Second try: same type, close marks (within ±1)
    close = [
        p for p in question_patterns
        if p.get("type") == q_type
        and abs(p.get("typical_marks", 0) - marks) <= 1
    ]
    if close:
        # Pick the one whose marks_range contains `marks`
        for p in close:
            lo, hi = p.get("marks_range", [0, 100])
            if lo <= marks <= hi:
                return p
        return close[0]

    # Third try: same type regardless of marks
    same_type = [p for p in question_patterns if p.get("type") == q_type]
    if same_type:
        return same_type[0]

    return None


# ── Main public function ───────────────────────────────────────────────────────

def predict_mark_scheme(
    subject: str,
    paper: int,
    question_text: str,
    marks: int,
    question_id: Optional[str] = None,
) -> dict:
    """Predict the likely mark scheme for a new question.

    Args:
        subject:       Subject name or manifest key (e.g. "economics")
        paper:         CIE paper component number (e.g. 2 for Paper 2x)
        question_text: The full question text
        marks:         Marks available for this question
        question_id:   Optional question identifier (e.g. "1(c)"); used for
                       direct pattern lookup if provided

    Returns a dict with:
        predicted_ms_block  (str):  Ready-to-inject text for the model prompt
        question_type       (str):  Inferred question type
        confidence          (float): 0-1 confidence score
        pattern_source      (str):  Where the prediction came from
        matched_pattern     (dict): The matched pattern entry (if any)
    """
    subject_key = _SUBJECT_MAP.get(subject.lower().strip(), subject)
    paper_component = paper if paper >= 10 else paper  # keep as component

    # Load patterns
    patterns_data = _load_patterns(subject_key, paper_component)

    # Classify question type
    q_type = _classify_question(question_text, marks)

    # Skeleton text (fallback)
    skeleton = _TYPE_SKELETONS.get(q_type, "Award marks for correct and relevant responses.")

    matched_pattern: dict | None = None
    confidence: float = 0.3  # baseline
    pattern_source = "heuristic"

    if patterns_data:
        qps = patterns_data.get("question_patterns", [])

        # If question_id given, try direct lookup first
        if question_id:
            qid_clean = question_id.strip().lower()
            for p in qps:
                if p.get("question_id", "").lower() == qid_clean:
                    matched_pattern = p
                    confidence = p.get("confidence", 0.7)
                    pattern_source = f"direct_match:{question_id}"
                    break

        # Otherwise, find best structural match
        if not matched_pattern:
            matched_pattern = _find_best_pattern_match(question_text, marks, q_type, qps)
            if matched_pattern:
                confidence = matched_pattern.get("confidence", 0.6) * 0.85  # slight penalty for indirect match
                pattern_source = f"structural_match:{matched_pattern.get('question_id','?')}"

    # Build the prediction block
    lines: list[str] = []
    lines.append(f"[Predicted Mark Scheme — {q_type.replace('_',' ').title()} — {marks} marks]")
    lines.append("")

    if matched_pattern:
        # Use pattern data
        formula = matched_pattern.get("full_mark_formula", "")
        if formula:
            lines.append(f"Mark formula: {formula}")

        mark_structure = matched_pattern.get("mark_structure", "")
        if mark_structure:
            lines.append(f"Structure: {mark_structure}")

        common_pts = matched_pattern.get("common_mark_points", [])
        if common_pts:
            lines.append("")
            lines.append("Typical mark points (from historical mark schemes):")
            for pt in common_pts[:6]:
                lines.append(f"  • {pt}")

        penalties = matched_pattern.get("common_penalties", [])
        if penalties:
            lines.append("")
            lines.append("Common penalties:")
            for pen in penalties[:3]:
                lines.append(f"  ✗ {pen}")

        tendencies = matched_pattern.get("examiner_tendencies", "")
        if tendencies:
            lines.append("")
            lines.append(f"Examiner tendencies: {tendencies}")

        years = matched_pattern.get("years_analysed", [])
        if years:
            lines.append(f"\n(Pattern derived from years: {years})")
    else:
        # Fallback: type-specific skeleton
        lines.append(f"No historical pattern available — applying {q_type} template:")
        lines.append("")
        # Fill in marks placeholder
        body = skeleton.replace("[N]", str(marks))
        if q_type == "assess_analyse":
            # Split marks: 2/3 analysis, 1/3 evaluation
            analysis_marks = max(1, marks - (marks // 3))
            eval_marks = marks - analysis_marks
            body = body.replace("[analysis_marks]", str(analysis_marks))
            body = body.replace("[eval_marks]", str(eval_marks))
        lines.append(body)

    predicted_ms_block = "\n".join(lines)

    return {
        "predicted_ms_block": predicted_ms_block,
        "question_type":      q_type,
        "confidence":         round(confidence, 3),
        "pattern_source":     pattern_source,
        "matched_pattern":    matched_pattern,
    }


def predict_paper_mark_schemes(
    subject: str,
    paper: int,
    questions: list[dict],
) -> list[dict]:
    """Predict mark schemes for all questions in a paper.

    Args:
        subject:   Subject name or key
        paper:     Paper component number
        questions: List of dicts with keys: text (str), marks (int), id (str, optional)

    Returns list of prediction dicts (same order as input questions).
    """
    return [
        predict_mark_scheme(
            subject=subject,
            paper=paper,
            question_text=q.get("text", ""),
            marks=q.get("marks", 4),
            question_id=q.get("id"),
        )
        for q in questions
    ]


def build_prediction_context(
    subject: str,
    paper: int,
    question_text: str,
    marks: int,
    question_id: Optional[str] = None,
) -> str:
    """Return a ready-to-inject prediction context block for model prompts.

    Returns an empty string if no patterns available (safe to use unconditionally).
    """
    pred = predict_mark_scheme(subject, paper, question_text, marks, question_id)
    block = pred.get("predicted_ms_block", "")
    if not block:
        return ""

    conf = pred.get("confidence", 0)
    source = pred.get("pattern_source", "")

    header = (
        f"=== PREDICTED MARK SCHEME (confidence: {conf:.0%}, source: {source}) ===\n"
        "This prediction is based on historical CIE marking patterns for this paper type.\n"
        "Use it as a guide for structure — the actual question content will differ.\n\n"
    )
    return header + block + "\n"
