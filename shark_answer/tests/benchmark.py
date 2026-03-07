#!/usr/bin/env python3
"""Shark Answer automated benchmark.

Tests the full pipeline against real CIE past papers from the knowledge base.

Parts:
  1. Load QP PDF from KB  → vision extraction of questions
  2. Run each question through the appropriate pipeline (A / B / C)
  3. Score each best answer against the official mark scheme via Claude
  4. Track wall-clock time per stage and API cost per paper
  5. Generate a tabular report

Usage:
    python -m shark_answer.tests.benchmark                     # all 7 subjects
    python -m shark_answer.tests.benchmark --quick             # Economics + Physics only
    python -m shark_answer.tests.benchmark --subject economics
    python -m shark_answer.tests.benchmark --versions 3
    python -m shark_answer.tests.benchmark --max-questions 3   # limit Qs per paper
    python -m shark_answer.tests.benchmark --output report.txt
    python -m shark_answer.tests.benchmark --verbose           # debug logging
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
_DEFAULT_KB_DIR = Path.home() / "Desktop/BTC/AI/shark_answer_kb"
_RESULTS_DIR = Path(__file__).parent / "benchmark_results"

logger = logging.getLogger("benchmark")


# ── Benchmark paper definitions ─────────────────────────────────────────────────
# qp_file / ms_file = None  →  resolved dynamically from manifest
_ALL_PAPERS: list[dict] = [
    {
        "subject_key":   "economics",
        "manifest_key":  "economics_9708",
        "qp_file":       "2024_june_qp_21.pdf",
        "ms_file":       "2024_june_ms_21.pdf",
        "paper_number":  21,
    },
    {
        "subject_key":   "physics",
        "manifest_key":  "physics_9702",
        "qp_file":       "2024_june_qp_21.pdf",
        "ms_file":       "2024_june_ms_21.pdf",
        "paper_number":  21,
    },
    {
        "subject_key":   "chemistry",
        "manifest_key":  "chemistry_9701",
        "qp_file":       "2024_june_qp_21.pdf",
        "ms_file":       "2024_june_ms_21.pdf",
        "paper_number":  21,
    },
    {
        "subject_key":   "math",
        "manifest_key":  "math_9709",
        "qp_file":       "2024_june_qp_12.pdf",
        "ms_file":       "2024_june_ms_12.pdf",
        "paper_number":  12,
    },
    {
        "subject_key":   "further_math",
        "manifest_key":  "further_math_9231",
        "qp_file":       "2024_june_qp_11.pdf",
        "ms_file":       "2024_june_ms_11.pdf",
        "paper_number":  11,
    },
    {
        "subject_key":   "computer_science",
        "manifest_key":  "cs_9618",
        "qp_file":       "2024_june_qp_21.pdf",
        "ms_file":       "2024_june_ms_21.pdf",
        "paper_number":  21,
    },
    # NOTE: Chinese (8238) is intentionally excluded from the default run list.
    # All 8238 papers are multiple-choice (Paper 1 = Listening, Paper 2 = Reading).
    # Pipeline B (Essay) is not appropriate for MCQ format.  A dedicated MCQ
    # pipeline is required before Chinese can be benchmarked.
    # To run Chinese manually: --subject chinese
    # {
    #     "subject_key":   "chinese",
    #     "manifest_key":  "chinese_8238",
    #     "qp_file":       None,
    #     "ms_file":       None,
    #     "paper_number":  None,
    # },
]

_QUICK_SUBJECTS = {"economics", "physics"}


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class TimingBreakdown:
    extraction_s: float = 0.0
    solving_s:    float = 0.0
    scoring_s:    float = 0.0

    @property
    def total_s(self) -> float:
        return self.extraction_s + self.solving_s + self.scoring_s


@dataclass
class QuestionResult:
    number:        str
    text:          str
    marks_total:   int
    marks_achieved: int
    score_str:     str           # "8/10"
    grade_estimate: str          # "A*", "A", …, "U"
    best_answer:   str
    pipeline:      str
    providers_ok:  list[str] = field(default_factory=list)
    providers_fail: list[str] = field(default_factory=list)
    timing_s:      float = 0.0
    errors:        list[str] = field(default_factory=list)


@dataclass
class PaperResult:
    subject_key:  str
    manifest_key: str
    qp_file:      str
    ms_file:      str
    paper_number: Optional[int]
    questions:    list[QuestionResult] = field(default_factory=list)
    timing:       TimingBreakdown = field(default_factory=TimingBreakdown)
    cost_usd:     float = 0.0
    error:        str = ""        # paper-level error (e.g. PDF not found)

    # ── Aggregates (exclude SCORE_FAILED questions from totals) ─────
    @property
    def scored_questions(self) -> list:
        return [q for q in self.questions if q.grade_estimate != "SCORE_FAILED"]

    @property
    def total_marks(self) -> int:
        return sum(q.marks_total for q in self.scored_questions)

    @property
    def total_achieved(self) -> int:
        return sum(q.marks_achieved for q in self.scored_questions)

    @property
    def failed_scoring_count(self) -> int:
        return len(self.questions) - len(self.scored_questions)

    @property
    def score_pct(self) -> float:
        return (self.total_achieved / self.total_marks * 100) if self.total_marks else 0.0

    @property
    def grade(self) -> str:
        pct = self.score_pct
        if pct >= 90: return "A*"
        if pct >= 80: return "A"
        if pct >= 70: return "B"
        if pct >= 60: return "C"
        if pct >= 50: return "D"
        if pct >= 40: return "E"
        return "U"

    @property
    def all_providers_ok(self) -> list[str]:
        ok: set[str] = set()
        for q in self.questions:
            ok.update(q.providers_ok)
        return sorted(ok)

    @property
    def all_providers_fail(self) -> list[str]:
        fail: set[str] = set()
        for q in self.questions:
            fail.update(q.providers_fail)
        return sorted(fail)


# ── Standalone scorer (no app globals) ─────────────────────────────────────────

# Scorer fallback chain:
#   1. Claude Haiku  — ~60× cheaper than Opus, fast for simple JSON scoring
#   2. GPT-4o        — reliable fallback when Claude API is overloaded
#   3. Gemini 2.5    — final fallback if both Claude and OpenAI are unavailable
#
# The chain is tried in order; the first successful response wins.

# Module-level cached Haiku scorer (shared Claude API key, Haiku model).
_haiku_scorer_cache: Optional[object] = None


def _get_haiku_scorer(registry) -> object | None:
    """Return a ClaudeProvider using the cheap Haiku model, or None if not configured."""
    global _haiku_scorer_cache
    if _haiku_scorer_cache is not None:
        return _haiku_scorer_cache

    from shark_answer.config import ModelProvider
    from shark_answer.providers.claude_provider import ClaudeProvider

    claude_cfg = registry.config.models.get(ModelProvider.CLAUDE)
    if not claude_cfg or not claude_cfg.api_key:
        return None

    _haiku_scorer_cache = ClaudeProvider(
        api_key=claude_cfg.api_key,
        model_name="claude-haiku-4-5-20251001",
    )
    return _haiku_scorer_cache


def _build_scorer_chain(registry) -> list[object]:
    """Build an ordered list of scorer provider instances for fallback scoring.

    Returns [Haiku, GPT-4o, Gemini] — skipping any provider not configured.
    Callers should try each in sequence and use the first successful response.
    """
    from shark_answer.config import ModelProvider

    scorers: list[object] = []

    # 1. Haiku — cheapest, primary scorer
    haiku = _get_haiku_scorer(registry)
    if haiku:
        scorers.append(("Claude Haiku", haiku))

    # 2. GPT-4o — reliable fallback
    gpt4o = registry.get(ModelProvider.GPT4O)
    if gpt4o:
        scorers.append(("GPT-4o", gpt4o))

    # 3. Gemini — final fallback
    gemini = registry.get(ModelProvider.GEMINI)
    if gemini:
        scorers.append(("Gemini", gemini))

    return scorers


_BENCH_SCORE_SYSTEM = """\
You are a CIE A-Level chief examiner marking a student answer against the official mark scheme.

Mark scheme reference:
{ms_excerpt}

Read the mark scheme carefully, then score the student answer.
Respond with valid JSON only (no markdown fences):
{{
  "marks_achieved": <integer>,
  "total_marks": <integer>,
  "grade_estimate": "A*" or "A" or "B" or "C" or "D" or "E" or "U",
  "mark_points_hit": ["brief description of each mark point awarded"],
  "mark_points_missed": ["brief description of each mark point not awarded"]
}}

Rules:
- Be strict and accurate — award marks only for points genuinely addressed
- grade_estimate: A*(≥90%), A(≥80%), B(≥70%), C(≥60%), D(≥50%), E(≥40%), U(<40%)
- mark_points_hit and mark_points_missed must reference specific MS criteria
- IMPORTANT: Keep every entry in mark_points_hit and mark_points_missed under 12 words
- IMPORTANT: Output raw JSON only — do NOT wrap in markdown code fences
"""


def _strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences if present.

    Handles all common variants:
      - ```json\\n{...}\\n```
      - ```\\n{...}\\n```
      - leading/trailing whitespace around the fences
    Uses regex so it degrades gracefully when fences are absent or malformed.
    """
    import re
    text = text.strip()
    if text.startswith("```"):
        # Strip opening fence (```json or just ```)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # Strip closing fence (``` possibly followed by whitespace/newlines)
        text = re.sub(r'\n?\s*```\s*$', '', text)
        text = text.strip()
    return text


def _parse_score_response(
    resp_content: str, marks: int
) -> tuple[int, int, str, list[str], list[str]] | None:
    """Parse a scorer JSON response. Returns None if the response is unparseable."""
    raw = _strip_json_fences(resp_content)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    fragment = raw[start:end]
    # ── Attempt to repair truncated JSON (unclosed arrays / object) ───────────
    # If `end` landed on an inner `}` (truncation), `json.loads` will fail.
    # We try up to 3 repair attempts by appending missing closing brackets.
    parse_attempts = [fragment]
    # Try closing an open array then the object, or just the object
    if not fragment.rstrip().endswith("}"):
        parse_attempts += [fragment + ']}', fragment + ']}}\n', fragment + '}']
    for attempt_text in parse_attempts:
        try:
            data     = json.loads(attempt_text)
            achieved = int(data.get("marks_achieved", 0))
            total    = int(data.get("total_marks", marks))
            grade    = str(data.get("grade_estimate", "?"))
            hits     = list(data.get("mark_points_hit", []))
            misses   = list(data.get("mark_points_missed", []))
            return achieved, total, grade, hits, misses
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    return None


async def _bench_score(
    registry,
    cost_tracker,
    question_text: str,
    marks: int,
    answer_text: str,
    ms_text: str,
    subject: str,
) -> tuple[int, int, str, list[str], list[str]]:
    """Score an answer against the MS using a fallback chain of scorers.

    Tries: Claude Haiku → GPT-4o → Gemini (first success wins).
    Each scorer gets 2 attempts before falling through to the next.
    Returns (marks_achieved, total_marks, grade_estimate, hits, misses).
    Returns grade="SCORE_FAILED" only when ALL scorers are unavailable.
    """
    from shark_answer.providers.registry import JUDGE_TIMEOUT

    scorer_chain = _build_scorer_chain(registry)
    if not scorer_chain:
        return 0, marks, "SCORE_FAILED", [], ["No scorers available (Claude/GPT-4o/Gemini)"]

    # Truncate MS excerpt to keep prompt under ~16k chars
    ms_excerpt = ms_text[:12_000]
    if len(ms_text) > 12_000:
        ms_excerpt += "\n[...mark scheme truncated for token budget]"

    system = _BENCH_SCORE_SYSTEM.format(ms_excerpt=ms_excerpt)
    prompt = (
        f"Question [{marks} marks]:\n{question_text}\n\n"
        f"Student answer:\n{answer_text}"
    )

    for scorer_name, scorer_inst in scorer_chain:
        logger.info("Scoring with %s...", scorer_name)
        last_error = f"{scorer_name} failed"

        for attempt in range(2):   # 2 attempts per scorer before falling through
            if attempt > 0:
                logger.info("  [%s] retry attempt 2...", scorer_name)

            try:
                resp = await asyncio.wait_for(
                    scorer_inst.generate(
                        prompt=prompt,
                        system=system,
                        temperature=0.0,   # deterministic: same answer = same score
                        max_tokens=4096,   # 4096 prevents truncation of mark-point lists
                    ),
                    timeout=JUDGE_TIMEOUT,
                )
                cost_tracker.record(resp, subject, "benchmark_scoring")
            except asyncio.TimeoutError:
                last_error = f"{scorer_name} timeout"
                logger.warning("  [%s] timeout on attempt %d", scorer_name, attempt + 1)
                break   # don't retry timeout — move to next scorer immediately
            except Exception as exc:
                last_error = f"{scorer_name} API error: {exc}"
                logger.warning("  [%s] API error attempt %d: %s", scorer_name, attempt + 1, exc)
                continue

            if not resp.success:
                last_error = f"{scorer_name} model error: {resp.error}"
                logger.warning("  [%s] model error attempt %d: %s",
                               scorer_name, attempt + 1, resp.error)
                # 529 overloaded → move to next scorer immediately (no point retrying)
                if "529" in str(resp.error) or "overload" in str(resp.error).lower():
                    logger.info("  [%s] overloaded, skipping to next scorer", scorer_name)
                    break
                continue

            parsed = _parse_score_response(resp.content, marks)
            if parsed is not None:
                logger.info("  [%s] scored successfully", scorer_name)
                return parsed

            last_error = f"{scorer_name} parse error"
            logger.warning("  [%s] parse error attempt %d: %.100s",
                           scorer_name, attempt + 1, resp.content)
            # On parse failure: make second attempt with stricter JSON-only prompt
            if attempt == 0:
                prompt = (
                    f"Question [{marks} marks]:\n{question_text}\n\n"
                    f"Student answer:\n{answer_text}\n\n"
                    "IMPORTANT: Respond with ONLY a raw JSON object. "
                    "No markdown, no explanation, no code fences. "
                    "Start your response with { and end with }."
                )

        logger.warning("Scorer %s exhausted (%s), trying next in chain", scorer_name, last_error)

    logger.error("All scorers failed. Last error: %s", last_error)
    return 0, marks, "SCORE_FAILED", [], [f"SCORE_FAILED — all scorers unavailable: {last_error}"]


# ── File helpers ────────────────────────────────────────────────────────────────

def _resolve_paper_files(paper_def: dict, manifest: dict, kb_dir: Path) -> dict:
    """Resolve qp_file / ms_file from manifest if not hard-coded."""
    mkey = paper_def["manifest_key"]
    if mkey not in manifest:
        return {**paper_def, "error": f"Subject '{mkey}' not in manifest"}

    subj_data = manifest[mkey]

    qp_file  = paper_def["qp_file"]
    ms_file  = paper_def["ms_file"]
    pnum     = paper_def["paper_number"]

    if qp_file is None:
        # Pick the first available question paper
        qps = sorted(subj_data.get("question_papers", []),
                     key=lambda e: e.get("year", 0), reverse=True)
        if not qps:
            return {**paper_def, "error": f"No question papers in manifest for {mkey}"}
        entry  = qps[0]
        qp_file = entry.get("renamed") or entry.get("original", "")
        pnum    = entry.get("paper")

    if ms_file is None:
        # Match MS by paper number
        mss = sorted(subj_data.get("mark_schemes", []),
                     key=lambda e: e.get("year", 0), reverse=True)
        matched = [e for e in mss if e.get("paper") == pnum]
        if not matched:
            matched = mss  # fallback: latest MS regardless of paper
        if not matched:
            return {**paper_def, "error": f"No mark schemes in manifest for {mkey}"}
        ms_file = matched[0].get("renamed") or matched[0].get("original", "")

    return {
        **paper_def,
        "qp_file": qp_file,
        "ms_file": ms_file,
        "paper_number": pnum,
        "error": None,
    }


def _load_file_bytes(kb_dir: Path, manifest_key: str, category: str, filename: str) -> bytes | None:
    """Load a PDF from the KB directory."""
    path = kb_dir / manifest_key / category / filename
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    return path.read_bytes()


def _load_txt(kb_dir: Path, manifest_key: str, category: str, filename: str) -> str:
    """Load the extracted .txt companion of a PDF. Empty string if missing."""
    txt_path = kb_dir / manifest_key / category / Path(filename).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="replace")
    # Fallback: try reading the PDF directly (won't work well without text extraction)
    logger.warning("No .txt for %s — MS scoring will be limited", txt_path.name)
    return ""


# ── "Hence" question context injection ─────────────────────────────────────────

import re as _re

_HENCE_RE = _re.compile(
    r'\b(hence|using (your|the) (answer|result)|using (part|your answer from)|'
    r'using (the result from)|from (part|your answer)|'
    r'using (the value|the expression|this result|your result))\b',
    _re.IGNORECASE,
)


def _is_hence_question(text: str) -> bool:
    """Return True if the question text implies it needs a prior sub-part answer."""
    return bool(_HENCE_RE.search(text))


def _get_major_question(number: str) -> str:
    """Extract the top-level question number, e.g. '3' from '3(b)(ii)'."""
    m = _re.match(r'^(\d+)', number.strip())
    return m.group(1) if m else number.strip()


def _build_hence_context(
    current_number: str,
    current_text: str,
    answered_parts: dict[str, str],
) -> str:
    """Build a prior-answer context block for a 'Hence' question.

    Returns an empty string if:
    - The question is not a 'Hence' question, OR
    - No prior answers from the same major question group are available.

    Otherwise returns a labelled block ready to prepend to full_paper_text.
    """
    if not _is_hence_question(current_text):
        return ""

    major = _get_major_question(current_number)

    # Collect all answered parts from the same major question (e.g. same Q3)
    prior: list[tuple[str, str]] = [
        (num, ans)
        for num, ans in answered_parts.items()
        if _get_major_question(num) == major and num != current_number
    ]

    if not prior:
        return ""

    # Sort numerically/lexicographically so parts appear in order
    prior.sort(key=lambda x: x[0])

    lines = [
        f"=== PRIOR ANSWERS FOR QUESTION {major} ===",
        "The current question says 'Hence...' — use these earlier answers as input:\n",
    ]
    for num, ans in prior:
        # Truncate very long answers so we don't balloon the prompt
        display_ans = ans[:2000] + "\n[...truncated]" if len(ans) > 2000 else ans
        lines.append(f"--- Part {num} answer ---\n{display_ans}")

    return "\n\n".join(lines)


# ── Paper benchmark ─────────────────────────────────────────────────────────────

async def run_paper_benchmark(
    paper_def: dict,
    kb_dir: Path,
    registry,
    config,
    cost_tracker,
    max_versions: int = 1,
    max_questions: int = 5,
    fair_mode: bool = True,
) -> PaperResult:
    """Run the full benchmark for a single paper."""
    from shark_answer.config import Subject, Language, SUBJECT_PIPELINE_MAP, VISION_ONLY_SUBJECTS
    from shark_answer.pipelines.pipeline_a_science import run_pipeline_a
    from shark_answer.pipelines.pipeline_b_essay import run_pipeline_b
    from shark_answer.pipelines.pipeline_c_cs import run_pipeline_c
    from shark_answer.pipelines.router import route_question
    from shark_answer.knowledge_base.retriever import build_prompt_context
    from shark_answer.utils.file_converter import convert_file_to_images
    from shark_answer.config import Pipeline

    mkey    = paper_def["manifest_key"]
    qp_file = paper_def["qp_file"]
    ms_file = paper_def["ms_file"]
    pnum    = paper_def["paper_number"]
    skey    = paper_def["subject_key"]

    result = PaperResult(
        subject_key=skey,
        manifest_key=mkey,
        qp_file=qp_file,
        ms_file=ms_file,
        paper_number=pnum,
    )

    # ── Map subject_key → Subject enum ────────────────────────────────────────
    try:
        subject_enum = Subject(skey)
    except ValueError:
        result.error = f"Unknown subject enum value: {skey}"
        return result

    # ── 1. Load QP text (.txt preferred; PDF as fallback for images) ─────────
    t0 = time.perf_counter()
    from shark_answer.utils.image_extractor import (
        extract_questions_from_text,
        extract_questions_from_images,
        extract_questions_whole_paper,
    )

    qp_text = _load_txt(kb_dir, mkey, "question_papers", qp_file)  # pre-extracted .txt
    qp_bytes = _load_file_bytes(kb_dir, mkey, "question_papers", qp_file)  # PDF bytes

    if qp_bytes is None and not qp_text:
        result.error = f"QP not found: {kb_dir / mkey / 'question_papers' / qp_file}"
        return result

    # For vision-only subjects (Math, Further Math) PyMuPDF cannot extract
    # equations — the output is garbled characters.  Skip txt-first entirely and
    # go straight to vision.  Also clear qp_text so we don't pass garbled text
    # as pipeline context.
    force_vision = subject_enum in VISION_ONLY_SUBJECTS
    if force_vision:
        qp_text = ""  # don't pass garbled txt as full_paper_text to pipelines
        print(f"  [{skey}] Vision-only subject — skipping .txt, using image extraction…",
              flush=True)

    # ── 2. Extract questions: .txt-first, vision fallback ────────────────────
    questions: list = []
    extraction_method = "none"

    if qp_text and not force_vision:
        # PRIMARY: parse questions directly from pre-extracted .txt
        # Reliable, fast, includes ALL table/figure data — no API calls needed
        print(f"  [{skey}] Extracting questions from .txt (text-first mode)…", flush=True)
        try:
            questions = extract_questions_from_text(qp_text)
            if questions:
                extraction_method = "txt"
                print(f"  [{skey}] Text parser found {len(questions)} question(s).",
                      flush=True)
        except Exception as exc:
            logger.warning("Text extraction failed for %s: %s", qp_file, exc)

    if not questions and qp_bytes:
        # FALLBACK: vision extraction from PDF images
        print(f"  [{skey}] Text extraction found 0 questions — "
              "falling back to vision extraction…", flush=True)
        try:
            images = convert_file_to_images(qp_file, qp_bytes)
        except Exception as exc:
            result.error = f"PDF→image conversion failed: {exc}"
            return result

        if not images:
            result.error = "PDF produced no images"
            return result

        print(f"  [{skey}] Extracting questions from {len(images)} page(s) via whole-paper vision…",
              flush=True)
        try:
            # Whole-paper extraction: single call with all pages for full context
            questions, extract_resp = await extract_questions_whole_paper(registry, images, skey)
            if questions:
                extraction_method = "whole-paper-vision"
            elif not questions:
                # Fallback to page-by-page if whole-paper returned nothing
                logger.warning("[%s] Whole-paper extraction returned 0 questions — trying page-by-page", skey)
                questions, _ = await extract_questions_from_images(registry, images)
                if questions:
                    extraction_method = "vision"
        except Exception as exc:
            result.error = f"Question extraction failed: {exc}"
            return result

    result.timing.extraction_s = time.perf_counter() - t0

    if not questions:
        result.error = "No questions extracted from paper (both txt and vision failed)"
        return result

    # Drop questions with 0 marks — cover pages / instructions
    real_questions = [q for q in questions if q.marks > 0]
    dropped = len(questions) - len(real_questions)
    if dropped:
        print(f"  [{skey}] Dropped {dropped} zero-mark item(s) (cover/instruction pages).",
              flush=True)
    questions = real_questions

    if not questions:
        result.error = "No scoreable questions extracted (all had 0 marks)"
        return result

    # ── 2b. Filter phantom question IDs ──────────────────────────────────────
    # The txt parser can misread numbers from equation labels, data values, or
    # table entries in the paper body as question numbers, creating phantom IDs
    # such as "13", "22", "17" that inflate the denominator with questions the
    # pipeline cannot meaningfully answer.
    #
    # Safety rule (txt extraction only — vision extraction is already clean):
    #   A bare integer > 10 with NO sub-part suffix is almost certainly a phantom.
    #   Valid CIE top-level question numbers are 1–10; anything higher that has no
    #   letter/paren suffix (e.g. "13", not "13(a)") is flagged and removed.
    _BARE_INT_RE = re.compile(r"^\d+$")

    def _is_phantom_qnum(num: str) -> bool:
        num = num.strip()
        if _BARE_INT_RE.fullmatch(num):
            try:
                return int(num) > 10
            except ValueError:
                return False
        return False

    # Computer Science papers (cs_9618) genuinely have question numbers > 10
    # (e.g. Q21 is a real 7-mark question).  Only apply the phantom filter to
    # subjects whose txt parser is known to hallucinate high bare integers.
    # Physics and chemistry are already vision-only so they never reach here.
    _PHANTOM_FILTER_SUBJECTS = frozenset({"economics_9708", "biology_9700"})
    if extraction_method == "txt" and skey in _PHANTOM_FILTER_SUBJECTS:
        phantom_ids = [q.number for q in questions if _is_phantom_qnum(q.number)]
        if phantom_ids:
            questions = [q for q in questions if not _is_phantom_qnum(q.number)]
            print(
                f"  [{skey}] Filtered {len(phantom_ids)} phantom question ID(s) "
                f"(bare integers > 10 from txt parser): {phantom_ids}",
                flush=True,
            )

    print(f"  [{skey}] {len(questions)} real question(s) found "
          f"[via {extraction_method}]. Processing up to {max_questions}.", flush=True)
    questions = questions[:max_questions]

    # ── 4. Load MS text ───────────────────────────────────────────────────────
    ms_text = _load_txt(kb_dir, mkey, "mark_schemes", ms_file)

    # ── 5. Get KB context (once per paper) ───────────────────────────────────
    # Fair mode: exclude the paper year so the model can't see the mark scheme
    # it's being tested against. Cheat mode: include all years (leaky, inflates scores).
    paper_year: Optional[int] = None
    if qp_file:
        # Extract year from filename like "2024_june_qp_21.pdf" → 2024
        import re as _re
        m = _re.match(r"(\d{4})_", qp_file)
        if m:
            paper_year = int(m.group(1))
    exclude_yr = paper_year if fair_mode else None
    mode_label = "FAIR" if fair_mode else "⚠ CHEAT (data leakage — year included)"
    print(f"  KB mode: {mode_label}"
          + (f" (excluding {paper_year})" if exclude_yr else ""), flush=True)
    kb_context = build_prompt_context(
        subject=skey, paper_number=pnum, exclude_year=exclude_yr
    )

    # ── 6. Solve each question ────────────────────────────────────────────────
    cost_before_solving = cost_tracker.total_cost
    t_solve = time.perf_counter()

    # Tracks best answers already generated this paper, keyed by question number.
    # Used to inject prior sub-part answers into "Hence..." questions.
    answered_parts: dict[str, str] = {}

    for qi, q in enumerate(questions):
        # 10-second pause between questions to avoid API rate limiting
        if qi > 0:
            await asyncio.sleep(10)
        q_t0 = time.perf_counter()
        print(f"  [{skey}] Q{q.number} ({q.marks}m) → pipeline…", flush=True)

        # Determine pipeline
        try:
            pipeline = route_question(q, subject_enum)
        except Exception:
            pipeline = SUBJECT_PIPELINE_MAP.get(subject_enum)

        if pipeline is None:
            qr = QuestionResult(
                number=q.number, text=q.text, marks_total=q.marks,
                marks_achieved=0, score_str=f"0/{q.marks}",
                grade_estimate="U", best_answer="", pipeline="?",
                errors=["No pipeline for subject"],
            )
            result.questions.append(qr)
            continue

        # Build "Hence" context — prepended to full_paper_text when applicable
        hence_ctx = _build_hence_context(q.number, q.text, answered_parts)
        if hence_ctx:
            print(f"  [{skey}] Q{q.number} — 'Hence' detected, injecting prior answers",
                  flush=True)
            # Prepend prior answers, then append the QP text (tables / figures)
            effective_paper_text = hence_ctx + ("\n\n" + qp_text if qp_text else "")
        else:
            effective_paper_text = qp_text

        # Run pipeline
        pipe_errors: list[str] = []
        providers_ok:   list[str] = []
        providers_fail: list[str] = []
        pipeline_result = None

        try:
            if pipeline == Pipeline.SCIENCE_MATH:
                pipeline_result = await run_pipeline_a(
                    question=q, subject=skey, registry=registry,
                    config=config, cost_tracker=cost_tracker,
                    kb_context=kb_context, language="en",
                    max_versions=max_versions,
                    paper=pnum,
                    full_paper_text=effective_paper_text,
                )
            elif pipeline == Pipeline.ESSAY:
                pipeline_result = await run_pipeline_b(
                    question=q, subject=skey, registry=registry,
                    config=config, cost_tracker=cost_tracker,
                    kb_context=kb_context, language="en",
                    max_versions=max_versions,
                    paper=pnum,
                    full_paper_text=effective_paper_text,
                )
            elif pipeline == Pipeline.CS:
                pipeline_result = await run_pipeline_c(
                    question=q, subject=skey, registry=registry,
                    config=config, cost_tracker=cost_tracker,
                    kb_context=kb_context, language="en",
                    max_versions=max_versions,
                    paper=pnum,
                    full_paper_text=effective_paper_text,
                )
            else:
                pipe_errors.append(f"Pipeline {pipeline} not implemented")
        except Exception as exc:
            pipe_errors.append(f"Pipeline exception: {exc}")
            logger.exception("Pipeline error for Q%s", q.number)

        # Extract best answer from pipeline result
        best_answer = ""
        if pipeline_result and pipeline_result.versions:
            best_answer = pipeline_result.versions[0].answer_text
            pipe_errors.extend(pipeline_result.errors)
            # Collect provider health from version labels
            for v in pipeline_result.versions:
                if v.provider:
                    providers_ok.append(v.provider)
        elif pipeline_result:
            pipe_errors.extend(pipeline_result.errors)

        # Store answer for future "Hence" questions in the same paper
        if best_answer:
            answered_parts[q.number] = best_answer

        result.timing.solving_s += time.perf_counter() - q_t0

        # ── 7. Score answer against MS ────────────────────────────────────────
        achieved = 0
        total    = q.marks
        grade    = "?"
        score_str = f"?/{q.marks}"

        if best_answer and ms_text:
            t_score = time.perf_counter()
            print(f"  [{skey}] Q{q.number} scoring…", flush=True)
            # 5-second pause before each scoring call to avoid rate-limiting Claude
            await asyncio.sleep(5)
            try:
                achieved, total, grade, hits, misses = await _bench_score(
                    registry=registry,
                    cost_tracker=cost_tracker,
                    question_text=q.text,
                    marks=q.marks,
                    answer_text=best_answer,
                    ms_text=ms_text,
                    subject=skey,
                )
                if grade == "SCORE_FAILED":
                    score_str = f"FAILED/{total}"
                    pipe_errors.extend(misses)   # misses holds the failure reason
                else:
                    score_str = f"{achieved}/{total}"
            except Exception as exc:
                pipe_errors.append(f"Scoring error: {exc}")
                grade = "SCORE_FAILED"
                score_str = f"FAILED/{q.marks}"
            result.timing.scoring_s += time.perf_counter() - t_score
        elif not ms_text:
            pipe_errors.append("MS text unavailable — scoring skipped")

        qr = QuestionResult(
            number=q.number,
            text=q.text[:200],       # truncate for report
            marks_total=total,
            marks_achieved=achieved,
            score_str=score_str,
            grade_estimate=grade,
            best_answer=best_answer[:500],
            pipeline=pipeline.value if pipeline else "?",
            providers_ok=providers_ok,
            providers_fail=providers_fail,
            timing_s=time.perf_counter() - q_t0,
            errors=pipe_errors,
        )
        result.questions.append(qr)

        print(
            f"  [{skey}] Q{q.number}: {score_str} ({grade})  "
            f"[{qr.timing_s:.1f}s]",
            flush=True,
        )

    result.timing.solving_s = time.perf_counter() - t_solve - result.timing.scoring_s
    result.cost_usd = cost_tracker.total_cost - cost_before_solving

    return result


# ── Report generation ───────────────────────────────────────────────────────────

def _cie_grade_from_pct(pct: float) -> str:
    if pct >= 90: return "A*"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 60: return "C"
    if pct >= 50: return "D"
    if pct >= 40: return "E"
    return "U"


def generate_report(results: list[PaperResult], versions: int, max_q: int) -> str:
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 72)
    lines.append("  SHARK ANSWER  —  BENCHMARK REPORT")
    lines.append(f"  Generated: {ts}   |  versions={versions}  max_q={max_q}")
    lines.append("=" * 72)
    lines.append("")

    # ── Paper summary table ────────────────────────────────────────────────────
    lines.append("┌─────────────────────┬──────────┬───────┬───────┬──────┬─────────┐")
    lines.append("│ Subject             │ Paper    │ Score │ Grade │ Time │    Cost │")
    lines.append("├─────────────────────┼──────────┼───────┼───────┼──────┼─────────┤")

    total_achieved = 0
    total_marks    = 0
    total_cost     = 0.0
    total_time     = 0.0

    for r in results:
        subj  = r.subject_key[:21]
        paper = (r.qp_file or "?")[:8]
        score = f"{r.total_achieved}/{r.total_marks}" if r.total_marks else "—"
        grade = r.grade if r.total_marks else "—"
        t     = r.timing.total_s
        cost  = r.cost_usd

        if r.error:
            score = "ERROR"
            grade = "—"

        lines.append(
            f"│ {subj:<21} │ {paper:<8} │ {score:>5} │ {grade:>5} │"
            f" {t:>4.0f}s │ ${cost:>6.3f} │"
        )
        total_achieved += r.total_achieved
        total_marks    += r.total_marks
        total_cost     += cost
        total_time     += t

    lines.append("├─────────────────────┼──────────┼───────┼───────┼──────┼─────────┤")
    overall_pct   = (total_achieved / total_marks * 100) if total_marks else 0.0
    overall_grade = _cie_grade_from_pct(overall_pct)
    lines.append(
        f"│ {'TOTAL / OVERALL':<21} │ {'':8} │ "
        f"{total_achieved}/{total_marks} │ {overall_grade:>5} │"
        f" {total_time:>4.0f}s │ ${total_cost:>6.3f} │"
    )
    lines.append("└─────────────────────┴──────────┴───────┴───────┴──────┴─────────┘")
    lines.append("")

    # ── Per-paper detail ───────────────────────────────────────────────────────
    for r in results:
        lines.append(f"── {r.subject_key.upper()}  ({r.manifest_key})")
        if r.error:
            lines.append(f"   ERROR: {r.error}")
            lines.append("")
            continue

        lines.append(
            f"   Paper: {r.qp_file}   MS: {r.ms_file}   "
            f"Paper#: {r.paper_number}"
        )
        lines.append(
            f"   Timing: extract={r.timing.extraction_s:.1f}s  "
            f"solve={r.timing.solving_s:.1f}s  "
            f"score={r.timing.scoring_s:.1f}s  "
            f"total={r.timing.total_s:.1f}s"
        )
        lines.append(f"   Cost: ${r.cost_usd:.4f}")

        if r.all_providers_ok:
            lines.append(f"   Models OK:   {', '.join(r.all_providers_ok)}")
        if r.all_providers_fail:
            lines.append(f"   Models FAIL: {', '.join(r.all_providers_fail)}")

        if r.failed_scoring_count:
            lines.append(f"   ⚠ {r.failed_scoring_count} question(s) had SCORE_FAILED "
                         f"(excluded from score totals)")

        lines.append("")
        lines.append(f"   {'Q#':<8} {'Marks':>6} {'Score':>10} {'Grade':>12} {'Time':>6}")
        lines.append(f"   {'─'*8} {'─'*6} {'─'*10} {'─'*12} {'─'*6}")
        for q in r.questions:
            grade_col = q.grade_estimate if q.grade_estimate != "SCORE_FAILED" else "SCORE_FAILED"
            lines.append(
                f"   {q.number:<8} {q.marks_total:>6} {q.score_str:>10} "
                f"{grade_col:>12} {q.timing_s:>5.1f}s"
            )
            if q.errors:
                for e in q.errors[:2]:
                    lines.append(f"             ⚠ {e[:70]}")
        lines.append("")

    # ── Model health table ─────────────────────────────────────────────────────
    all_ok:   dict[str, int] = {}
    all_fail: dict[str, int] = {}
    for r in results:
        for q in r.questions:
            for p in q.providers_ok:
                all_ok[p] = all_ok.get(p, 0) + 1
            for p in q.providers_fail:
                all_fail[p] = all_fail.get(p, 0) + 1

    if all_ok or all_fail:
        lines.append("── MODEL HEALTH")
        all_providers = sorted(set(list(all_ok.keys()) + list(all_fail.keys())))
        lines.append(f"   {'Provider':<16} {'OK':>4}  {'FAIL':>4}")
        lines.append(f"   {'─'*16} {'─'*4}  {'─'*4}")
        for p in all_providers:
            ok   = all_ok.get(p, 0)
            fail = all_fail.get(p, 0)
            lines.append(f"   {p:<16} {ok:>4}  {fail:>4}")
        lines.append("")

    # ── Areas for improvement ──────────────────────────────────────────────────
    lines.append("── AREAS FOR IMPROVEMENT")
    worst_subjects = sorted(
        [r for r in results if not r.error and r.total_marks > 0],
        key=lambda r: r.score_pct,
    )[:3]
    if worst_subjects:
        for r in worst_subjects:
            lines.append(
                f"   {r.subject_key:<20}  {r.score_pct:>5.1f}%  "
                f"(grade {r.grade})"
            )
        lines.append("")

    # Failed models
    persistent_failures = {p for p, cnt in all_fail.items() if cnt >= 2}
    if persistent_failures:
        lines.append(
            "   Models with repeated failures: "
            + ", ".join(sorted(persistent_failures))
        )
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shark Answer automated benchmark against CIE past papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Run Economics + Physics only (1 version each)",
    )
    p.add_argument(
        "--subject",
        help="Run a single subject (e.g. economics, physics, math)",
    )
    p.add_argument(
        "--versions", type=int, default=1,
        help="Max answer versions per question (default: 1)",
    )
    p.add_argument(
        "--max-questions", type=int, default=5,
        metavar="N",
        help="Max questions to process per paper (default: 5)",
    )
    p.add_argument(
        "--output",
        help="Save report to this file (default: auto-named in benchmark_results/)",
    )
    p.add_argument(
        "--kb-dir", default=str(_DEFAULT_KB_DIR),
        help=f"KB root directory (default: {_DEFAULT_KB_DIR})",
    )
    p.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help=(
            "Number of times to run each paper (default: 1, max: 3). "
            "When N>1, reports best/worst/avg/stddev and grades on average score."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # ── Fair / cheat mode ──────────────────────────────────────────────────
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fair",
        dest="fair_mode", action="store_true", default=True,
        help=(
            "Fair evaluation: exclude the paper year's mark schemes from KB context "
            "so the model cannot directly see the answer it is scored against. "
            "(DEFAULT — use this for honest performance measurement)"
        ),
    )
    mode_group.add_argument(
        "--cheat",
        dest="fair_mode", action="store_false",
        help=(
            "Cheat mode: include ALL years in KB context (data leakage). "
            "The model can see the exact mark scheme it is being scored against. "
            "Scores will be inflated. Labelled ⚠ CHEAT in the report."
        ),
    )

    return p.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    from shark_answer.config import AppConfig
    from shark_answer.providers.registry import ProviderRegistry
    from shark_answer.utils.cost_tracker import CostTracker

    kb_dir = Path(args.kb_dir).expanduser()
    if not kb_dir.exists():
        print(f"ERROR: KB directory not found: {kb_dir}", file=sys.stderr)
        return 1

    manifest_path = kb_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # ── Init app config + providers ────────────────────────────────────────────
    print("Initialising Shark Answer config…", flush=True)
    config = AppConfig.from_env()
    registry = ProviderRegistry(config)
    cost_tracker = CostTracker(budget_warning_usd=50.0)

    available = [m.value for m in config.models]
    print(f"Available models: {available}\n", flush=True)

    if not available:
        print("ERROR: No API keys configured. Check your .env file.", file=sys.stderr)
        return 1

    # ── Select papers ──────────────────────────────────────────────────────────
    papers = list(_ALL_PAPERS)

    if args.quick:
        papers = [p for p in papers if p["subject_key"] in _QUICK_SUBJECTS]
    elif args.subject:
        papers = [p for p in papers if p["subject_key"] == args.subject]
        if not papers:
            valid = [p["subject_key"] for p in _ALL_PAPERS]
            print(
                f"ERROR: Unknown subject '{args.subject}'. Valid: {valid}",
                file=sys.stderr,
            )
            return 1

    max_versions = max(1, min(5, args.versions))
    max_q        = max(1, args.max_questions)
    fair_mode    = getattr(args, "fair_mode", True)
    num_runs     = max(1, min(3, getattr(args, "runs", 1)))

    mode_str = "FAIR MODE (no data leakage)" if fair_mode else "⚠  CHEAT MODE (data leakage — scores inflated)"
    runs_str = f", runs={num_runs}" if num_runs > 1 else ""
    print(
        f"Benchmark: {len(papers)} paper(s), "
        f"versions={max_versions}, max_q={max_q}, mode={mode_str}{runs_str}\n"
        + ("  [QUICK MODE: Economics + Physics only]\n" if args.quick else ""),
        flush=True,
    )

    # ── Run benchmarks ─────────────────────────────────────────────────────────
    results: list[PaperResult] = []
    t_global = time.perf_counter()

    for paper_def in papers:
        resolved = _resolve_paper_files(paper_def, manifest, kb_dir)
        err = resolved.get("error")
        if err:
            print(f"\n[{resolved['subject_key']}] SKIP: {err}", flush=True)
            results.append(PaperResult(
                subject_key=resolved["subject_key"],
                manifest_key=resolved["manifest_key"],
                qp_file=str(resolved.get("qp_file") or "?"),
                ms_file=str(resolved.get("ms_file") or "?"),
                paper_number=resolved.get("paper_number"),
                error=err,
            ))
            continue

        skey = resolved["subject_key"]
        print(
            f"\n{'='*60}\n"
            f"  Benchmarking: {skey.upper()} "
            f"({resolved['manifest_key']})\n"
            f"  QP: {resolved['qp_file']}   MS: {resolved['ms_file']}\n"
            f"{'='*60}",
            flush=True,
        )

        # ── Multi-run: run each paper num_runs times ───────────────────────────
        run_results: list[PaperResult] = []
        for run_i in range(num_runs):
            if num_runs > 1:
                print(f"  [{skey}] Run {run_i + 1}/{num_runs}…", flush=True)
            result = await run_paper_benchmark(
                paper_def=resolved,
                kb_dir=kb_dir,
                registry=registry,
                config=config,
                cost_tracker=cost_tracker,
                max_versions=max_versions,
                max_questions=max_q,
                fair_mode=fair_mode,
            )
            run_results.append(result)

            summary = (
                f"  → {skey} run {run_i+1}: {result.total_achieved}/{result.total_marks} "
                f"({result.grade})  ${result.cost_usd:.4f}  "
                f"{result.timing.total_s:.0f}s"
            )
            if result.error:
                summary = f"  → {skey} run {run_i+1}: ERROR — {result.error}"
            print(summary, flush=True)

        # For single runs: use the result directly
        # For multi-run: compute stats, use the run whose pct is closest to average
        if num_runs == 1 or not run_results:
            final_result = run_results[0] if run_results else PaperResult(
                subject_key=skey,
                manifest_key=resolved["manifest_key"],
                qp_file=str(resolved.get("qp_file") or "?"),
                ms_file=str(resolved.get("ms_file") or "?"),
                paper_number=resolved.get("paper_number"),
                error="No runs completed",
            )
        else:
            valid_runs = [r for r in run_results if not r.error and r.total_marks > 0]
            if valid_runs:
                import statistics as _stats
                pcts = [r.score_pct for r in valid_runs]
                avg_pct = _stats.mean(pcts)
                best_pct = max(pcts)
                worst_pct = min(pcts)
                stddev = _stats.stdev(pcts) if len(pcts) > 1 else 0.0
                # Pick the run closest to average
                final_result = min(valid_runs, key=lambda r: abs(r.score_pct - avg_pct))
                print(
                    f"  [{skey}] Multi-run stats: avg={avg_pct:.1f}% "
                    f"best={best_pct:.1f}% worst={worst_pct:.1f}% "
                    f"σ={stddev:.1f}%  grade-on-avg={_cie_grade_from_pct(avg_pct)}",
                    flush=True,
                )
            else:
                final_result = run_results[-1]

        results.append(final_result)
        summary = (
            f"  → {skey}: {final_result.total_achieved}/{final_result.total_marks} "
            f"({final_result.grade})  ${final_result.cost_usd:.4f}  "
            f"{final_result.timing.total_s:.0f}s"
        )
        if final_result.error:
            summary = f"  → {skey}: ERROR — {final_result.error}"
        print(summary, flush=True)

    elapsed = time.perf_counter() - t_global
    print(
        f"\nAll papers done in {elapsed:.1f}s. "
        f"Total cost: ${cost_tracker.total_cost:.4f}\n",
        flush=True,
    )

    # ── Generate report ────────────────────────────────────────────────────────
    report = generate_report(results, max_versions, max_q)
    # Prepend mode header
    mode_header = (
        "=" * 72 + "\n"
        f"  BENCHMARK MODE: {'FAIR (no data leakage)' if fair_mode else '⚠  CHEAT (data leakage — scores INFLATED)'}\n"
        + ("  Fair mode excludes the paper year from KB context.\n" if fair_mode else
           "  Cheat mode includes the mark scheme year — NOT a valid performance measure!\n")
        + "=" * 72 + "\n\n"
    )
    report = mode_header + report
    print(report)

    # ── Save report ────────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject_part = "quick" if args.quick else (args.subject or "all")
        mode_part = "fair" if fair_mode else "cheat"
        out_path = _RESULTS_DIR / f"benchmark_{subject_part}_{mode_part}_{ts}.txt"

    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}", flush=True)

    # Also save raw JSON for programmatic use
    json_path = out_path.with_suffix(".json")
    raw = []
    for r in results:
        raw.append({
            "subject_key":  r.subject_key,
            "manifest_key": r.manifest_key,
            "qp_file":      r.qp_file,
            "ms_file":      r.ms_file,
            "paper_number": r.paper_number,
            "total_marks":  r.total_marks,
            "total_achieved": r.total_achieved,
            "score_pct":    round(r.score_pct, 1),
            "grade":        r.grade,
            "cost_usd":     round(r.cost_usd, 6),
            "timing_s":     {
                "extraction": round(r.timing.extraction_s, 2),
                "solving":    round(r.timing.solving_s, 2),
                "scoring":    round(r.timing.scoring_s, 2),
            },
            "error": r.error,
            "questions": [
                {
                    "number":         q.number,
                    "marks_total":    q.marks_total,
                    "marks_achieved": q.marks_achieved,
                    "score_str":      q.score_str,
                    "grade_estimate": q.grade_estimate,
                    "pipeline":       q.pipeline,
                    "timing_s":       round(q.timing_s, 2),
                    "providers_ok":   q.providers_ok,
                    "providers_fail": q.providers_fail,
                    "errors":         q.errors,
                    "text":           q.text,
                    "best_answer":    q.best_answer,
                }
                for q in r.questions
            ],
        })
    json_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON data saved to:  {json_path}", flush=True)

    return 0


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
