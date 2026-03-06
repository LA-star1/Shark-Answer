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
    {
        "subject_key":   "chinese",
        "manifest_key":  "chinese_8238",
        "qp_file":       None,   # resolved at runtime
        "ms_file":       None,
        "paper_number":  None,
    },
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

# Module-level cached Haiku scorer — same Anthropic API key as Claude Opus,
# but claude-haiku-4-5 which is ~60× cheaper for simple mark-scheme comparisons.
_haiku_scorer_cache: Optional[object] = None   # ClaudeProvider instance


def _get_haiku_scorer(registry) -> object | None:
    """Get or create a Haiku scorer instance, cached for the lifetime of the process.

    Uses the same Anthropic API key as Claude Opus but with the Haiku model
    (claude-haiku-4-5-20251001 at $0.25/$1.25 per MTok vs Opus $15/$75).
    Benchmark scoring is simple mark-scheme comparison — Haiku is sufficient.
    """
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
"""


def _strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


async def _bench_score(
    registry,
    cost_tracker,
    question_text: str,
    marks: int,
    answer_text: str,
    ms_text: str,
    subject: str,
) -> tuple[int, int, str, list[str], list[str]]:
    """Score an answer against the MS using Claude.

    Returns (marks_achieved, total_marks, grade_estimate, hits, misses).
    On unrecoverable failure returns grade="SCORE_FAILED" so the report
    distinguishes a genuine 0 from a scoring API error.
    Retries up to 2 times with a 10-second delay on timeout/parse failure.
    """
    # Use Haiku for scoring — same API key as Opus but 60× cheaper for this task
    scorer = _get_haiku_scorer(registry)
    if not scorer:
        return 0, marks, "SCORE_FAILED", [], ["Claude/Haiku unavailable — scoring skipped"]

    # Truncate MS excerpt to keep prompt under ~16k chars
    ms_excerpt = ms_text[:12_000]
    if len(ms_text) > 12_000:
        ms_excerpt += "\n[...mark scheme truncated for token budget]"

    system = _BENCH_SCORE_SYSTEM.format(ms_excerpt=ms_excerpt)
    prompt = (
        f"Question [{marks} marks]:\n{question_text}\n\n"
        f"Student answer:\n{answer_text}"
    )

    last_error = "unknown"
    for attempt in range(3):   # 3 attempts total (initial + 2 retries)
        if attempt > 0:
            logger.info("Scoring retry %d/2 for Q (waiting 10s)…", attempt)
            await asyncio.sleep(10)

        try:
            resp = await scorer.generate(
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=1024,
            )
            cost_tracker.record(resp, subject, "benchmark_scoring")
        except Exception as exc:
            last_error = f"API error: {exc}"
            logger.warning("Scoring attempt %d failed: %s", attempt + 1, exc)
            continue   # retry

        if not resp.success:
            last_error = f"Model error: {resp.error}"
            logger.warning("Scoring attempt %d unsuccessful: %s", attempt + 1, resp.error)
            continue   # retry

        try:
            raw = _strip_json_fences(resp.content)
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON object found in response")
            data     = json.loads(raw[start:end])
            achieved = int(data.get("marks_achieved", 0))
            total    = int(data.get("total_marks", marks))
            grade    = str(data.get("grade_estimate", "?"))
            hits     = list(data.get("mark_points_hit", []))
            misses   = list(data.get("mark_points_missed", []))
            return achieved, total, grade, hits, misses   # success
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = f"Parse error: {exc}"
            logger.warning(
                "Score parse error on attempt %d: %s\n---\n%s",
                attempt + 1, exc, resp.content[:300],
            )
            # retry

    # All 3 attempts exhausted
    logger.warning("Scoring failed after 3 attempts. Last error: %s", last_error)
    return 0, marks, "SCORE_FAILED", [], [f"SCORE_FAILED after 3 attempts: {last_error}"]


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


# ── Paper benchmark ─────────────────────────────────────────────────────────────

async def run_paper_benchmark(
    paper_def: dict,
    kb_dir: Path,
    registry,
    config,
    cost_tracker,
    max_versions: int = 1,
    max_questions: int = 5,
) -> PaperResult:
    """Run the full benchmark for a single paper."""
    from shark_answer.config import Subject, Language, SUBJECT_PIPELINE_MAP
    from shark_answer.pipelines.pipeline_a_science import run_pipeline_a
    from shark_answer.pipelines.pipeline_b_essay import run_pipeline_b
    from shark_answer.pipelines.pipeline_c_cs import run_pipeline_c
    from shark_answer.pipelines.router import route_question
    from shark_answer.knowledge_base.retriever import build_prompt_context
    from shark_answer.utils.file_converter import convert_file_to_images
    from shark_answer.utils.image_extractor import extract_questions_from_images
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

    # ── 1. Load QP PDF ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    qp_bytes = _load_file_bytes(kb_dir, mkey, "question_papers", qp_file)
    if qp_bytes is None:
        result.error = f"QP not found: {kb_dir / mkey / 'question_papers' / qp_file}"
        return result

    # ── 2. Convert PDF → images ───────────────────────────────────────────────
    print(f"  [{skey}] Converting PDF to images…", flush=True)
    try:
        images = convert_file_to_images(qp_file, qp_bytes)
    except Exception as exc:
        result.error = f"PDF→image conversion failed: {exc}"
        return result

    if not images:
        result.error = "PDF produced no images"
        return result

    # ── 3. Extract questions ──────────────────────────────────────────────────
    print(f"  [{skey}] Extracting questions from {len(images)} page(s)…", flush=True)
    try:
        questions, _ = await extract_questions_from_images(registry, images)
    except Exception as exc:
        result.error = f"Question extraction failed: {exc}"
        return result

    result.timing.extraction_s = time.perf_counter() - t0

    if not questions:
        result.error = "No questions extracted from paper"
        return result

    # Drop questions with 0 marks — these are cover pages / instructions, not real questions
    real_questions = [q for q in questions if q.marks > 0]
    dropped = len(questions) - len(real_questions)
    if dropped:
        print(f"  [{skey}] Dropped {dropped} zero-mark item(s) (cover/instruction pages).",
              flush=True)
    questions = real_questions

    if not questions:
        result.error = "No scoreable questions extracted (all had 0 marks)"
        return result

    print(f"  [{skey}] {len(questions)} real question(s) found. "
          f"Processing up to {max_questions}.", flush=True)
    questions = questions[:max_questions]

    # ── 4. Load MS text ───────────────────────────────────────────────────────
    ms_text = _load_txt(kb_dir, mkey, "mark_schemes", ms_file)

    # ── 5. Get KB context (once per paper) ───────────────────────────────────
    kb_context = build_prompt_context(subject=skey, paper_number=pnum)

    # ── 6. Solve each question ────────────────────────────────────────────────
    cost_before_solving = cost_tracker.total_cost
    t_solve = time.perf_counter()

    for q in questions:
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
                )
            elif pipeline == Pipeline.ESSAY:
                pipeline_result = await run_pipeline_b(
                    question=q, subject=skey, registry=registry,
                    config=config, cost_tracker=cost_tracker,
                    kb_context=kb_context, language="en",
                    max_versions=max_versions,
                )
            elif pipeline == Pipeline.CS:
                pipeline_result = await run_pipeline_c(
                    question=q, subject=skey, registry=registry,
                    config=config, cost_tracker=cost_tracker,
                    kb_context=kb_context, language="en",
                    max_versions=max_versions,
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
        "--verbose", action="store_true",
        help="Enable debug logging",
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

    print(
        f"Benchmark: {len(papers)} paper(s), "
        f"versions={max_versions}, max_q={max_q}\n"
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

        result = await run_paper_benchmark(
            paper_def=resolved,
            kb_dir=kb_dir,
            registry=registry,
            config=config,
            cost_tracker=cost_tracker,
            max_versions=max_versions,
            max_questions=max_q,
        )
        results.append(result)

        summary = (
            f"  → {skey}: {result.total_achieved}/{result.total_marks} "
            f"({result.grade})  ${result.cost_usd:.4f}  "
            f"{result.timing.total_s:.0f}s"
        )
        if result.error:
            summary = f"  → {skey}: ERROR — {result.error}"
        print(summary, flush=True)

    elapsed = time.perf_counter() - t_global
    print(
        f"\nAll papers done in {elapsed:.1f}s. "
        f"Total cost: ${cost_tracker.total_cost:.4f}\n",
        flush=True,
    )

    # ── Generate report ────────────────────────────────────────────────────────
    report = generate_report(results, max_versions, max_q)
    print(report)

    # ── Save report ────────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quick" if args.quick else (args.subject or "all")
        out_path = _RESULTS_DIR / f"benchmark_{mode}_{ts}.txt"

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
