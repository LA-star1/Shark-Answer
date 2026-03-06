"""Pipeline C: Computer Science.

Models (2026):
  claude-opus-4.6  — primary + JUDGE (SWE-bench leader, best code reasoning)
  minimax-m2.5     — primary (strong software engineering)
  glm-5            — primary (top HumanEval score)
  deepseek-v3.2    — primary (strong coder, cost-efficient)

Flow:
1. All 4 primary models solve independently
2. For programming questions: execute and test code to verify
3. Format pseudocode to match CIE pseudocode conventions (Cambridge notation)
4. Generate up to 5 versions using different algorithms/approaches
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from shark_answer.config import AppConfig, Pipeline
from shark_answer.knowledge_base.predictor import build_prediction_context
from shark_answer.modules.examiner_profile import ExaminerProfile
from shark_answer.modules.explanation import build_explanation_prompt
from shark_answer.pipelines.base import AnswerVersion, PipelineResult
from shark_answer.providers.registry import ProviderRegistry, SOLVER_TIMEOUT
from shark_answer.utils.cost_tracker import CostTracker
from shark_answer.utils.image_extractor import ExtractedQuestion

logger = logging.getLogger(__name__)

CS_SOLVE_SYSTEM = """You are writing the FINAL ANSWER that a top A-Level student will copy word-for-word onto their CIE Computer Science exam paper. This must be a complete, ready-to-submit exam answer — NOT a lesson, NOT a tutorial.

CRITICAL — Do NOT include teaching preamble like "Here's how to approach this..." or "Let me explain...". Write ONLY the direct exam answer.

CIE Pseudocode Conventions (Cambridge notation) — MUST follow exactly:
- DECLARE x : INTEGER  (variable declarations)
- ← for assignment (not = or :=)
- IF...THEN...ELSE...ENDIF
- WHILE...DO...ENDWHILE
- FOR...TO...NEXT (or FOR...TO...STEP...NEXT)
- REPEAT...UNTIL
- CASE OF...OTHERWISE...ENDCASE
- PROCEDURE name(params) ... ENDPROCEDURE
- FUNCTION name(params) RETURNS type ... RETURN value ... ENDFUNCTION
- CALL for procedure calls
- OPENFILE, READFILE, WRITEFILE, CLOSEFILE
- INPUT and OUTPUT (not READ/PRINT)
- Array indices start at 1 (not 0) unless specified
- String: LENGTH(), SUBSTRING(), UCASE(), LCASE(), & for concatenation

For theory questions:
- Write precise definitions (exactly as they'd earn marks)
- Use [DIAGRAM: ...] for any required diagrams
- Give examples only where the mark scheme explicitly rewards them

CRITICAL LENGTH RULE — match answer length to marks allocated:
- [1] mark: 1 sentence maximum
- [2] marks: 2–4 lines
- [3–4] marks: 4–8 lines
- [5–6] marks (code/pseudocode): complete working solution, no extra explanation
- [8–10] marks: full solution with comments where needed
- [12–15] marks (extended): full solution + brief inline explanation
DO NOT pad. Write only what earns marks.

Answer style: {answer_style}

{marking_context}
{examiner_guidance}"""

CS_ALT_SYSTEM = """You are a CIE A-Level Computer Science expert.
Solve this question using a DIFFERENT algorithm/approach than shown below.

Already solved with: {existing_approach}

Use a completely different approach. For example:
- If they used linear search → use binary search
- If they used bubble sort → use insertion sort or merge sort
- If they used iteration → use recursion (or vice versa)
- If they used array → use linked list approach
- If they used stack → use queue-based approach

Follow CIE pseudocode conventions (Cambridge notation) as specified.

{marking_context}
{examiner_guidance}"""

CODE_VERIFY_PROMPT = """The following is a CIE pseudocode solution. Convert it to
executable Python code that I can run to verify correctness.

Pseudocode:
{pseudocode}

Test cases to include:
{test_cases}

Output ONLY valid Python code wrapped in ```python ... ```.
Include test assertions at the bottom. Print "ALL TESTS PASSED" if all assertions pass."""


async def run_pipeline_c(
    question: ExtractedQuestion,
    subject: str,
    registry: ProviderRegistry,
    config: AppConfig,
    cost_tracker: CostTracker,
    kb_context: str = "",
    examiner_profile: Optional[ExaminerProfile] = None,
    language: str = "en",
    max_versions: int = 5,
    paper: Optional[int] = None,
) -> PipelineResult:
    """Run Pipeline C for a CS question."""
    result = PipelineResult(
        question=question,
        pipeline="C",
        subject=subject,
    )

    # ── Build enriched context: predicted MS + historical reference ─────────
    prediction_ctx = build_prediction_context(
        subject=subject,
        paper=paper or 1,   # CS Paper 1 (theory) or Paper 2 (algorithms)
        question_text=question.text,
        marks=question.marks,
    )
    if prediction_ctx and kb_context:
        marking_context = prediction_ctx + "\n\n=== HISTORICAL REFERENCE ===\n" + kb_context
    elif prediction_ctx:
        marking_context = prediction_ctx
    else:
        # kb_context is pre-built by the caller (app.py) via knowledge_base.retriever
        marking_context = kb_context

    examiner_guidance = ""
    if examiner_profile:
        examiner_guidance = examiner_profile.to_prompt_guidance()

    lang_suffix = "\n\nProvide your answer in Chinese (简体中文), but keep code/pseudocode in English." if language == "zh" else ""

    VERSION_STYLES = [
        "Formal Academic — precise CIE-standard phrasing, full working",
        "Concise — same logic, minimal words, clear structure",
        "Natural Voice — written as a top student would naturally write it",
        "Alternative Algorithm — different data structure or algorithm approach",
        "Extended — maximum marks, every edge case handled explicitly",
    ]

    system_prompt = CS_SOLVE_SYSTEM.format(
        answer_style=VERSION_STYLES[0],
        marking_context=marking_context,
        examiner_guidance=examiner_guidance,
    )

    question_prompt = (
        f"Question {question.number} [{question.marks} marks]:\n"
        f"{question.text}{lang_suffix}"
    )

    # Step 1: All primary models solve (judge also included in primary list)
    primary = config.get_pipeline_models(Pipeline.CS, "primary")
    all_models = primary

    if not all_models:
        result.errors.append("No models configured for Pipeline C")
        return result

    logger.info("Pipeline C: %d models solving Q%s", len(all_models), question.number)
    responses = await registry.call_models_parallel(
        providers=all_models,
        prompt=question_prompt,
        system=system_prompt,
        temperature=0.3,
        max_tokens=4096,
    )
    cost_tracker.record_batch(responses, subject, "C")

    successful = [(m, r) for m, r in zip(all_models, responses) if r.success]
    if not successful:
        result.errors.append("All models failed for Pipeline C")
        return result

    # Step 2: For code questions, verify by execution
    is_code = question.question_type in ("code", "pseudocode")

    style_labels = [
        "V1 — Formal Academic",
        "V2 — Concise",
        "V3 — Natural Voice",
        "V4 — Alternative Algorithm",
        "V5 — Extended Working",
    ]

    for i, (model, resp) in enumerate(successful):
        verified = False
        if is_code:
            verified = await _verify_code(
                resp.content, question.text, registry, config, cost_tracker, subject
            )
        lbl = style_labels[i] if i < len(style_labels) else f"V{i+1} — Solution"
        version = AnswerVersion(
            version_number=len(result.versions) + 1,
            answer_text=resp.content,
            provider=model.value,
            verified=verified,
            language=language,
            approach_label=lbl,
        )
        result.versions.append(version)

    # Step 3: Generate alternative approach versions
    remaining = max_versions - len(result.versions)
    if remaining > 0 and successful:
        best = successful[0][1].content
        alt_system = CS_ALT_SYSTEM.format(
            existing_approach="the approach shown below",
            marking_context=marking_context,
            examiner_guidance=examiner_guidance,
        )
        alt_prompt = (
            f"Original question:\n{question.text}\n\n"
            f"Already solved:\n{best}\n\n"
            f"Solve using a DIFFERENT algorithm.{lang_suffix}"
        )

        for j in range(remaining):
            alt_model = primary[0] if primary else all_models[0]
            alt_resp = await registry.call_models_parallel(
                providers=[alt_model],
                prompt=alt_prompt,
                system=alt_system,
                temperature=0.5 + j * 0.1,
                max_tokens=4096,
            )
            cost_tracker.record_batch(alt_resp, subject, "C-alt")
            if alt_resp[0].success:
                verified = False
                if is_code:
                    verified = await _verify_code(
                        alt_resp[0].content, question.text,
                        registry, config, cost_tracker, subject,
                    )
                v_num = len(result.versions) + 1
                lbl = style_labels[v_num - 1] if v_num - 1 < len(style_labels) else f"V{v_num} — Alt {j+1}"
                version = AnswerVersion(
                    version_number=v_num,
                    answer_text=alt_resp[0].content,
                    provider=alt_model.value,
                    verified=verified,
                    language=language,
                    approach_label=lbl,
                )
                result.versions.append(version)

    # Step 4: Generate explanations
    explain_model = primary[0] if primary else all_models[0]
    for version in result.versions:
        explain_prompt = build_explanation_prompt(
            Pipeline.CS, question.text, version.answer_text, language
        )
        explain_resp = await registry.call_models_parallel(
            providers=[explain_model],
            prompt=explain_prompt,
            system="You are a CIE A-Level CS tutor creating study explanations.",
            temperature=0.4, max_tokens=3000,
        )
        cost_tracker.record_batch(explain_resp, subject, "C-explain")
        if explain_resp[0].success:
            version.explanation_text = explain_resp[0].content

    return result


async def _verify_code(
    answer_text: str,
    question_text: str,
    registry: ProviderRegistry,
    config: AppConfig,
    cost_tracker: CostTracker,
    subject: str,
) -> bool:
    """Verify a code answer by converting to Python and executing."""
    primary = config.get_pipeline_models(Pipeline.CS, "primary")
    if not primary:
        return False

    model = primary[0]
    inst = registry.get(model)
    if not inst:
        return False

    # Ask the model to convert pseudocode to testable Python
    prompt = CODE_VERIFY_PROMPT.format(
        pseudocode=answer_text,
        test_cases=f"Based on the question:\n{question_text}\nGenerate 3-5 test cases.",
    )
    try:
        resp = await asyncio.wait_for(
            inst.generate(
                prompt=prompt,
                system="Convert CIE pseudocode to runnable Python with test assertions.",
                temperature=0.1, max_tokens=3000,
            ),
            timeout=SOLVER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("[%s] C-verify timed out after %.0fs", model.value, SOLVER_TIMEOUT)
        return False
    cost_tracker.record(resp, subject, "C-verify")

    if not resp.success:
        return False

    # Extract Python code
    python_code = _extract_python_code(resp.content)
    if not python_code:
        return False

    # Execute in sandbox
    return await _execute_python_safely(python_code)


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown code fences."""
    lines = text.split("\n")
    code_lines: list[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```python"):
            in_block = True
            continue
        elif line.strip() == "```" and in_block:
            break
        elif in_block:
            code_lines.append(line)
    return "\n".join(code_lines)


async def _execute_python_safely(code: str, timeout: int = 10) -> bool:
    """Execute Python code in a subprocess with timeout."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        Path(tmp_path).unlink(missing_ok=True)

        if proc.returncode == 0:
            output = stdout.decode()
            if "ALL TESTS PASSED" in output:
                logger.info("Code verification: ALL TESTS PASSED")
                return True
            logger.warning("Code ran but tests didn't all pass: %s", output[:200])
            return False
        else:
            logger.warning("Code execution failed: %s", stderr.decode()[:200])
            return False

    except asyncio.TimeoutError:
        logger.warning("Code execution timed out after %ds", timeout)
        return False
    except Exception as e:
        logger.warning("Code execution error: %s", e)
        return False
