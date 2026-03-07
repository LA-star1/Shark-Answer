"""Pipeline A: Science & Math (Physics, Chemistry, Biology, Math, Further Math).

Models (2026):
  o3-mini        — primary, strong math reasoning (temp; revert to o3-pro after org verify)
  deepseek-v3.2  — primary, IMO/IOI-level STEM
  gemini-3.1-pro-preview — primary, top GPQA science score
  claude-opus-4.6— primary + JUDGE (arbitrates disputes)
  glm-5          — supplementary, strong all-rounder

Flow:
1. Up to 5 primary models solve each question independently
2. For calculation questions: auto-verify using Python (sympy / numerical checks)
3. If models disagree: flag, run focused debate → Claude judges final answer
4. If all agree and computation confirms: skip debate (save API calls)
5. Generate up to 5 answer versions with different solving methods
6. Each version includes detailed explanation
"""

from __future__ import annotations

import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider, Pipeline
from shark_answer.knowledge_base.predictor import build_prediction_context
from shark_answer.modules.examiner_profile import ExaminerProfile
from shark_answer.modules.explanation import build_explanation_prompt
from shark_answer.pipelines.base import AnswerVersion, PipelineResult
from shark_answer.providers.base import ModelResponse
from shark_answer.providers.registry import ProviderRegistry, SHORT_TIMEOUT
from shark_answer.utils.cost_tracker import CostTracker
from shark_answer.utils.image_extractor import ExtractedQuestion
from shark_answer.utils.math_verifier import verify_numeric_agreement

logger = logging.getLogger(__name__)

SOLVE_SYSTEM = """You are writing the FINAL ANSWER that a top A-Level student will copy word-for-word onto their CIE exam paper. This must be a complete, ready-to-submit exam answer — NOT a lesson, NOT a tutorial, NOT an explanation of the question.

CRITICAL — Do NOT include:
- "Here's how to approach this..." / "Let me explain..."
- Meta-commentary about the question
- Teaching or learning content
- Anything the student wouldn't write on their paper

DO write (exactly as it would appear on the exam paper):
- All required working steps with clear labels (a), (b)(i), etc.
- Formulas stated before substitution: F = ma → F = (2.0)(5.0) = 10 N
- Every numerical step shown (method marks)
- Correct SI units at every step and in the final answer
- Significant figures matching the question data
- Diagrams described inline as [DIAGRAM: ...]
- Final boxed answer per sub-part

CRITICAL LENGTH RULE — match answer length to marks allocated:
- [1] mark: 1 sentence maximum
- [2] marks: 2–4 lines
- [3–4] marks: 4–8 lines
- [5–6] marks (calculation): half page maximum, show all working
- [8–10] marks: half to one page
- [12–15] marks (essay-style): one to 1.5 pages maximum
DO NOT pad. Write only what earns marks.

Answer style: {answer_style}

{marking_context}
{examiner_guidance}"""

ALTERNATIVE_METHODS_SYSTEM = """You are writing an ALTERNATIVE exam answer using a completely different method.

The question was already solved using: {existing_method}
You MUST use a different mathematical approach:
- Energy conservation → Newton's laws / kinematics
- Differentiation → geometric / graphical method
- Simultaneous equations → matrix or substitution method
- Integration by parts → substitution or partial fractions

Write the answer exactly as a student would write it on the CIE exam paper. Full working, correct units, final answer with s.f. No teaching commentary.

CRITICAL LENGTH RULE — match answer length to marks allocated:
- [1] mark: 1 sentence maximum
- [2] marks: 2–4 lines
- [3–4] marks: 4–8 lines
- [5–6] marks (calculation): half page maximum, show all working
- [8–10] marks: half to one page
- [12–15] marks (essay-style): one to 1.5 pages maximum
DO NOT pad. Write only what earns marks.

Answer style: {answer_style}

{marking_context}
{examiner_guidance}"""

DEBATE_SYSTEM = """You are resolving a disagreement between AI models on a CIE A-Level question.

Model answers received:
{model_answers}

The models DISAGREE on the answer. Analyze each approach:
1. Identify where each model's reasoning diverges
2. Check each model's arithmetic and formula application
3. Determine which answer is correct (or if a new approach is needed)
4. Provide the verified correct solution with full working

Output the CORRECT solution with clear explanation of where the error occurred."""



def _get_models_for_marks(
    marks: int,
    config: "AppConfig",
) -> tuple[list["ModelProvider"], float | None]:
    """Return (model_list, timeout_seconds) based on question mark count.

    Smart routing: appropriate models per mark count.
    NEVER hard-caps timeouts — always returns None so that per-model
    defaults from MODEL_TIMEOUTS are used (o3-pro=180s, deepseek=120s, …).
    Reasoning models must never be cut short regardless of question size.
    """
    from shark_answer.config import Pipeline

    if marks == 1:
        # 1-mark: fast models only — o3-pro is overkill for single-mark questions
        order = [ModelProvider.CLAUDE, ModelProvider.DEEPSEEK, ModelProvider.GEMINI]
    elif marks <= 3:
        # 2-3 marks: include o3-pro for multi-step short questions
        order = [
            ModelProvider.O3PRO,    ModelProvider.CLAUDE,
            ModelProvider.DEEPSEEK, ModelProvider.GEMINI,
        ]
    elif marks <= 6:
        # Medium questions (4-6m): full reasoning ensemble
        order = [
            ModelProvider.O3PRO,    ModelProvider.DEEPSEEK,
            ModelProvider.QWEN,     ModelProvider.CLAUDE,
            ModelProvider.GEMINI,
        ]
    else:
        # Long/hard questions (7+ m): full arsenal
        order = [
            ModelProvider.O3PRO,    ModelProvider.DEEPSEEK,
            ModelProvider.QWEN,     ModelProvider.CLAUDE,
            ModelProvider.GEMINI,
        ]

    available = config.get_available_models(order)
    if not available:
        # Fallback: whatever is configured for Pipeline A primary
        available = config.get_pipeline_models(Pipeline.SCIENCE_MATH, "primary")
    # NEVER hard-cap timeouts: return None so per-model defaults (MODEL_TIMEOUTS) apply
    return available, None

async def run_pipeline_a(
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
    full_paper_text: str = "",
) -> PipelineResult:
    """Run Pipeline A for a science/math question."""
    result = PipelineResult(
        question=question,
        pipeline="A",
        subject=subject,
    )

    # ── Build enriched context: predicted MS + historical reference ─────────
    # build_prediction_context() returns "" safely if no patterns file exists yet.
    prediction_ctx = build_prediction_context(
        subject=subject,
        paper=paper or 1,
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

    # ── Prepend full paper text so models can look up tables/figures ──────────
    if full_paper_text:
        paper_ctx = (
            "=== QUESTION PAPER TEXT (use any tables, figures, or data referenced "
            "in the question from this source) ===\n"
            + full_paper_text[:8000]
            + "\n"
        )
        marking_context = paper_ctx + (("\n" + marking_context) if marking_context else "")

    examiner_guidance = ""
    if examiner_profile:
        examiner_guidance = examiner_profile.to_prompt_guidance()

    # Version style labels (V1=Formal, V2=Concise, V3=Natural Voice)
    VERSION_STYLES = [
        "Formal Academic — precise, complete, examiner-approved phrasing",
        "Concise — same key steps, half the words, bullet-point format where allowed",
        "Natural Voice — written as a top student would naturally write it in an exam",
        "Alternative Method — different mathematical approach, same final answer",
        "Extended Working — maximum marks shown, every assumption stated explicitly",
    ]

    system_prompt = SOLVE_SYSTEM.format(
        answer_style=VERSION_STYLES[0],
        marking_context=marking_context,
        examiner_guidance=examiner_guidance,
    )

    lang_suffix = "\n\nProvide your answer in Chinese (简体中文)." if language == "zh" else ""
    question_prompt = f"Question {question.number} [{question.marks} marks]:\n{question.text}{lang_suffix}"

    # Step 1: 3 primary models solve independently
    primary_models = config.get_pipeline_models(Pipeline.SCIENCE_MATH, "primary")
    if not primary_models:
        result.errors.append("No primary models configured for Pipeline A")
        return result

    # ── Route by marks ─────────────────────────────────────────────────────
    # SHORT PATH (1–3 marks): smart-routed models, no debate, no alternative methods.
    if question.marks <= 3:
        short_models, short_timeout = _get_models_for_marks(question.marks, config)
        logger.info("Pipeline A [SHORT path, %dm]: %d models solving Q%s (timeout=%s)",
                    question.marks, len(short_models), question.number, short_timeout)
        responses = await registry.call_models_parallel(
            providers=short_models,
            prompt=question_prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=1024,
            timeout=short_timeout,
        )
        cost_tracker.record_batch(responses, subject, "A")
        # Filter out both API failures (r.success=False) AND empty responses
        # (r.content="" happens when a model returns a success status but no text,
        # e.g. Gemini returning None content on ambiguous vision-extracted questions).
        successful = [(m, r) for m, r in zip(short_models, responses)
                      if r.success and r.content.strip()]
        if not successful:
            result.errors.append("Short path: all models failed or returned empty")
            return result
        for i, (model, resp) in enumerate(successful):
            result.versions.append(AnswerVersion(
                version_number=i + 1,
                answer_text=resp.content,
                provider=model.value,
                language=language,
                approach_label="Direct Answer",
            ))
        # One brief explanation for the first version
        if result.versions:
            ep = build_explanation_prompt(
                Pipeline.SCIENCE_MATH, question.text, result.versions[0].answer_text, language
            )
            ep_resp = await registry.call_models_parallel(
                providers=[primary_models[0]],
                prompt=ep,
                system="You are a CIE A-Level tutor. Write a brief study note.",
                temperature=0.4,
                max_tokens=800,
                # timeout=None → per-model defaults
            )
            cost_tracker.record_batch(ep_resp, subject, "A-explain")
            if ep_resp[0].success:
                result.versions[0].explanation_text = ep_resp[0].content
        return result

    # FULL PATH (4+ marks): smart-routed models, optional debate, alt methods
    smart_models, smart_timeout = _get_models_for_marks(question.marks, config)
    logger.info("Pipeline A [FULL path, %dm]: %d models solving Q%s (timeout=%s)",
                question.marks, len(smart_models), question.number, smart_timeout)
    responses = await registry.call_models_parallel(
        providers=smart_models,
        prompt=question_prompt,
        system=system_prompt,
        temperature=0.3,
        max_tokens=4096,
        timeout=smart_timeout,
    )
    cost_tracker.record_batch(responses, subject, "A")

    successful = [(m, r) for m, r in zip(smart_models, responses)
                  if r.success and r.content.strip()]
    if not successful:
        result.errors.append("All primary models failed or returned empty")
        return result

    # Step 2: Check agreement (for calculation questions)
    is_calc = question.question_type in ("calculation", "proof")
    answers_text = [r.content for _, r in successful]

    if is_calc and len(successful) >= 2:
        agreed, values = verify_numeric_agreement(answers_text)
        if agreed:
            logger.info("Q%s: All models agree (values: %s). Skipping debate.",
                        question.number, values)
            result.verification_notes = f"All models agree: {values}"
        else:
            logger.info("Q%s: Models DISAGREE (values: %s). Running debate.",
                        question.number, values)
            result.disagreement_resolved = False

            # Step 3: Debate round between disagreeing models
            model_answers_text = "\n\n".join(
                f"--- {m.value} ---\n{r.content}" for m, r in successful
            )
            debate_prompt = f"Question:\n{question.text}\n\nResolve the disagreement and provide the correct answer."
            debate_system = DEBATE_SYSTEM.format(model_answers=model_answers_text)

            judge_models = config.get_pipeline_models(Pipeline.SCIENCE_MATH, "judge")
            if judge_models:
                debate_responses = await registry.call_models_parallel(
                    providers=judge_models,
                    prompt=debate_prompt,
                    system=debate_system,
                    temperature=0.1,
                    max_tokens=4096,
                )
                cost_tracker.record_batch(debate_responses, subject, "A-debate")

                for dr in debate_responses:
                    if dr.success:
                        # Replace the first answer with the debate-resolved answer
                        successful[0] = (judge_models[0], dr)
                        result.disagreement_resolved = True
                        result.verification_notes = (
                            f"Disagreement resolved via debate. Original values: {values}"
                        )
                        break

    # Step 4: Create answer versions (each model generates with its own style prompt)
    style_labels = [
        "V1 — Formal Academic",
        "V2 — Concise",
        "V3 — Natural Voice",
        "V4 — Alternative Method",
        "V5 — Extended Working",
    ]

    for i, (model, resp) in enumerate(successful[:max_versions]):
        style_idx = min(i, len(VERSION_STYLES) - 1)
        version = AnswerVersion(
            version_number=i + 1,
            answer_text=resp.content,
            provider=model.value,
            verified=is_calc and result.disagreement_resolved,
            language=language,
            approach_label=style_labels[i] if i < len(style_labels) else f"V{i+1}",
        )
        result.versions.append(version)

    # Step 5: Generate alternative method versions if we have room
    remaining_slots = max_versions - len(result.versions)
    if remaining_slots > 0 and successful:
        best_answer = successful[0][1].content
        for j in range(remaining_slots):
            style_idx = len(result.versions)
            style = VERSION_STYLES[min(style_idx, len(VERSION_STYLES) - 1)]
            alt_system = ALTERNATIVE_METHODS_SYSTEM.format(
                existing_method="the approach shown below",
                answer_style=style,
                marking_context=marking_context,
                examiner_guidance=examiner_guidance,
            )
            alt_prompt = (
                f"Original question:\n{question.text}\n\n"
                f"Already solved with this approach:\n{best_answer}\n\n"
                f"Now solve using a COMPLETELY DIFFERENT method.{lang_suffix}"
            )
            alt_model = smart_models[0] if smart_models else primary_models[0]
            alt_response = await registry.call_models_parallel(
                providers=[alt_model],
                prompt=alt_prompt,
                system=alt_system,
                temperature=0.5 + j * 0.1,
                max_tokens=4096,
            )
            cost_tracker.record_batch(alt_response, subject, "A-alt")
            if alt_response[0].success:
                v_num = len(result.versions) + 1
                lbl = style_labels[v_num - 1] if v_num - 1 < len(style_labels) else f"V{v_num} — Alt {j+1}"
                version = AnswerVersion(
                    version_number=v_num,
                    answer_text=alt_response[0].content,
                    provider=alt_model.value,
                    language=language,
                    approach_label=lbl,
                )
                result.versions.append(version)

    # Step 6: Generate explanations for each version
    explain_model = (smart_models[0] if 'smart_models' in dir() and smart_models
                     else primary_models[0])
    for version in result.versions:
        explain_prompt = build_explanation_prompt(
            Pipeline.SCIENCE_MATH, question.text, version.answer_text, language
        )
        explain_resp = await registry.call_models_parallel(
            providers=[explain_model],
            prompt=explain_prompt,
            system="You are a CIE A-Level tutor creating study explanations.",
            temperature=0.4,
            max_tokens=3000,
        )
        cost_tracker.record_batch(explain_resp, subject, "A-explain")
        if explain_resp[0].success:
            version.explanation_text = explain_resp[0].content

    return result
