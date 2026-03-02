"""Pipeline A: Science & Math (Physics, Chemistry, Biology, Math, Further Math).

Models (2026):
  o3-pro         — primary, strongest math reasoning
  deepseek-v3.2  — primary, IMO/IOI-level STEM
  gemini-3.1-pro — primary, top GPQA science score
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
from shark_answer.knowledge_base.store import KnowledgeBase
from shark_answer.modules.examiner_profile import ExaminerProfile
from shark_answer.modules.explanation import build_explanation_prompt
from shark_answer.pipelines.base import AnswerVersion, PipelineResult
from shark_answer.providers.base import ModelResponse
from shark_answer.providers.registry import ProviderRegistry
from shark_answer.utils.cost_tracker import CostTracker
from shark_answer.utils.image_extractor import ExtractedQuestion
from shark_answer.utils.math_verifier import verify_numeric_agreement

logger = logging.getLogger(__name__)

SOLVE_SYSTEM = """You are an expert CIE A-Level examiner and tutor.
Solve the following question at A/A* standard.

Requirements:
- Show ALL working clearly (method marks matter)
- State formulas before substituting values
- Include correct SI units at every step
- For multi-part questions, clearly label each part
- Use proper mathematical notation
- If a diagram would help, describe it in [DIAGRAM: ...]
- Give the final answer with appropriate significant figures and units

{marking_context}
{examiner_guidance}"""

ALTERNATIVE_METHODS_SYSTEM = """You are an expert CIE A-Level tutor.
Solve this question using a DIFFERENT method than the one shown below.

The question has already been solved using: {existing_method}
You MUST use a completely different approach. For example:
- If they used energy conservation → use Newton's laws / kinematics
- If they used differentiation → use geometric / graphical approach
- If they used simultaneous equations → use matrix method or substitution
- If they used integration by parts → use substitution or partial fractions

Generate a COMPLETE solution at A/A* standard with full working.

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


async def run_pipeline_a(
    question: ExtractedQuestion,
    subject: str,
    registry: ProviderRegistry,
    config: AppConfig,
    cost_tracker: CostTracker,
    knowledge_base: Optional[KnowledgeBase] = None,
    examiner_profile: Optional[ExaminerProfile] = None,
    language: str = "en",
    max_versions: int = 5,
) -> PipelineResult:
    """Run Pipeline A for a science/math question."""
    result = PipelineResult(
        question=question,
        pipeline="A",
        subject=subject,
    )

    # Build context
    marking_context = ""
    if knowledge_base:
        marking_context = knowledge_base.get_marking_context(
            subject, question.topic_hints[0] if question.topic_hints else ""
        )

    examiner_guidance = ""
    if examiner_profile:
        examiner_guidance = examiner_profile.to_prompt_guidance()

    system_prompt = SOLVE_SYSTEM.format(
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

    logger.info("Pipeline A: %d models solving Q%s", len(primary_models), question.number)
    responses = await registry.call_models_parallel(
        providers=primary_models,
        prompt=question_prompt,
        system=system_prompt,
        temperature=0.3,
        max_tokens=4096,
    )
    cost_tracker.record_batch(responses, subject, "A")

    successful = [(m, r) for m, r in zip(primary_models, responses) if r.success]
    if not successful:
        result.errors.append("All primary models failed")
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

    # Step 4: Create answer versions
    # Version 1-N from primary model outputs
    for i, (model, resp) in enumerate(successful[:max_versions]):
        version = AnswerVersion(
            version_number=i + 1,
            answer_text=resp.content,
            provider=model.value,
            verified=is_calc and result.disagreement_resolved,
            language=language,
            approach_label=f"Primary solution ({model.value})",
        )
        result.versions.append(version)

    # Step 5: Generate alternative method versions if we have room
    remaining_slots = max_versions - len(result.versions)
    if remaining_slots > 0 and successful:
        best_answer = successful[0][1].content
        alt_system = ALTERNATIVE_METHODS_SYSTEM.format(
            existing_method="the approach shown below",
            marking_context=marking_context,
            examiner_guidance=examiner_guidance,
        )
        alt_prompt = (
            f"Original question:\n{question.text}\n\n"
            f"Already solved with this approach:\n{best_answer}\n\n"
            f"Now solve using a COMPLETELY DIFFERENT method.{lang_suffix}"
        )

        # Use the first available primary model for alternatives
        alt_model = primary_models[0]
        for j in range(remaining_slots):
            alt_response = await registry.call_models_parallel(
                providers=[alt_model],
                prompt=alt_prompt,
                system=alt_system,
                temperature=0.5 + j * 0.1,  # slight variation
                max_tokens=4096,
            )
            cost_tracker.record_batch(alt_response, subject, "A-alt")
            if alt_response[0].success:
                version = AnswerVersion(
                    version_number=len(result.versions) + 1,
                    answer_text=alt_response[0].content,
                    provider=alt_model.value,
                    language=language,
                    approach_label=f"Alternative method {j + 1}",
                )
                result.versions.append(version)

    # Step 6: Generate explanations for each version
    explain_model = primary_models[0]
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
