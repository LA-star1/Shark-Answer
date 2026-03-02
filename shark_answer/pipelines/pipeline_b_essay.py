"""Pipeline B: Essay Subjects (Economics, Biology essay questions).

Flow:
1. Angle generation: 5 models each brainstorm a DIFFERENT argument angle
2. Full draft: Each model writes a complete answer from its assigned angle
3. Quality gate: Judge scores drafts against CIE mark scheme criteria
4. Anti-AI detection: Humanize each version with different writing personality
5. Output: Up to 5 versions with genuinely different arguments
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider, Pipeline
from shark_answer.knowledge_base.store import KnowledgeBase
from shark_answer.modules.examiner_profile import ExaminerProfile
from shark_answer.modules.explanation import build_explanation_prompt
from shark_answer.pipelines.base import AnswerVersion, PipelineResult
from shark_answer.providers.registry import ProviderRegistry
from shark_answer.utils.cost_tracker import CostTracker
from shark_answer.utils.image_extractor import ExtractedQuestion

logger = logging.getLogger(__name__)

ANGLE_SYSTEM = """You are a CIE A-Level {subject} essay specialist.

Your task: brainstorm a UNIQUE argument angle for the question below.
You are Model #{model_index} of 5. Each model must take a DIFFERENT perspective.

Model perspective assignments:
- Model 1: Orthodox/mainstream textbook argument
- Model 2: Counter-intuitive or contrarian position
- Model 3: Real-world case study driven approach (specific country/company/event)
- Model 4: Theoretical deep-dive with evaluation of assumptions
- Model 5: Comparative/multi-perspective analysis

You are Model #{model_index}. Stay in your lane — do NOT overlap with other angles.

Output format (JSON):
{{
  "angle_title": "Brief title of your argument angle",
  "thesis": "Your main argument in 1-2 sentences",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "examples": ["Specific example 1", "Specific example 2"],
  "counter_argument": "Main counter to address",
  "evaluation_approach": "How you'll weigh the evidence"
}}"""

DRAFT_SYSTEM = """You are a CIE A-Level {subject} essay writer.
Write a COMPLETE essay answer at A/A* standard.

You MUST follow this specific argument angle:
{angle_json}

CIE A-Level marking criteria you MUST hit:
- Knowledge & Understanding: Accurate definitions, concepts, theories
- Application: Apply theory to the specific context of the question
- Analysis: Develop chains of reasoning (cause → effect → consequence)
- Evaluation: Weigh arguments, reach justified conclusions, consider significance

Requirements:
- Use precise subject terminology throughout
- Include at least 2 specific real-world examples with data/dates
- Write an introduction that directly addresses the question
- Each paragraph should have a clear point, evidence, and analysis
- Include counter-arguments with genuine evaluation (not just "however...")
- Conclusion must make a justified judgement

{marking_context}
{examiner_guidance}

Word count target: {word_target} words."""

JUDGE_SYSTEM = """You are a senior CIE A-Level {subject} examiner.
Score this essay against the CIE mark scheme criteria.

Score each criterion out of 25:
1. Knowledge & Understanding (KU): Definitions, theories, concepts
2. Application (AP): Applying theory to the specific question context
3. Analysis (AN): Chains of reasoning, cause-effect, logical development
4. Evaluation (EV): Weighing arguments, justified conclusions, nuance

Also assess:
5. Structure & Communication: Clear, logical, well-organized
6. Use of Examples: Specific, relevant, well-integrated

Output format (JSON):
{{
  "scores": {{"KU": 0, "AP": 0, "AN": 0, "EV": 0, "structure": 0, "examples": 0}},
  "total": 0,
  "grade": "A*/A/B/C",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "revision_instructions": "Specific instructions to improve to A/A* if needed"
}}"""

HUMANIZE_SYSTEM = """You are rewriting an A-Level essay to sound authentically human.
Apply personality #{personality_num}:

Personality 1: Confident and direct. Short punchy sentences mixed with longer analytical ones.
    Uses "I would argue that..." and "Crucially,..." — minimal hedging.
Personality 2: Methodical and cautious. Builds arguments step by step.
    Uses "It could be argued..." and "To some extent..." — academic hedging.
Personality 3: Example-driven storyteller. Leads with concrete cases before theory.
    Uses "Consider the case of..." and "This demonstrates..." — narrative style.
Personality 4: Debate-style. Presents strong thesis, anticipates objections, rebuts.
    Uses "Critics might contend..." and "Nevertheless,..." — adversarial clarity.
Personality 5: Reflective and nuanced. Emphasizes complexity and trade-offs.
    Uses "The reality is more nuanced..." and "While X is true, Y complicates..." — balanced.

You are Personality #{personality_num}. Rewrite the essay in this voice while:
- Keeping ALL the substantive content, arguments, and examples
- Maintaining A/A* quality
- Making it sound like a real student wrote it (natural flow, occasional imperfections)
- Varying sentence length naturally
- Avoiding AI tells: no "delve", "tapestry", "in conclusion", "it is important to note"
- Do NOT start paragraphs with "Furthermore", "Moreover", "Additionally" repeatedly

{examiner_guidance}"""


async def run_pipeline_b(
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
    """Run Pipeline B for an essay question."""
    result = PipelineResult(
        question=question,
        pipeline="B",
        subject=subject,
    )

    marking_context = ""
    if knowledge_base:
        marking_context = knowledge_base.get_marking_context(
            subject, question.topic_hints[0] if question.topic_hints else ""
        )

    examiner_guidance = ""
    if examiner_profile:
        examiner_guidance = examiner_profile.to_prompt_guidance()

    lang_suffix = "\n\nWrite your answer in Chinese (简体中文)." if language == "zh" else ""

    brainstorm_models = config.get_pipeline_models(Pipeline.ESSAY, "brainstorm")
    if not brainstorm_models:
        result.errors.append("No brainstorm models configured for Pipeline B")
        return result

    # Limit to max_versions models
    brainstorm_models = brainstorm_models[:max_versions]
    question_prompt = (
        f"Question [{question.marks} marks]:\n{question.text}\n\n"
        f"Brainstorm your unique argument angle.{lang_suffix}"
    )

    # ===== Step 1: Angle Generation (parallel) =====
    logger.info("Pipeline B: Generating %d angles for Q%s",
                len(brainstorm_models), question.number)

    import asyncio

    async def _get_angle(model: ModelProvider, index: int):
        inst = registry.get(model)
        if not inst:
            return model, None
        sys = ANGLE_SYSTEM.format(
            subject=subject, model_index=index + 1,
        )
        resp = await inst.generate(
            prompt=question_prompt, system=sys,
            temperature=0.7, max_tokens=1500,
        )
        cost_tracker.record(resp, subject, "B-angle")
        return model, resp

    angle_tasks = [_get_angle(m, i) for i, m in enumerate(brainstorm_models)]
    angle_results = await asyncio.gather(*angle_tasks)

    angles: list[tuple[ModelProvider, str]] = []
    for model, resp in angle_results:
        if resp and resp.success:
            angles.append((model, resp.content))

    if not angles:
        result.errors.append("All angle generation calls failed")
        return result

    # ===== Step 2: Full Draft (parallel) =====
    logger.info("Pipeline B: Writing %d drafts", len(angles))
    word_target = max(300, question.marks * 40)  # rough heuristic

    async def _write_draft(model: ModelProvider, angle_json: str, idx: int):
        inst = registry.get(model)
        if not inst:
            return model, None
        sys = DRAFT_SYSTEM.format(
            subject=subject,
            angle_json=angle_json,
            marking_context=marking_context,
            examiner_guidance=examiner_guidance,
            word_target=word_target,
        )
        prompt = f"Write your complete essay answer.{lang_suffix}"
        resp = await inst.generate(
            prompt=prompt, system=sys,
            temperature=0.6, max_tokens=6000,
        )
        cost_tracker.record(resp, subject, "B-draft")
        return model, resp

    draft_tasks = [_write_draft(m, a, i) for i, (m, a) in enumerate(angles)]
    draft_results = await asyncio.gather(*draft_tasks)

    drafts: list[tuple[ModelProvider, str, str]] = []  # (model, angle, draft)
    for (model_a, angle), (model_d, resp) in zip(angles, draft_results):
        if resp and resp.success:
            drafts.append((model_d, angle, resp.content))

    if not drafts:
        result.errors.append("All draft writing calls failed")
        return result

    # ===== Step 3: Quality Gate =====
    logger.info("Pipeline B: Judging %d drafts", len(drafts))
    judge_models = config.get_pipeline_models(Pipeline.ESSAY, "judge")
    judge_model = judge_models[0] if judge_models else brainstorm_models[0]

    passing_drafts: list[tuple[ModelProvider, str, str, float]] = []

    for model, angle, draft in drafts:
        judge_inst = registry.get(judge_model)
        if not judge_inst:
            passing_drafts.append((model, angle, draft, 75.0))
            continue

        judge_sys = JUDGE_SYSTEM.format(subject=subject)
        judge_prompt = f"Question:\n{question.text}\n\nEssay to evaluate:\n{draft}"
        judge_resp = await judge_inst.generate(
            prompt=judge_prompt, system=judge_sys,
            temperature=0.1, max_tokens=2000,
        )
        cost_tracker.record(judge_resp, subject, "B-judge")

        score = 75.0  # default
        if judge_resp.success:
            score = _extract_score(judge_resp.content)

        if score >= 70:  # A threshold
            passing_drafts.append((model, angle, draft, score))
        else:
            logger.info("Draft from %s scored %.1f (below A), requesting revision",
                        model.value, score)
            # Revision attempt
            revision_instructions = _extract_revision_instructions(judge_resp.content)
            inst = registry.get(model)
            if inst:
                revision_prompt = (
                    f"Revise this essay based on examiner feedback:\n\n"
                    f"FEEDBACK:\n{revision_instructions}\n\n"
                    f"ORIGINAL ESSAY:\n{draft}\n\n"
                    f"Rewrite to achieve A/A* standard.{lang_suffix}"
                )
                rev_resp = await inst.generate(
                    prompt=revision_prompt,
                    system=DRAFT_SYSTEM.format(
                        subject=subject, angle_json=angle,
                        marking_context=marking_context,
                        examiner_guidance=examiner_guidance,
                        word_target=word_target,
                    ),
                    temperature=0.4, max_tokens=6000,
                )
                cost_tracker.record(rev_resp, subject, "B-revision")
                if rev_resp.success:
                    passing_drafts.append((model, angle, rev_resp.content, score + 5))

    # ===== Step 4: Humanize with different writing personalities =====
    logger.info("Pipeline B: Humanizing %d drafts", len(passing_drafts))

    async def _humanize(model: ModelProvider, draft: str, personality: int):
        inst = registry.get(model)
        if not inst:
            return draft
        sys = HUMANIZE_SYSTEM.format(
            personality_num=personality,
            examiner_guidance=examiner_guidance,
        )
        prompt = f"Rewrite this essay in your personality voice:\n\n{draft}{lang_suffix}"
        resp = await inst.generate(
            prompt=prompt, system=sys,
            temperature=0.7, max_tokens=6000,
        )
        cost_tracker.record(resp, subject, "B-humanize")
        return resp.content if resp.success else draft

    humanize_tasks = [
        _humanize(model, draft, i + 1)
        for i, (model, _, draft, _) in enumerate(passing_drafts[:max_versions])
    ]
    humanized = await asyncio.gather(*humanize_tasks)

    # ===== Step 5: Build versions with explanations =====
    for i, ((model, angle, _, score), final_text) in enumerate(
        zip(passing_drafts[:max_versions], humanized)
    ):
        # Generate explanation
        explain_prompt = build_explanation_prompt(
            Pipeline.ESSAY, question.text, final_text, language
        )
        explain_model = judge_models[0] if judge_models else brainstorm_models[0]
        explain_resp = await registry.call_models_parallel(
            providers=[explain_model],
            prompt=explain_prompt,
            system="You are a CIE A-Level essay coach creating study explanations.",
            temperature=0.4, max_tokens=3000,
        )
        cost_tracker.record_batch(explain_resp, subject, "B-explain")
        explanation = explain_resp[0].content if explain_resp[0].success else ""

        version = AnswerVersion(
            version_number=i + 1,
            answer_text=final_text,
            explanation_text=explanation,
            approach_label=_extract_angle_title(angle),
            provider=model.value,
            quality_score=score,
            language=language,
        )
        result.versions.append(version)

    return result


def _extract_score(judge_output: str) -> float:
    """Extract total score from judge JSON output."""
    try:
        # Find JSON in output
        start = judge_output.find("{")
        end = judge_output.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(judge_output[start:end])
            return float(data.get("total", 75))
    except (json.JSONDecodeError, ValueError):
        pass
    return 75.0


def _extract_revision_instructions(judge_output: str) -> str:
    """Extract revision instructions from judge output."""
    try:
        start = judge_output.find("{")
        end = judge_output.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(judge_output[start:end])
            return data.get("revision_instructions", "Improve evaluation depth and add specific examples.")
    except (json.JSONDecodeError, ValueError):
        pass
    return "Improve evaluation depth and add specific examples."


def _extract_angle_title(angle_json: str) -> str:
    """Extract the angle title from the brainstorm output."""
    try:
        start = angle_json.find("{")
        end = angle_json.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(angle_json[start:end])
            return data.get("angle_title", "Essay argument")
    except (json.JSONDecodeError, ValueError):
        pass
    return "Essay argument"
