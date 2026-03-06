"""Pipeline B: Essay Subjects (Economics, Biology essay questions).

Models (2026) with FIXED argument angles:
  claude-opus-4.6  → Orthodox/mainstream view  +  JUDGE (scores all drafts)
  gpt-5.2-thinking → Critical/contrarian perspective
  gemini-3.1-pro-preview → Case-study driven (real-world examples)
  deepseek-v3.2    → Theoretical/academic framework
  qwen3-max        → Comparative analysis (cross-country/cross-policy)
  glm-5            → Policy-oriented perspective
  kimi-k2.5        → Data/evidence-based empirical approach
  minimax-m2.5     → Backup/reserve

Flow:
1. Each assigned model brainstorms from its FIXED angle (no overlap)
2. Each model writes a full draft in its angle voice
3. Claude (judge) scores every draft against CIE mark scheme criteria
4. Below-A drafts get one revision attempt
5. Humanize each passing draft with a different writing personality
6. Output: up to 7 versions with genuinely different arguments
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider, Pipeline, PIPELINE_CONFIG

# Judge / explain fallback chain — tried in order until one succeeds.
# Claude is primary (best essay scoring); GPT-4o and Gemini are hot standbys.
JUDGE_FALLBACK_CHAIN: list[ModelProvider] = [
    ModelProvider.CLAUDE,
    ModelProvider.GPT4O,
    ModelProvider.GEMINI,
]
from shark_answer.knowledge_base.predictor import build_prediction_context
from shark_answer.modules.examiner_profile import ExaminerProfile
from shark_answer.modules.explanation import build_explanation_prompt
from shark_answer.pipelines.base import AnswerVersion, PipelineResult
from shark_answer.providers.registry import ProviderRegistry
from shark_answer.utils.cost_tracker import CostTracker
from shark_answer.utils.image_extractor import ExtractedQuestion

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model → assigned argument angle  (mirrors PIPELINE_CONFIG["B_essay"]["angles"])
# ──────────────────────────────────────────────────────────────────────────────
_ESSAY_MODEL_ANGLES: dict[ModelProvider, str] = PIPELINE_CONFIG["B_essay"]["angles"]

# Human-readable angle descriptions for display labels
_ANGLE_DISPLAY: dict[str, str] = {
    "orthodox/mainstream textbook view":            "Mainstream / Textbook",
    "critical/contrarian perspective":              "Critical / Contrarian",
    "case-study driven (real-world examples)":      "Case-Study Driven",
    "theoretical/academic framework":               "Theoretical Framework",
    "comparative analysis (cross-country/cross-policy)": "Comparative Analysis",
    "policy-oriented perspective":                  "Policy-Oriented",
    "data/evidence-based empirical approach":       "Data & Evidence",
}

# ──────────────────────────────────────────────────────────────────────────────
# KB context budget: premium models get full 29k context (~$0.44/call input);
# budget models get a 3k excerpt (~750 tokens, ~$0.003/call at Kimi rates).
# The context is ordered by importance so the first 3k chars contain the
# subject summary and the most relevant mark-scheme content.
# ──────────────────────────────────────────────────────────────────────────────
_PREMIUM_KB_MODELS: frozenset[ModelProvider] = frozenset({
    ModelProvider.CLAUDE,
    ModelProvider.GPT4O,
})
_BUDGET_KB_CHARS: int = 3_000


def _trim_kb_context(kb_context: str, max_chars: int = _BUDGET_KB_CHARS) -> str:
    """Return a shorter KB context excerpt for budget solver models.

    The retriever builds context in priority order (subject summary → mark schemes
    → examiner reports), so the first 3 k chars contain the most useful content.
    """
    if not kb_context or len(kb_context) <= max_chars:
        return kb_context
    truncated = kb_context[:max_chars]
    # Snap to last paragraph boundary to avoid cutting mid-sentence
    last_para = truncated.rfind("\n\n")
    if last_para > int(max_chars * 0.7):
        truncated = truncated[:last_para]
    return truncated + "\n\n[...context trimmed — budget model receives summary only]"


ANGLE_SYSTEM = """You are a CIE A-Level {subject} essay specialist.

Your task: brainstorm a UNIQUE argument angle for the question below.
You have been assigned the following perspective — you MUST argue from it:

  ASSIGNED ANGLE: {angle}

Do NOT overlap with, or drift towards, any other perspective. Stay strictly
within your assigned viewpoint.

Output format (JSON):
{{
  "angle_title": "Brief title that captures your specific angle",
  "thesis": "Your main argument in 1-2 sentences",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "examples": ["Specific example 1", "Specific example 2"],
  "counter_argument": "Main counter to address",
  "evaluation_approach": "How you'll weigh the evidence"
}}"""

DRAFT_SYSTEM = """You are writing the FINAL ESSAY ANSWER that a top A-Level student will copy word-for-word onto their CIE exam paper.

CRITICAL — Do NOT include:
- "Here's my approach..." / "Let me explain..."
- Commentary about the question or task
- Teaching content or meta-discussion

Write ONLY the complete exam essay, starting directly with your introduction paragraph.

Your assigned argument angle:
{angle_json}

CIE A-Level marking criteria (MUST hit all four for A*):
- Knowledge & Understanding: Accurate definitions, theories, key concepts
- Application: Apply theory directly to the question's specific context
- Analysis: Develop chains of reasoning (cause → effect → consequence)
- Evaluation: Weigh evidence, reach justified conclusions with nuance

Essay requirements (exactly as it would appear on the exam paper):
- Precise subject terminology throughout
- Minimum 2 specific real-world examples with data/dates
- Introduction that directly answers the question (state your line of argument)
- Each body paragraph: clear point → evidence → analysis → link back
- Counter-arguments addressed with genuine evaluation
- Conclusion with justified judgement (not a summary — a verdict)

CRITICAL LENGTH RULE: Write exactly {word_target} words. Do NOT exceed this. Do NOT pad with repetition or summary restatements. Every sentence must earn marks.

{marking_context}
{examiner_guidance}
"""

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
Personality 6: Data-forward. Opens with statistics, quantifies every claim.
    Uses "Evidence suggests..." and "According to data from..." — empirical tone.
Personality 7: Institutional/policy. Frames everything around real-world policy implications.
    Uses "Policymakers must consider..." and "The regulatory context suggests..." — applied.

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
    kb_context: str = "",
    examiner_profile: Optional[ExaminerProfile] = None,
    language: str = "en",
    max_versions: int = 7,
    paper: Optional[int] = None,
) -> PipelineResult:
    """Run Pipeline B for an essay question."""
    result = PipelineResult(
        question=question,
        pipeline="B",
        subject=subject,
    )

    # ── Build enriched context: predicted MS + historical reference ─────────
    prediction_ctx = build_prediction_context(
        subject=subject,
        paper=paper or 2,   # essays are almost always Paper 2-4
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

    lang_suffix = "\n\nWrite your answer in Chinese (简体中文)." if language == "zh" else ""

    # Build ordered list of (model, angle) for available configured models
    all_angle_models = config.get_available_models(list(_ESSAY_MODEL_ANGLES.keys()))
    if not all_angle_models:
        # Fall back to any available brainstorm models
        all_angle_models = config.get_pipeline_models(Pipeline.ESSAY, "brainstorm")
    if not all_angle_models:
        result.errors.append("No brainstorm models configured for Pipeline B")
        return result

    # ALL configured models always participate in angle + draft generation.
    # max_versions only limits the FINAL output (how many versions the user sees).
    # This ensures cross-model quality competition even when versions=1.
    angle_models = all_angle_models
    word_target = max(300, question.marks * 40)

    question_prompt = (
        f"Question [{question.marks} marks]:\n{question.text}\n\n"
        f"Brainstorm your unique argument angle.{lang_suffix}"
    )

    # ===== Step 1: Angle Generation (parallel, each model gets its fixed angle) =====
    logger.info("Pipeline B: Generating %d angles for Q%s",
                len(angle_models), question.number)

    async def _get_angle(model: ModelProvider) -> tuple[ModelProvider, str | None]:
        inst = registry.get(model)
        if not inst:
            return model, None
        assigned_angle = _ESSAY_MODEL_ANGLES.get(model, "balanced multi-perspective analysis")
        sys = ANGLE_SYSTEM.format(subject=subject, angle=assigned_angle)
        resp = await inst.generate(
            prompt=question_prompt, system=sys,
            temperature=0.7, max_tokens=1500,
        )
        cost_tracker.record(resp, subject, "B-angle")
        return model, (resp.content if resp.success else None)

    angle_tasks = [_get_angle(m) for m in angle_models]
    angle_results: list[tuple[ModelProvider, str | None]] = await asyncio.gather(*angle_tasks)

    angles: list[tuple[ModelProvider, str]] = [
        (m, content) for m, content in angle_results if content is not None
    ]
    if not angles:
        result.errors.append("All angle generation calls failed")
        return result

    # ===== Step 2: Full Draft (parallel) =====
    logger.info("Pipeline B: Writing %d drafts", len(angles))

    async def _write_draft(model: ModelProvider, angle_json: str) -> tuple[ModelProvider, str | None]:
        inst = registry.get(model)
        if not inst:
            return model, None
        # Premium models (Claude, GPT) receive the full 29k KB context.
        # Budget models receive a 3k excerpt to cut input-token cost ~10×.
        ctx = marking_context if model in _PREMIUM_KB_MODELS else _trim_kb_context(marking_context)
        sys = DRAFT_SYSTEM.format(
            subject=subject,
            angle_json=angle_json,
            marking_context=ctx,
            examiner_guidance=examiner_guidance,
            word_target=word_target,
        )
        prompt = f"Write your complete essay answer.{lang_suffix}"
        resp = await inst.generate(
            prompt=prompt, system=sys,
            temperature=0.6, max_tokens=6000,
        )
        cost_tracker.record(resp, subject, "B-draft")
        return model, (resp.content if resp.success else None)

    draft_tasks = [_write_draft(m, a) for m, a in angles]
    draft_results: list[tuple[ModelProvider, str | None]] = await asyncio.gather(*draft_tasks)

    # (model, angle_json, draft_text)
    drafts: list[tuple[ModelProvider, str, str]] = [
        (m, angles[i][1], draft)
        for i, (m, draft) in enumerate(draft_results)
        if draft is not None
    ]
    if not drafts:
        result.errors.append("All draft writing calls failed")
        return result

    # ===== Step 3: Quality Gate (judge scores each draft via fallback chain) =====
    # JUDGE_FALLBACK_CHAIN = [Claude, GPT-4o, Gemini] — first available succeeds.
    # Phase A: all judge calls run in PARALLEL (asyncio.gather) so latency is
    #          bounded by the single slowest call, not N×slowest.
    # Phase B: revisions for below-A drafts run sequentially (usually 0-1 drafts).
    # If all three judge providers are down the draft passes with a default score
    # so the pipeline never blocks completely on judge unavailability.
    logger.info("Pipeline B: Judging %d drafts in parallel (chain: %s)",
                len(drafts), " → ".join(p.value for p in JUDGE_FALLBACK_CHAIN))

    judge_sys = JUDGE_SYSTEM.format(subject=subject)

    async def _judge_one(
        model: ModelProvider, angle: str, draft: str
    ) -> tuple[ModelProvider, str, str, object]:
        """Judge a single draft; returns (model, angle, draft, judge_resp)."""
        j_prompt = f"Question:\n{question.text}\n\nEssay to evaluate:\n{draft}"
        j_resp = await registry.call_with_fallback(
            JUDGE_FALLBACK_CHAIN,
            prompt=j_prompt,
            system=judge_sys,
            temperature=0.1,
            max_tokens=2000,
        )
        cost_tracker.record(j_resp, subject, "B-judge")
        return model, angle, draft, j_resp

    # Phase A — parallel judge pass
    judge_tasks = [_judge_one(m, a, d) for m, a, d in drafts]
    judge_results: list[tuple] = await asyncio.gather(*judge_tasks)

    # Phase B — process results; revise below-A drafts (sequential, usually few)
    passing_drafts: list[tuple[ModelProvider, str, str, float]] = []

    for model, angle, draft, judge_resp in judge_results:
        if not judge_resp.success:
            logger.warning("All judge providers failed for draft from %s; passing with default score",
                           model.value)
            passing_drafts.append((model, angle, draft, 75.0))
            continue

        score = _extract_score(judge_resp.content)

        if score >= 70:  # A-grade threshold
            passing_drafts.append((model, angle, draft, score))
        else:
            logger.info("Draft from %s scored %.1f (below A), requesting revision",
                        model.value, score)
            revision_instructions = _extract_revision_instructions(judge_resp.content)
            inst = registry.get(model)
            if inst:
                revision_prompt = (
                    f"Revise this essay based on examiner feedback:\n\n"
                    f"FEEDBACK:\n{revision_instructions}\n\n"
                    f"ORIGINAL ESSAY:\n{draft}\n\n"
                    f"Rewrite to achieve A/A* standard.{lang_suffix}"
                )
                rev_ctx = marking_context if model in _PREMIUM_KB_MODELS else _trim_kb_context(marking_context)
                rev_resp = await inst.generate(
                    prompt=revision_prompt,
                    system=DRAFT_SYSTEM.format(
                        subject=subject, angle_json=angle,
                        marking_context=rev_ctx,
                        examiner_guidance=examiner_guidance,
                        word_target=word_target,
                    ),
                    temperature=0.4, max_tokens=6000,
                )
                cost_tracker.record(rev_resp, subject, "B-revision")
                if rev_resp.success:
                    passing_drafts.append((model, angle, rev_resp.content, score + 5))

    # Sort by score (best first) before limiting to max_versions for output.
    # This ensures versions=1 always outputs the highest-scoring draft.
    passing_drafts.sort(key=lambda x: x[3], reverse=True)

    # ===== Step 4: Humanize with model-matched writing personalities =====
    logger.info("Pipeline B: Humanizing %d drafts (keeping top %d)", len(passing_drafts), max_versions)

    async def _humanize(model: ModelProvider, draft: str, personality: int) -> str:
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
    humanized: list[str] = await asyncio.gather(*humanize_tasks)

    # ===== Step 5: Build versions with explanations =====
    # Explanations use the same fallback chain as judging, run in PARALLEL so
    # N explanations cost only one round-trip latency instead of N.
    top_versions = list(zip(passing_drafts[:max_versions], humanized))

    async def _explain_one(final_text: str) -> str:
        e_prompt = build_explanation_prompt(Pipeline.ESSAY, question.text, final_text, language)
        e_resp = await registry.call_with_fallback(
            JUDGE_FALLBACK_CHAIN,
            prompt=e_prompt,
            system="You are a CIE A-Level essay coach creating study explanations.",
            temperature=0.4,
            max_tokens=3000,
        )
        cost_tracker.record(e_resp, subject, "B-explain")
        return e_resp.content if e_resp.success else ""

    explanations: list[str] = await asyncio.gather(
        *[_explain_one(ft) for (_, _, _, _), ft in top_versions]
    )

    for i, (((model, angle, _, score), final_text), explanation) in enumerate(
        zip(top_versions, explanations)
    ):
        # Derive display label from angle or model
        assigned_angle = _ESSAY_MODEL_ANGLES.get(model, "")
        label = _ANGLE_DISPLAY.get(assigned_angle, None) or _extract_angle_title(angle)

        version = AnswerVersion(
            version_number=i + 1,
            answer_text=final_text,
            explanation_text=explanation,
            approach_label=label,
            provider=model.value,
            quality_score=score,
            language=language,
        )
        result.versions.append(version)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_score(judge_output: str) -> float:
    """Extract total score from judge JSON output."""
    try:
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
            return data.get(
                "revision_instructions",
                "Improve evaluation depth and add specific examples.",
            )
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
