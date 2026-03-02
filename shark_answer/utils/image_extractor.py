"""Extract exam questions from photos using AI vision."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shark_answer.config import ModelProvider
from shark_answer.providers.base import ModelResponse
from shark_answer.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are an expert at reading CIE A-Level exam papers.
Extract ALL questions from the image with perfect accuracy.

Rules:
1. Preserve ALL mathematical formulas in LaTeX notation (e.g., $F = ma$, $\\int_0^1 x^2 dx$)
2. Describe any diagrams, circuit diagrams, or graphs in detail within [DIAGRAM: description]
3. Preserve data tables as markdown tables
4. Number each question and sub-question exactly as shown (e.g., 1(a)(i))
5. Include mark allocations in square brackets, e.g. [3 marks]
6. If a question references a diagram/figure, describe what it shows
7. Preserve units and significant figures exactly as printed

Output format — return a JSON array:
[
  {
    "number": "1(a)(i)",
    "text": "Full question text with $LaTeX$ formulas",
    "marks": 3,
    "has_diagram": true,
    "diagram_description": "A circuit diagram showing...",
    "question_type": "calculation|explanation|essay|diagram|code|proof",
    "topic_hints": ["mechanics", "kinematics"]
  }
]
"""

EXTRACTION_PROMPT = """Extract ALL questions from this exam paper image.
Return valid JSON only. Be extremely precise with formulas, numbers, and units.
If the image is unclear in parts, note it with [UNCLEAR: description]."""

# Vision-capable providers tried in priority order for extraction.
# NOTE: DeepSeek-chat and Qwen-plus do NOT support image inputs — excluded.
# Grok uses grok-2-vision-1212 which does support images.
_VISION_PROVIDER_PRIORITY = [
    ModelProvider.CLAUDE,
    ModelProvider.GPT4O,
    ModelProvider.GEMINI,
    ModelProvider.GROK,
]


@dataclass
class ExtractedQuestion:
    """A single question extracted from an exam paper."""
    number: str
    text: str
    marks: int = 0
    has_diagram: bool = False
    diagram_description: str = ""
    question_type: str = "calculation"  # calculation, explanation, essay, diagram, code, proof
    topic_hints: list[str] = field(default_factory=list)


def _first_available_vision_provider(registry: ProviderRegistry) -> ModelProvider | None:
    """Return the first configured vision provider, or None if none are set up."""
    for p in _VISION_PROVIDER_PRIORITY:
        if registry.get(p) is not None:
            return p
    return None


async def extract_questions_from_image(
    registry: ProviderRegistry,
    image_data: bytes,
    provider: ModelProvider | None = None,
) -> tuple[list[ExtractedQuestion], ModelResponse]:
    """Extract questions from a single exam paper image.

    Auto-selects the best available vision provider (Claude → GPT-4o → Gemini …).
    Falls back through the priority list if the preferred provider is not configured
    or if its API call fails.
    Returns (questions, raw_response).
    """
    # Determine provider order to try
    if provider is not None:
        candidates = [provider] + [p for p in _VISION_PROVIDER_PRIORITY if p != provider]
    else:
        candidates = list(_VISION_PROVIDER_PRIORITY)

    last_response: ModelResponse | None = None
    tried: list[str] = []

    for p in candidates:
        inst = registry.get(p)
        if inst is None:
            continue  # not configured

        logger.info("Attempting question extraction with provider: %s", p.value)
        response = await inst.generate_with_image(
            prompt=EXTRACTION_PROMPT,
            image_data=image_data,
            system=EXTRACTION_SYSTEM,
            temperature=0.1,
            max_tokens=8192,
        )
        last_response = response
        tried.append(p.value)

        if response.success:
            questions = _parse_extraction_response(response.content)
            if questions:
                logger.info("Extracted %d question(s) using %s", len(questions), p.value)
                return questions, response
            else:
                logger.warning(
                    "Provider %s returned a response but no questions were parsed; "
                    "trying next provider.", p.value
                )
        else:
            logger.warning(
                "Provider %s vision call failed (%s); trying next provider.",
                p.value, response.error,
            )

    # All providers failed or returned no questions
    if last_response is None:
        # No providers configured at all
        from shark_answer.providers.base import ModelResponse as MR
        last_response = MR(
            content="",
            provider="none",
            model_name="",
            success=False,
            error=(
                "No vision-capable AI providers are configured. "
                "Please add at least one API key to your .env file "
                "(ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_GEMINI_API_KEY)."
            ),
        )

    logger.error(
        "Question extraction failed after trying providers: %s. Last error: %s",
        tried, last_response.error,
    )
    return [], last_response


async def extract_questions_from_images(
    registry: ProviderRegistry,
    images: list[bytes],
    provider: ModelProvider | None = None,
) -> tuple[list[ExtractedQuestion], list[ModelResponse]]:
    """Extract questions from multiple exam paper images.

    Processes images sequentially to maintain question ordering.
    """
    all_questions: list[ExtractedQuestion] = []
    all_responses: list[ModelResponse] = []

    for i, img in enumerate(images):
        logger.info("Extracting questions from image %d/%d", i + 1, len(images))
        questions, resp = await extract_questions_from_image(registry, img, provider)
        all_questions.extend(questions)
        all_responses.append(resp)

    # De-duplicate by question number
    seen: set[str] = set()
    unique: list[ExtractedQuestion] = []
    for q in all_questions:
        if q.number not in seen:
            seen.add(q.number)
            unique.append(q)

    return unique, all_responses


def _parse_extraction_response(content: str) -> list[ExtractedQuestion]:
    """Parse the JSON response from vision model into ExtractedQuestion objects."""
    import json

    # Try to extract JSON from the response (might be wrapped in markdown)
    text = content.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        logger.error("No JSON array found in extraction response")
        return []

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s", e)
        return []

    questions: list[ExtractedQuestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Use `or` fallbacks to guard against None values returned by some models
        raw_marks = item.get("marks")
        raw_topic = item.get("topic_hints")
        questions.append(ExtractedQuestion(
            number=str(item.get("number") or ""),
            text=str(item.get("text") or ""),
            marks=int(raw_marks) if raw_marks is not None else 0,
            has_diagram=bool(item.get("has_diagram") or False),
            diagram_description=str(item.get("diagram_description") or ""),
            question_type=str(item.get("question_type") or "calculation"),
            topic_hints=list(raw_topic) if isinstance(raw_topic, list) else [],
        ))

    return questions
