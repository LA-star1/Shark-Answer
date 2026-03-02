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


async def extract_questions_from_image(
    registry: ProviderRegistry,
    image_data: bytes,
    provider: ModelProvider = ModelProvider.CLAUDE,
) -> tuple[list[ExtractedQuestion], ModelResponse]:
    """Extract questions from a single exam paper image.

    Uses Claude by default (strongest vision for academic content).
    Returns (questions, raw_response).
    """
    inst = registry.get(provider)
    if inst is None:
        raise ValueError(f"Provider {provider.value} not configured")

    response = await inst.generate_with_image(
        prompt=EXTRACTION_PROMPT,
        image_data=image_data,
        system=EXTRACTION_SYSTEM,
        temperature=0.1,
        max_tokens=8192,
    )

    if not response.success:
        logger.error("Image extraction failed: %s", response.error)
        return [], response

    questions = _parse_extraction_response(response.content)
    return questions, response


async def extract_questions_from_images(
    registry: ProviderRegistry,
    images: list[bytes],
    provider: ModelProvider = ModelProvider.CLAUDE,
) -> tuple[list[ExtractedQuestion], list[ModelResponse]]:
    """Extract questions from multiple exam paper images.

    Processes images sequentially to maintain question ordering.
    """
    all_questions: list[ExtractedQuestion] = []
    all_responses: list[ModelResponse] = []

    for i, img in enumerate(images):
        logger.info("Extracting questions from image %d/%d", i + 1, len(images))
        questions, resp = await extract_questions_from_image(
            registry, img, provider
        )
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
        questions.append(ExtractedQuestion(
            number=str(item.get("number", "")),
            text=str(item.get("text", "")),
            marks=int(item.get("marks", 0)),
            has_diagram=bool(item.get("has_diagram", False)),
            diagram_description=str(item.get("diagram_description", "")),
            question_type=str(item.get("question_type", "calculation")),
            topic_hints=list(item.get("topic_hints", [])),
        ))

    return questions
