"""Pipeline D: Physics Practical Prediction (skeleton).

Input: Confidential instructions document (equipment list, photo or PDF)
Process: Cross-reference equipment with past papers
Output: Top 3 most likely experiments + outline of expected report structure

This is a lower-priority skeleton — core logic stubs with TODOs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider, Pipeline
from shark_answer.providers.registry import ProviderRegistry
from shark_answer.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# Historical experiment database (simplified — expand with real data)
PAST_EXPERIMENTS = [
    {
        "year": "2023", "session": "May/June",
        "topic": "Simple Harmonic Motion",
        "equipment": ["pendulum", "stopwatch", "ruler", "clamp stand", "protractor"],
        "description": "Investigate the relationship between length and period of a pendulum",
    },
    {
        "year": "2023", "session": "Oct/Nov",
        "topic": "Electrical Resistance",
        "equipment": ["ammeter", "voltmeter", "power supply", "resistance wire", "ruler", "micrometer"],
        "description": "Investigate how resistance varies with length/diameter of a wire",
    },
    {
        "year": "2022", "session": "May/June",
        "topic": "Refraction of Light",
        "equipment": ["ray box", "glass block", "protractor", "ruler", "pins"],
        "description": "Verify Snell's law by measuring angles of incidence and refraction",
    },
    {
        "year": "2022", "session": "Oct/Nov",
        "topic": "Springs and Hooke's Law",
        "equipment": ["spring", "masses", "ruler", "clamp stand", "pointer"],
        "description": "Investigate the extension of a spring with applied force",
    },
    {
        "year": "2021", "session": "May/June",
        "topic": "Thermal Energy Transfer",
        "equipment": ["thermometer", "beaker", "stopwatch", "heater", "insulation"],
        "description": "Investigate cooling curves or specific heat capacity",
    },
]


@dataclass
class PracticalPrediction:
    """A predicted practical experiment."""
    rank: int
    topic: str
    confidence: float  # 0-1
    equipment_match: list[str]  # equipment items that matched
    description: str
    report_outline: str = ""
    past_reference: str = ""  # reference to past paper


@dataclass
class PracticalResult:
    """Result from practical prediction pipeline."""
    predictions: list[PracticalPrediction] = field(default_factory=list)
    equipment_detected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


EQUIPMENT_EXTRACTION_PROMPT = """Extract ALL equipment items from this confidential
practical exam instructions document.

Return a JSON array of equipment items, e.g.:
["ruler", "stopwatch", "pendulum bob", "clamp stand", "protractor", "string"]

Be thorough — include every piece of equipment mentioned, even common items like rulers."""


async def run_pipeline_d(
    image_data: Optional[bytes],
    text_content: Optional[str],
    registry: ProviderRegistry,
    config: AppConfig,
    cost_tracker: CostTracker,
) -> PracticalResult:
    """Run Pipeline D for physics practical prediction.

    Args:
        image_data: Photo of confidential instructions (optional)
        text_content: Text content of instructions (optional, e.g. from PDF extraction)
    """
    result = PracticalResult()

    primary = config.get_pipeline_models(Pipeline.PRACTICAL, "primary")
    if not primary:
        result.errors.append("No models configured for Pipeline D")
        return result

    model = primary[0]

    # Step 1: Extract equipment list
    if image_data:
        responses = await registry.call_models_with_image_parallel(
            providers=[model],
            prompt=EQUIPMENT_EXTRACTION_PROMPT,
            image_data=image_data,
            temperature=0.1, max_tokens=1000,
        )
        cost_tracker.record_batch(responses, "physics", "D")
        if responses[0].success:
            result.equipment_detected = _parse_equipment(responses[0].content)
    elif text_content:
        responses = await registry.call_models_parallel(
            providers=[model],
            prompt=f"{EQUIPMENT_EXTRACTION_PROMPT}\n\nDocument text:\n{text_content}",
            system="",
            temperature=0.1, max_tokens=1000,
        )
        cost_tracker.record_batch(responses, "physics", "D")
        if responses[0].success:
            result.equipment_detected = _parse_equipment(responses[0].content)

    if not result.equipment_detected:
        result.errors.append("Could not extract equipment list")
        return result

    # Step 2: Cross-reference with past experiments
    equipment_set = {e.lower() for e in result.equipment_detected}
    scored: list[tuple[dict, float, list[str]]] = []

    for exp in PAST_EXPERIMENTS:
        exp_equip = {e.lower() for e in exp["equipment"]}
        overlap = equipment_set & exp_equip
        if overlap:
            score = len(overlap) / len(exp_equip)
            scored.append((exp, score, list(overlap)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Build predictions
    for rank, (exp, score, matched) in enumerate(scored[:3], 1):
        # TODO: Use AI to generate detailed report outline based on matched experiment
        prediction = PracticalPrediction(
            rank=rank,
            topic=exp["topic"],
            confidence=score,
            equipment_match=matched,
            description=exp["description"],
            past_reference=f"{exp['year']} {exp['session']}",
            report_outline=_generate_report_skeleton(exp["topic"]),
        )
        result.predictions.append(prediction)

    return result


def _parse_equipment(content: str) -> list[str]:
    """Parse equipment list from model output."""
    import json
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _generate_report_skeleton(topic: str) -> str:
    """Generate a basic practical report outline. TODO: AI-generated version."""
    return f"""=== Practical Report: {topic} ===

1. AIM
   - State the aim of the experiment

2. VARIABLES
   - Independent variable: [to be determined from equipment]
   - Dependent variable: [to be measured]
   - Control variables: [list at least 3]

3. APPARATUS
   - [Listed from extracted equipment]
   - Include diagram of setup

4. METHOD
   - Step-by-step procedure (at least 8 steps)
   - Include safety precautions
   - Describe how to minimize random errors

5. RESULTS TABLE
   - Appropriate column headings with units
   - At least 6 sets of readings
   - Include repeat readings

6. GRAPH
   - Appropriate axes with labels and units
   - Line of best fit
   - Gradient calculation (if needed)

7. ANALYSIS
   - Process data using appropriate equations
   - Calculate percentage uncertainties
   - Compare with theoretical/expected values

8. EVALUATION
   - Sources of error (systematic and random)
   - Improvements to the method
   - Reliability of results
"""
