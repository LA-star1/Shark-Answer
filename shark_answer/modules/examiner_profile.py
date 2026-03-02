"""Examiner Preference Module.

CIE examiners in different regions have different marking tendencies.
This module manages configurable examiner profiles that tailor answer
tone, depth, and style.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExaminerProfile:
    """An examiner marking tendency profile."""
    name: str                          # e.g., "Strict Evaluator", "Application-Focused"
    subject: str                       # e.g., "economics", "physics"
    region: str = "default"            # e.g., "UK", "SEA", "China"
    description: str = ""

    # Marking tendency weights (0.0 to 1.0)
    evaluation_depth: float = 0.5      # How much depth in evaluation/analysis
    real_world_examples: float = 0.5   # Preference for real-world examples
    diagram_preference: float = 0.5    # How much diagrams are valued
    formula_rigour: float = 0.5        # How strict on formula presentation
    structure_strictness: float = 0.5  # How strict on answer structure
    conciseness: float = 0.5           # Prefers concise vs detailed

    # Style preferences
    preferred_tone: str = "formal"     # formal, semi-formal, academic
    penalizes_formulaic: bool = False  # True if penalizes template-like answers
    values_originality: bool = False   # True if rewards original arguments
    strict_on_units: bool = True       # True if penalizes missing/wrong units

    # Custom instructions (appended to generation prompt)
    custom_instructions: str = ""

    def to_prompt_guidance(self) -> str:
        """Convert profile to prompt guidance text for answer generation."""
        parts: list[str] = [
            f"=== EXAMINER PROFILE: {self.name} ({self.region}) ===",
        ]

        if self.evaluation_depth > 0.7:
            parts.append("- This examiner values DEEP evaluation. Provide thorough "
                         "analysis with multiple perspectives and counter-arguments.")
        elif self.evaluation_depth < 0.3:
            parts.append("- This examiner prefers concise evaluation. Be focused, "
                         "avoid over-elaboration.")

        if self.real_world_examples > 0.7:
            parts.append("- This examiner STRONGLY prefers real-world, current examples. "
                         "Use specific case studies, data, and recent events.")
        elif self.real_world_examples < 0.3:
            parts.append("- This examiner values theoretical rigour over real-world "
                         "examples. Focus on economic/scientific principles.")

        if self.diagram_preference > 0.7:
            parts.append("- Include diagrams where possible. Clearly label axes, "
                         "curves, and equilibrium points.")

        if self.formula_rigour > 0.7:
            parts.append("- Show ALL formula derivation steps. State formulas before "
                         "substitution. Include units at every step.")

        if self.penalizes_formulaic:
            parts.append("- AVOID formulaic/template answer structures. This examiner "
                         "penalizes mechanical answers. Show genuine understanding.")

        if self.values_originality:
            parts.append("- This examiner rewards original thinking and novel "
                         "examples. Avoid overused textbook examples.")

        if self.strict_on_units:
            parts.append("- Always include correct SI units. Missing units will "
                         "lose marks.")

        parts.append(f"- Preferred tone: {self.preferred_tone}")

        if self.custom_instructions:
            parts.append(f"- Additional: {self.custom_instructions}")

        return "\n".join(parts)


# Default profiles
DEFAULT_PROFILES: list[ExaminerProfile] = [
    ExaminerProfile(
        name="Standard CIE",
        subject="general",
        region="default",
        description="Balanced CIE marking standard",
        evaluation_depth=0.5,
        real_world_examples=0.5,
        diagram_preference=0.5,
    ),
    ExaminerProfile(
        name="Strict Evaluator",
        subject="economics",
        region="UK",
        description="UK-based examiner who demands deep evaluation",
        evaluation_depth=0.9,
        real_world_examples=0.8,
        penalizes_formulaic=True,
        values_originality=True,
    ),
    ExaminerProfile(
        name="Diagram-Heavy Physics",
        subject="physics",
        region="SEA",
        description="South-East Asian examiner who values diagrams and method marks",
        diagram_preference=0.9,
        formula_rigour=0.8,
        strict_on_units=True,
    ),
    ExaminerProfile(
        name="Application-Focused",
        subject="economics",
        region="SEA",
        description="Values application of theory to real data",
        evaluation_depth=0.6,
        real_world_examples=0.9,
        values_originality=False,
    ),
]


class ExaminerProfileManager:
    """Manages examiner profiles with file persistence."""

    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, ExaminerProfile] = {}
        self._load()

    def _load(self) -> None:
        """Load profiles from disk, seeding defaults if needed."""
        profile_file = self.profile_dir / "profiles.json"
        if profile_file.exists():
            data = json.loads(profile_file.read_text(encoding="utf-8"))
            for item in data:
                p = ExaminerProfile(**item)
                self._profiles[p.name] = p
        else:
            # Seed with defaults
            for p in DEFAULT_PROFILES:
                self._profiles[p.name] = p
            self._save()

    def _save(self) -> None:
        profile_file = self.profile_dir / "profiles.json"
        data = [asdict(p) for p in self._profiles.values()]
        profile_file.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    def get_profile(self, name: str) -> Optional[ExaminerProfile]:
        return self._profiles.get(name)

    def get_profile_for_subject(self, subject: str,
                                region: str = "default") -> ExaminerProfile:
        """Get best matching profile for a subject and region."""
        # Exact match
        for p in self._profiles.values():
            if p.subject == subject and p.region == region:
                return p
        # Subject match with default region
        for p in self._profiles.values():
            if p.subject == subject and p.region == "default":
                return p
        # Fallback to general default
        return self._profiles.get("Standard CIE", DEFAULT_PROFILES[0])

    def list_profiles(self) -> list[ExaminerProfile]:
        return list(self._profiles.values())

    def add_profile(self, profile: ExaminerProfile) -> None:
        self._profiles[profile.name] = profile
        self._save()

    def update_profile(self, name: str, **kwargs) -> Optional[ExaminerProfile]:
        p = self._profiles.get(name)
        if not p:
            return None
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        self._save()
        return p

    def delete_profile(self, name: str) -> bool:
        if name in self._profiles:
            del self._profiles[name]
            self._save()
            return True
        return False
