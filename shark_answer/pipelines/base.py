"""Base types and interfaces for pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shark_answer.utils.image_extractor import ExtractedQuestion


@dataclass
class AnswerVersion:
    """A single answer version with explanation."""
    version_number: int
    answer_text: str
    explanation_text: str = ""
    approach_label: str = ""       # e.g., "Energy conservation method"
    provider: str = ""             # which model generated it
    verified: bool = False         # True if computationally verified (Pipeline A)
    quality_score: Optional[str] = None    # e.g. "14/15", set by scoring loop
    language: str = "en"


@dataclass
class PipelineResult:
    """Result from processing a single question through a pipeline."""
    question: ExtractedQuestion
    versions: list[AnswerVersion] = field(default_factory=list)
    pipeline: str = ""
    subject: str = ""
    verification_notes: str = ""   # notes from math/code verification
    disagreement_resolved: bool = True
    cost_entries: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
