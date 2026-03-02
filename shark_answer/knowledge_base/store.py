"""Mark scheme and examiner report knowledge base.

Stores and retrieves marking criteria per subject and topic.
Simple file-based store — can be swapped for a vector DB later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MarkSchemeCriteria:
    """Marking criteria for a specific subject/topic."""
    subject: str
    paper: str                    # e.g., "9702/42" or "9708/31"
    year: str                     # e.g., "2024"
    session: str                  # e.g., "May/June" or "Oct/Nov"
    topic: str                    # e.g., "Kinematics", "Market Failure"
    question_number: str          # e.g., "1(a)(i)"
    marks: int = 0
    marking_points: list[str] = field(default_factory=list)
    common_errors: list[str] = field(default_factory=list)
    examiner_notes: str = ""
    grade_boundaries: dict[str, int] = field(default_factory=dict)  # {"A*": 85, "A": 75, ...}


@dataclass
class ExaminerReport:
    """Examiner report data for a paper."""
    subject: str
    paper: str
    year: str
    session: str
    general_comments: str = ""
    question_comments: dict[str, str] = field(default_factory=dict)  # q_number -> comment
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


class KnowledgeBase:
    """File-based knowledge base for mark schemes and examiner reports."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.mark_schemes_dir = base_dir / "mark_schemes"
        self.examiner_reports_dir = base_dir / "examiner_reports"
        self.mark_schemes_dir.mkdir(parents=True, exist_ok=True)
        self.examiner_reports_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index
        self._mark_schemes: list[MarkSchemeCriteria] = []
        self._examiner_reports: list[ExaminerReport] = []
        self._loaded = False

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._load_mark_schemes()
        self._load_examiner_reports()
        self._loaded = True

    def _load_mark_schemes(self) -> None:
        index_file = self.mark_schemes_dir / "index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            self._mark_schemes = [
                MarkSchemeCriteria(**item) for item in data
            ]

    def _load_examiner_reports(self) -> None:
        index_file = self.examiner_reports_dir / "index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            self._examiner_reports = [
                ExaminerReport(**item) for item in data
            ]

    def _save_mark_schemes(self) -> None:
        index_file = self.mark_schemes_dir / "index.json"
        data = [asdict(ms) for ms in self._mark_schemes]
        index_file.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def _save_examiner_reports(self) -> None:
        index_file = self.examiner_reports_dir / "index.json"
        data = [asdict(er) for er in self._examiner_reports]
        index_file.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def add_mark_scheme(self, criteria: MarkSchemeCriteria) -> None:
        """Add or update a mark scheme entry."""
        self._load_if_needed()
        # Replace if exists for same subject/paper/year/question
        self._mark_schemes = [
            ms for ms in self._mark_schemes
            if not (ms.subject == criteria.subject and ms.paper == criteria.paper
                    and ms.year == criteria.year
                    and ms.question_number == criteria.question_number)
        ]
        self._mark_schemes.append(criteria)
        self._save_mark_schemes()

    def add_examiner_report(self, report: ExaminerReport) -> None:
        """Add or update an examiner report."""
        self._load_if_needed()
        self._examiner_reports = [
            er for er in self._examiner_reports
            if not (er.subject == report.subject and er.paper == report.paper
                    and er.year == report.year)
        ]
        self._examiner_reports.append(report)
        self._save_examiner_reports()

    def get_mark_scheme(
        self,
        subject: str,
        topic: Optional[str] = None,
        question_number: Optional[str] = None,
    ) -> list[MarkSchemeCriteria]:
        """Retrieve relevant mark scheme criteria."""
        self._load_if_needed()
        results = [ms for ms in self._mark_schemes if ms.subject == subject]
        if topic:
            topic_lower = topic.lower()
            results = [ms for ms in results if topic_lower in ms.topic.lower()]
        if question_number:
            results = [ms for ms in results if ms.question_number == question_number]
        return results

    def get_examiner_reports(
        self,
        subject: str,
        paper: Optional[str] = None,
    ) -> list[ExaminerReport]:
        """Retrieve relevant examiner reports."""
        self._load_if_needed()
        results = [er for er in self._examiner_reports if er.subject == subject]
        if paper:
            results = [er for er in results if er.paper == paper]
        return results

    def get_marking_context(self, subject: str, topic: str = "") -> str:
        """Build a context string with relevant marking criteria for prompts."""
        schemes = self.get_mark_scheme(subject, topic)
        reports = self.get_examiner_reports(subject)

        parts: list[str] = []
        if schemes:
            parts.append("=== RELEVANT MARK SCHEME CRITERIA ===")
            for ms in schemes[-5:]:  # last 5 most recent
                parts.append(f"\nQ{ms.question_number} ({ms.year} {ms.session}):")
                parts.append(f"  Marks: {ms.marks}")
                for mp in ms.marking_points:
                    parts.append(f"  - {mp}")
                if ms.common_errors:
                    parts.append("  Common errors:")
                    for ce in ms.common_errors:
                        parts.append(f"    - {ce}")

        if reports:
            parts.append("\n=== EXAMINER REPORT INSIGHTS ===")
            for er in reports[-3:]:
                if er.general_comments:
                    parts.append(f"\n{er.year} {er.session}:")
                    parts.append(f"  {er.general_comments[:500]}")
                if er.weaknesses:
                    parts.append("  Common weaknesses:")
                    for w in er.weaknesses[:5]:
                        parts.append(f"    - {w}")

        return "\n".join(parts)
