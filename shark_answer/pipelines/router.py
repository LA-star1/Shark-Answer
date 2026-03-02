"""Route questions to the appropriate pipeline."""

from __future__ import annotations

from shark_answer.config import Pipeline, Subject, SUBJECT_PIPELINE_MAP
from shark_answer.utils.image_extractor import ExtractedQuestion


def classify_question_type(question: ExtractedQuestion, subject: Subject) -> Pipeline:
    """Determine the correct pipeline for a question.

    Biology is special: essay-type bio questions go to Pipeline B,
    while calculation/diagram bio questions go to Pipeline A.
    """
    if subject == Subject.BIOLOGY and question.question_type == "essay":
        return Pipeline.ESSAY

    return SUBJECT_PIPELINE_MAP.get(subject, Pipeline.SCIENCE_MATH)


def route_question(question: ExtractedQuestion, subject: Subject) -> Pipeline:
    """Route a question to its pipeline."""
    return classify_question_type(question, subject)
