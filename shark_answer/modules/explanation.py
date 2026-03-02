"""Answer Explanation Module.

Generates detailed explanations/walkthroughs alongside each answer.
- Science/Math: step-by-step reasoning, method choice, common mistakes
- Essays: argument structure outline, marking criteria targeted, example rationale
- CS: logic explanation per code block, complexity notes
"""

from __future__ import annotations

from shark_answer.config import Pipeline


def build_explanation_prompt(pipeline: Pipeline, question_text: str,
                             answer_text: str, language: str = "en") -> str:
    """Build a prompt to generate an explanation for a given answer.

    Args:
        pipeline: Which pipeline (determines explanation style)
        question_text: The original question
        answer_text: The answer to explain
        language: "en" or "zh" for bilingual support
    """
    lang_instruction = ""
    if language == "zh":
        lang_instruction = "\n\nProvide the explanation in Chinese (简体中文)."
    elif language == "en":
        lang_instruction = "\n\nProvide the explanation in English."

    if pipeline == Pipeline.SCIENCE_MATH:
        return f"""You are a CIE A-Level science/math tutor. Generate a detailed
explanation for this answer that a student could use to understand and defend it verbally.

QUESTION:
{question_text}

ANSWER:
{answer_text}

Your explanation MUST include:
1. **Method Selection**: Why this approach was chosen (e.g., "We use energy conservation
   rather than Newton's laws because...")
2. **Step-by-Step Reasoning**: Walk through each calculation step, explaining the physics/
   math behind each transformation
3. **Key Formulas**: List all formulas used and WHY they apply here
4. **Units Analysis**: Show how units work out at each step
5. **Common Mistakes**: List 2-3 mistakes students commonly make on this type of question
   and how to avoid them
6. **Quick Verification**: A sanity check (order of magnitude, dimensional analysis,
   limiting cases)

Format: Use clear headings and numbered steps.{lang_instruction}"""

    elif pipeline == Pipeline.ESSAY:
        return f"""You are a CIE A-Level essay writing coach. Generate a detailed
explanation for this essay answer that helps the student understand its structure and
defend it verbally.

QUESTION:
{question_text}

ANSWER:
{answer_text}

Your explanation MUST include:
1. **Argument Structure**: Outline the logical flow (thesis → evidence → counter → evaluation)
2. **Mark Scheme Targeting**: Which CIE criteria (Knowledge, Application, Analysis,
   Evaluation) each paragraph targets and how
3. **Example Rationale**: Why each specific example/case study was chosen and what it demonstrates
4. **Evaluation Depth**: How the answer achieves high-level evaluation (not just
   "on the other hand" but genuine weighing of evidence)
5. **Key Terminology**: List subject-specific terms used and their significance
6. **Potential Examiner Follow-ups**: Questions an examiner might ask and how to respond

Format: Use clear headings.{lang_instruction}"""

    elif pipeline == Pipeline.CS:
        return f"""You are a CIE A-Level Computer Science tutor. Generate a detailed
explanation for this answer, focusing on logic and algorithmic understanding.

QUESTION:
{question_text}

ANSWER:
{answer_text}

Your explanation MUST include:
1. **Algorithm Logic**: Plain-English walkthrough of what the code/pseudocode does
2. **Line-by-Line Breakdown**: Explain key lines and why they work
3. **Data Structure Choice**: Why these data structures were used
4. **Time & Space Complexity**: Big-O analysis with justification
5. **Edge Cases**: What edge cases are handled and any that might be missed
6. **Alternative Approaches**: Brief mention of other valid algorithms and trade-offs
7. **CIE Pseudocode Notes**: How the answer conforms to Cambridge pseudocode conventions

Format: Use clear headings and code annotations.{lang_instruction}"""

    else:
        return f"""Generate a detailed explanation for this answer.

QUESTION:
{question_text}

ANSWER:
{answer_text}

Include step-by-step reasoning, key concepts, and common pitfalls.{lang_instruction}"""
