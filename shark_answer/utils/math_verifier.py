"""Mathematical verification using sympy and numerical computation.

Used by Pipeline A (Science & Math) to auto-verify calculation answers.
"""

from __future__ import annotations

import logging
import re
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a mathematical verification."""
    verified: bool
    expected: Optional[str] = None
    computed: Optional[str] = None
    method: str = ""
    error: Optional[str] = None


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from model output.

    Looks for patterns like:
    - "= 42.5"
    - "answer is 42.5"
    - "The result: 42.5 m/s"
    - "≈ 3.14"
    """
    patterns = [
        r"(?:answer|result|value)\s*(?:is|=|:)\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:[a-zA-Z/²³\s]*)?$",
        r"≈\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        r"\\boxed\{([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\}",
    ]
    # Search from the end of text (final answer is usually last)
    lines = text.strip().split("\n")
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    return None


def verify_numeric_agreement(answers: list[str],
                             tolerance: float = 0.01) -> tuple[bool, list[float]]:
    """Check if multiple model answers agree numerically.

    Returns (all_agree, extracted_values).
    """
    values: list[float] = []
    for ans in answers:
        val = extract_numeric_answer(ans)
        if val is not None:
            values.append(val)

    if len(values) < 2:
        return False, values

    ref = values[0]
    if ref == 0:
        all_agree = all(abs(v) < tolerance for v in values)
    else:
        all_agree = all(abs(v - ref) / abs(ref) < tolerance for v in values)

    return all_agree, values


def verify_with_sympy(expression: str, expected_value: str) -> VerificationResult:
    """Verify a symbolic expression using sympy.

    Args:
        expression: A math expression string (e.g., "integrate(x**2, (x, 0, 1))")
        expected_value: The expected result as a string
    """
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        # Try to evaluate the expression
        result = sympy.sympify(expression)
        if hasattr(result, "evalf"):
            computed = float(result.evalf())
        else:
            computed = float(result)

        # Parse expected value
        expected = float(sympy.sympify(expected_value).evalf())

        # Compare
        if expected == 0:
            verified = abs(computed) < 1e-10
        else:
            verified = abs(computed - expected) / abs(expected) < 0.001

        return VerificationResult(
            verified=verified,
            expected=str(expected),
            computed=str(computed),
            method="sympy",
        )

    except Exception as e:
        return VerificationResult(
            verified=False,
            error=f"Sympy verification failed: {e}",
            method="sympy",
        )


def verify_physics_calculation(
    formula: str,
    variables: dict[str, float],
    expected_result: float,
    tolerance: float = 0.01,
) -> VerificationResult:
    """Verify a physics calculation by substituting values into a formula.

    Args:
        formula: Python-evaluable formula string (e.g., "0.5 * m * v**2")
        variables: Dict of variable names to values
        expected_result: Expected numerical result
        tolerance: Relative tolerance for comparison
    """
    try:
        # Build a safe namespace with math functions
        safe_ns = {
            "abs": abs, "round": round,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pi": math.pi, "e": math.e,
            **variables,
        }
        computed = float(eval(formula, {"__builtins__": {}}, safe_ns))

        if expected_result == 0:
            verified = abs(computed) < tolerance
        else:
            verified = abs(computed - expected_result) / abs(expected_result) < tolerance

        return VerificationResult(
            verified=verified,
            expected=str(expected_result),
            computed=str(computed),
            method="numerical_substitution",
        )

    except Exception as e:
        return VerificationResult(
            verified=False,
            error=f"Physics verification failed: {e}",
            method="numerical_substitution",
        )
