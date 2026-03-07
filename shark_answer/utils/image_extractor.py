"""Extract exam questions from photos using AI vision or pre-extracted text."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from shark_answer.config import ModelProvider
from shark_answer.providers.base import ModelResponse
from shark_answer.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Directory for saving raw vision responses for debugging
_DEBUG_DIR = Path("/tmp/shark_answer_debug")

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

CRITICAL — CONTEXT SELF-CONTAINMENT:
Every sub-question's "text" field MUST be fully self-contained.  That means:
• If a parent question (e.g. Q4) defines a function, variable, scenario, or gives data
  (e.g. "f(x) = 2x + 3 for x > 0"), that definition MUST appear verbatim at the start of
  EVERY child sub-question (Q4(a), Q4(b), …) even if it is not reprinted on the page.
• If a sub-question says "hence", "using your answer", or "from part (a)", include the
  full parent stem so the solver can understand what is being referenced.
• If a scenario/passage introduces data (e.g. a table, a paragraph of context), copy it
  into every sub-question that depends on it.
• Never output a bare sub-question like "(a) Find f⁻¹(x) [3]" without first showing
  whatever the parent question established (the function, the scenario, the data).

Output format — return a JSON array:
[
  {
    "number": "1(a)(i)",
    "parent_context": "Full text of the parent question stem / scenario / data (empty string if the top-level question has no preamble)",
    "text": "Sub-question text only (without repeating parent_context), with $LaTeX$ formulas",
    "marks": 3,
    "has_diagram": true,
    "diagram_description": "A circuit diagram showing...",
    "question_type": "calculation|explanation|essay|diagram|code|proof",
    "topic_hints": ["mechanics", "kinematics"]
  }
]

The parent_context field MUST contain everything the solver needs: function definitions, given data,
table contents, scenario paragraphs — anything stated in the parent Q that the sub-Q relies on.
If the sub-question IS the top-level question (no parent), set parent_context to "".
"""

EXTRACTION_PROMPT = """Extract ALL questions from this exam paper image.
Return valid JSON only — a JSON array (even if empty: []).
If there are no questions on this page (e.g. cover page, blank page, formula sheet),
return an empty array: []
Be extremely precise with formulas, numbers, and units.
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


# ── Whole-paper extraction: prompts & provider chain ──────────────────────────

WHOLE_PAPER_EXTRACTION_SYSTEM = """You are an expert at reading CIE A-Level exam papers.
Extract ALL questions from this COMPLETE exam paper (all pages are provided together).

Rules:
1. Preserve ALL mathematical formulas in LaTeX notation (e.g., $F = ma$, $\\int_0^1 x^2 dx$)
2. Describe any diagrams, circuit diagrams, or graphs in detail within [DIAGRAM: description]
3. Preserve data tables as markdown tables
4. Number each question and sub-question exactly as shown (e.g., 1(a)(i))
5. Include mark allocations in square brackets, e.g. [3 marks]
6. If a question references a diagram/figure, describe what it shows
7. Preserve units and significant figures exactly as printed

CRITICAL — CONTEXT SELF-CONTAINMENT:
Every sub-question's "text" field MUST be fully self-contained. That means:
• If a parent question (e.g. Q4) defines a function, variable, scenario, or gives data
  (e.g. "f(x) = 2x + 3 for x > 0"), that definition MUST appear verbatim at the start of
  EVERY child sub-question (Q4(a), Q4(b), …) even if it is not reprinted on the page.
• If a sub-question says "hence", "using your answer", or "from part (a)", include the
  full parent stem so the solver can understand what is being referenced.
• If a scenario/passage introduces data (e.g. a table, a paragraph of context), copy it
  into every sub-question that depends on it.
• Never output a bare sub-question like "(a) Find f⁻¹(x) [3]" without first showing
  whatever the parent question established (the function, the scenario, the data).

IMPORTANT: All pages are provided simultaneously — you have complete context for EVERY
question. There are NO orphaned sub-questions or missing parent stems.

Output format — return a JSON array:
[
  {
    "number": "1(a)(i)",
    "parent_context": "Full text of the parent question stem / scenario / data (empty string if the top-level question has no preamble)",
    "text": "Sub-question text only (without repeating parent_context), with $LaTeX$ formulas",
    "marks": 3,
    "has_diagram": true,
    "diagram_description": "A circuit diagram showing...",
    "question_type": "calculation|explanation|essay|diagram|code|proof",
    "topic_hints": ["mechanics", "kinematics"]
  }
]

The parent_context field MUST contain everything the solver needs: function definitions, given data,
table contents, scenario paragraphs — anything stated in the parent Q that the sub-Q relies on.
If the sub-question IS the top-level question (no parent), set parent_context to "".
"""

WHOLE_PAPER_EXTRACTION_PROMPT = """Extract ALL questions from this complete CIE exam paper.
All pages are provided at once — you have the full paper with complete context.
Return valid JSON only — a JSON array of ALL questions across all pages.
Be extremely precise with formulas, numbers, and units.
If any part of the image is unclear, note it with [UNCLEAR: description]."""


async def _call_gemini_whole_paper(
    registry: ProviderRegistry,
    images: list[bytes],
    system: str,
    prompt: str,
    max_tokens: int = 16384,
) -> "ModelResponse":
    """Make a single multi-image call to Gemini using all page images."""
    from shark_answer.providers.gemini_provider import GeminiProvider
    from shark_answer.providers.base import ModelResponse, TokenUsage
    from google.genai import types

    inst = registry.get(ModelProvider.GEMINI)
    if inst is None or not isinstance(inst, GeminiProvider):
        return ModelResponse(content="", provider="gemini", model_name="",
                             success=False, error="Gemini not configured")

    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_tokens,
    )
    if system:
        config.system_instruction = system

    contents = [types.Part.from_bytes(data=img, mime_type="image/png") for img in images]
    contents.append(prompt)

    try:
        resp = await inst.client.aio.models.generate_content(
            model=inst.model_name,
            contents=contents,
            config=config,
        )
        text = resp.text or ""
        usage_meta = resp.usage_metadata
        return ModelResponse(
            content=text,
            provider="gemini",
            model_name=inst.model_name,
            usage=TokenUsage(
                input_tokens=usage_meta.prompt_token_count if usage_meta else 0,
                output_tokens=usage_meta.candidates_token_count if usage_meta else 0,
            ),
        )
    except Exception as exc:
        return ModelResponse(content="", provider="gemini", model_name=getattr(inst, "model_name", ""),
                             success=False, error=str(exc))


async def _call_claude_whole_paper(
    registry: ProviderRegistry,
    images: list[bytes],
    system: str,
    prompt: str,
    max_tokens: int = 16384,
) -> "ModelResponse":
    """Make a single multi-image call to Claude using all page images."""
    import base64
    from shark_answer.providers.claude_provider import ClaudeProvider
    from shark_answer.providers.base import ModelResponse, TokenUsage

    inst = registry.get(ModelProvider.CLAUDE)
    if inst is None or not isinstance(inst, ClaudeProvider):
        return ModelResponse(content="", provider="claude", model_name="",
                             success=False, error="Claude not configured")

    content: list[dict] = []
    for img in images:
        b64 = base64.standard_b64encode(img).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    content.append({"type": "text", "text": prompt})

    kwargs: dict = {
        "model": inst.model_name,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        kwargs["system"] = system

    try:
        resp = await inst.client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return ModelResponse(
            content=text,
            provider="claude",
            model_name=inst.model_name,
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens or 0,
                output_tokens=resp.usage.output_tokens or 0,
            ),
        )
    except Exception as exc:
        return ModelResponse(content="", provider="claude", model_name=getattr(inst, "model_name", ""),
                             success=False, error=str(exc))


async def _call_gpt4o_whole_paper(
    registry: ProviderRegistry,
    images: list[bytes],
    system: str,
    prompt: str,
    max_tokens: int = 16384,
) -> "ModelResponse":
    """Make a single multi-image call to GPT-4o using all page images."""
    import base64
    from shark_answer.providers.openai_provider import OpenAIProvider
    from shark_answer.providers.base import ModelResponse, TokenUsage

    inst = registry.get(ModelProvider.GPT4O)
    if inst is None or not isinstance(inst, OpenAIProvider):
        return ModelResponse(content="", provider="gpt4o", model_name="",
                             success=False, error="GPT-4o not configured")

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    img_content: list[dict] = []
    for img in images:
        b64 = base64.standard_b64encode(img).decode("utf-8")
        img_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })
    img_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": img_content})

    try:
        resp = await inst.client.chat.completions.create(
            model=inst.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            provider="gpt4o",
            model_name=inst.model_name,
            usage=TokenUsage(
                input_tokens=(usage.prompt_tokens or 0) if usage else 0,
                output_tokens=(usage.completion_tokens or 0) if usage else 0,
            ),
        )
    except Exception as exc:
        return ModelResponse(content="", provider="gpt4o", model_name=getattr(inst, "model_name", ""),
                             success=False, error=str(exc))


async def extract_questions_whole_paper(
    registry: ProviderRegistry,
    images: list[bytes],
    subject: str = "",
) -> tuple[list["ExtractedQuestion"], "ModelResponse | None"]:
    """Extract all questions from a whole exam paper in a single API call.

    Sends ALL page images to Gemini 2.5 Pro in one request, eliminating
    cross-page context loss and orphaned sub-question numbering bugs.

    Fallback chain: Gemini 2.5 Pro → Claude Opus 4.6 → GPT-4o
    Timeout: 120 seconds per call.
    If paper > 30 pages, splits into two halves automatically.

    Returns (questions, last_response).
    """
    from shark_answer.providers.base import ModelResponse

    if not images:
        return [], None

    # Split papers > 30 pages into two halves to stay within model limits
    if len(images) > 30:
        mid = len(images) // 2
        logger.info(
            "Paper has %d pages — splitting at %d for whole-paper extraction",
            len(images), mid,
        )
        q1, r1 = await extract_questions_whole_paper(registry, images[:mid], subject)
        q2, r2 = await extract_questions_whole_paper(registry, images[mid:], subject)
        combined = q1 + q2
        combined = _fix_orphaned_subnumbers(combined)
        seen: set[str] = set()
        unique: list[ExtractedQuestion] = []
        for q in combined:
            if q.number not in seen:
                seen.add(q.number)
                unique.append(q)
        return unique, r2 or r1

    # Build the ordered fallback chain
    callers = [
        ("Gemini 2.5 Pro", lambda imgs: _call_gemini_whole_paper(
            registry, imgs,
            WHOLE_PAPER_EXTRACTION_SYSTEM,
            WHOLE_PAPER_EXTRACTION_PROMPT,
        )),
        ("Claude Opus", lambda imgs: _call_claude_whole_paper(
            registry, imgs,
            WHOLE_PAPER_EXTRACTION_SYSTEM,
            WHOLE_PAPER_EXTRACTION_PROMPT,
        )),
        ("GPT-4o", lambda imgs: _call_gpt4o_whole_paper(
            registry, imgs,
            WHOLE_PAPER_EXTRACTION_SYSTEM,
            WHOLE_PAPER_EXTRACTION_PROMPT,
        )),
    ]

    last_response: ModelResponse | None = None

    for name, caller in callers:
        logger.info(
            "Whole-paper extraction attempt with %s (%d pages)\u2026", name, len(images)
        )
        try:
            response = await asyncio.wait_for(caller(images), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning("%s whole-paper extraction timed out (120s)", name)
            continue

        last_response = response

        if not response.success:
            logger.warning(
                "%s whole-paper extraction failed: %s", name, response.error
            )
            continue

        # Debug: save raw response
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = name.replace(" ", "_")
            debug_file = (
                _DEBUG_DIR
                / f"whole_paper_{safe_name}_{subject}_{len(images)}p.txt"
            )
            debug_file.write_text(response.content, encoding="utf-8")
            logger.debug("Whole-paper response saved to %s", debug_file)
        except Exception:
            pass

        questions, parse_ok = _parse_extraction_response(response.content)
        if parse_ok and questions:
            logger.info(
                "Whole-paper extraction: %d question(s) via %s",
                len(questions), name,
            )
            return questions, response
        elif parse_ok and not questions:
            logger.warning("%s returned valid JSON but no questions", name)
            continue
        else:
            logger.warning(
                "%s returned unparseable response, trying next\u2026", name
            )

    logger.error("Whole-paper extraction failed for all providers")
    return [], last_response


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
            # ── Debug: save raw response to file ──────────────────────────────
            try:
                _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                debug_file = _DEBUG_DIR / f"extraction_{p.value}_{id(image_data)}.txt"
                debug_file.write_text(response.content, encoding="utf-8")
                logger.debug("Raw vision response saved to %s", debug_file)
            except Exception:
                pass  # never let debug logging break extraction
            # ─────────────────────────────────────────────────────────────────

            questions, parse_ok = _parse_extraction_response(response.content)
            if parse_ok and questions:
                logger.info("Extracted %d question(s) using %s", len(questions), p.value)
                return questions, response
            elif parse_ok and not questions:
                # Valid JSON returned empty array → page has no questions (cover, blank, formula)
                logger.info("Page has no questions (empty array) — provider %s", p.value)
                return [], response
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

    # ── Fix orphaned sub-question numbering (e.g. "(c)" → "1(c)") ───────────
    # This happens when a page starts mid-question without the major Q number.
    all_questions = _fix_orphaned_subnumbers(all_questions)

    # De-duplicate by question number
    seen: set[str] = set()
    unique: list[ExtractedQuestion] = []
    for q in all_questions:
        if q.number not in seen:
            seen.add(q.number)
            unique.append(q)

    return unique, all_responses


def _fix_orphaned_subnumbers(questions: list[ExtractedQuestion]) -> list[ExtractedQuestion]:
    """Assign major question numbers to orphaned sub-questions.

    When vision extracts a page that starts mid-question (e.g. "(c)" without "1"),
    the sub-question gets numbered "(c)" instead of "1(c)".  This function detects
    such orphaned sub-questions and assigns them to the last major question seen,
    also propagating the parent question's stem context when available.
    """
    # Pattern: orphaned sub-question whose number is just a letter group like "(c)" or "(iv)"
    _ORPHAN_RE = re.compile(r'^\(([a-z])\)$|^\(([ivxlcdm]+)\)$')

    current_major = ""       # last seen top-level Q number (e.g. "1", "2")
    parent_stem   = ""       # context text of the last top-level question

    result: list[ExtractedQuestion] = []
    for q in questions:
        num = q.number.strip()
        if _ORPHAN_RE.match(num) and current_major:
            # Orphaned — re-parent it
            new_num = f"{current_major}{num}"
            logger.debug("Re-parenting orphaned %s → %s (major Q%s)", num, new_num, current_major)
            # Prepend parent stem context if not already present
            new_text = q.text
            if parent_stem and parent_stem not in new_text:
                new_text = f"{parent_stem}\n\n{new_text}"
            result.append(ExtractedQuestion(
                number=new_num,
                text=new_text,
                marks=q.marks,
                has_diagram=q.has_diagram,
                diagram_description=q.diagram_description,
                question_type=q.question_type,
                topic_hints=q.topic_hints,
            ))
        else:
            result.append(q)
            # Update tracking if this is a top-level Q (number starts with a digit)
            if re.match(r'^\d', num):
                major = num.split("(")[0].strip()
                if major.isdigit() and not num[len(major):]:
                    # Pure top-level question (e.g. "1", "2") — save its stem
                    current_major = major
                    parent_stem   = q.text
                elif major != current_major:
                    # New major question number encountered in a sub-question string
                    # e.g. "3(a)" — update current_major but don't overwrite parent_stem
                    # unless the major changed
                    current_major = major

    return result


def _parse_extraction_response(content: str) -> tuple[list[ExtractedQuestion], bool]:
    """Parse the JSON response from vision model into ExtractedQuestion objects.

    Returns (questions, parse_ok) where parse_ok=True means the model returned
    valid JSON (even if the list is empty — e.g. a blank page with no questions).
    parse_ok=False means the response could not be parsed as JSON at all.
    """
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
        return [], False

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s", e)
        return [], False

    questions: list[ExtractedQuestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Use `or` fallbacks to guard against None values returned by some models
        raw_marks = item.get("marks")
        raw_topic = item.get("topic_hints")

        # ── Context self-containment: merge parent_context into text ──────────
        # If the vision model populated the explicit parent_context field, prepend
        # it to the question text so solvers always see the full question stem.
        parent_ctx = str(item.get("parent_context") or "").strip()
        q_text     = str(item.get("text") or "").strip()
        if parent_ctx and parent_ctx not in q_text:
            q_text = f"{parent_ctx}\n\n{q_text}"

        questions.append(ExtractedQuestion(
            number=str(item.get("number") or ""),
            text=q_text,
            marks=int(raw_marks) if raw_marks is not None else 0,
            has_diagram=bool(item.get("has_diagram") or False),
            diagram_description=str(item.get("diagram_description") or ""),
            question_type=str(item.get("question_type") or "calculation"),
            topic_hints=list(raw_topic) if isinstance(raw_topic, list) else [],
        ))

    # ── Second pass: propagate context from sibling questions ─────────────────
    # If the model still produced some short sub-questions without parent context,
    # scan for siblings in the same major-Q group that have longer text and use
    # any shared preamble prefix as context for the short ones.
    questions = _propagate_sibling_context(questions)

    return questions, True


def _propagate_sibling_context(questions: list[ExtractedQuestion]) -> list[ExtractedQuestion]:
    """Ensure each sub-question inherits the parent stem from its siblings.

    Groups questions by major question number (e.g. "4" for "4(a)", "4(b)").
    If the group has a shared leading paragraph that appears in most siblings but
    not all, it is prepended to the ones that lack it.  This is a safety net for
    vision models that missed the context-injection instruction.
    """
    if not questions:
        return questions

    from collections import defaultdict

    # ── Group by major question number ──────────────────────────────────────
    groups: dict[str, list[ExtractedQuestion]] = defaultdict(list)
    for q in questions:
        # "4(a)(i)" → major = "4", "4(a)" → major = "4", "4" → major = "4"
        major = q.number.split("(")[0].strip()
        groups[major].append(q)

    for major, grp in groups.items():
        if len(grp) < 2:
            continue  # single question in group — nothing to propagate

        # Find the longest leading paragraph shared by ≥ half of the group.
        # We compare the first paragraph (up to the first double-newline or 500 chars).
        def _first_para(text: str) -> str:
            idx = text.find("\n\n")
            return (text[:idx] if idx != -1 else text[:500]).strip()

        paras = [_first_para(q.text) for q in grp]
        # Count how often each first-para appears
        from collections import Counter
        counts = Counter(paras)
        most_common_para, freq = counts.most_common(1)[0]

        # Only propagate if the common para appears in ≥ half the group,
        # is non-trivial (>20 chars), and looks like a context/stem (not a question itself)
        is_context = most_common_para and len(most_common_para) > 20 and freq >= max(2, len(grp) // 2)
        # Avoid propagating if it's clearly a question (starts with "Find", "Show", etc.)
        question_starters = ("find", "show", "prove", "calculate", "determine", "state",
                             "explain", "describe", "sketch", "draw", "write", "give")
        looks_like_question = most_common_para.lower().startswith(question_starters)

        if is_context and not looks_like_question:
            for q in grp:
                if most_common_para not in q.text:
                    q.text = f"{most_common_para}\n\n{q.text}"
                    logger.debug(
                        "Propagated parent context to %s from sibling in Q%s group",
                        q.number, major,
                    )

    return questions


# ── Text-based question extraction (bypass vision for .txt QP files) ──────────

def _infer_question_type(text: str) -> str:
    """Infer CIE question type from keywords in question text."""
    t = text.lower()
    if any(w in t for w in ("calculate", "determine", "find the", "work out", "compute",
                             "show that", "prove", "derive")):
        return "calculation"
    if any(w in t for w in ("assess", "evaluate", "discuss", "consider", "analyse",
                             "to what extent", "how far", "whether")):
        return "essay"
    if any(w in t for w in ("explain", "describe", "state", "define", "what is meant",
                             "suggest", "outline")):
        return "explanation"
    if any(w in t for w in ("draw", "sketch", "label", "complete the diagram",
                             "complete fig", "on fig")):
        return "diagram"
    return "short_answer"


def _clean_qp_text(text: str) -> str:
    """Strip page markers, blank-line noise, and answer-line artifacts from QP text."""
    text = re.sub(r'\[Page \d+\]', '', text)
    text = re.sub(r'\[Turn over\]?', '', text)
    text = re.sub(r'\[Total:\s*\d+\]', '', text)
    text = re.sub(r'BLANK PAGE', '', text, flags=re.IGNORECASE)
    # Remove dotted answer lines (.....) and underscores used for writing space
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'_{4,}', '', text)
    # Remove CIE page headers (e.g. "9702/21/M/J/24")
    text = re.sub(r'\b\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}\b', '', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def extract_questions_from_text(qp_text: str) -> list[ExtractedQuestion]:
    """Parse CIE A-Level exam questions from a pre-extracted QP .txt file.

    Handles the standard CIE question hierarchy::

        1  Introduction text / data / tables ...
            (a) Sub-question intro ...
                (i)  Leaf question text [marks]
                (ii) Leaf question text [marks]
            (b) Leaf question text [marks]
        2  ...

    Each leaf question is emitted as one ExtractedQuestion whose `text` field
    concatenates: major-intro + sub-intro + leaf-text so that all table/figure
    data referenced by the question is included.

    Returns an empty list if no questions with marks > 0 could be found.
    """
    text = _clean_qp_text(qp_text)

    # ── Locate start of actual questions (skip cover / data / formulae pages) ─
    # CIE papers always start Q1 with a line like "\n1 " or "\n1\n"
    q_start = re.search(r'\n\s*1[\s\n]', text)
    if not q_start:
        logger.warning("extract_questions_from_text: could not find start of Q1 in text")
        return []

    preamble = text[:q_start.start()]   # data booklet / formulae — kept as reference
    body = text[q_start.start():]

    # ── Split into major-question blocks by "N " pattern at line start ────────
    # e.g. "\n1 ", "\n2 ", "\n3 " (up to question 20 to be safe)
    major_split = re.split(r'\n\s*(\d{1,2})\s+', '\n' + body)
    # Produces: ['', '1', 'text-of-Q1', '2', 'text-of-Q2', ...]

    questions: list[ExtractedQuestion] = []

    i = 1
    while i < len(major_split) - 1:
        q_major = major_split[i].strip()
        if not q_major.isdigit():
            i += 1
            continue
        q_body = major_split[i + 1] if i + 1 < len(major_split) else ""
        i += 2

        # ── Split major block into sub-questions (a), (b), (c) … ─────────────
        sub_split = re.split(r'\n\s*\(([a-z])\)\s+', '\n' + q_body)
        # Produces: [intro, 'a', 'text-of-(a)', 'b', 'text-of-(b)', ...]

        major_intro = sub_split[0].strip()   # text before first (a): passage, tables, etc.

        if len(sub_split) == 1:
            # No sub-questions: the entire block is a single leaf question
            marks_m = re.findall(r'\[(\d+)\]', q_body)
            marks = int(marks_m[-1]) if marks_m else 0
            if marks > 0:
                full_text = (preamble.strip() + "\n\n" if preamble.strip() else "") + q_body.strip()
                questions.append(ExtractedQuestion(
                    number=q_major,
                    text=full_text[:4000],
                    marks=marks,
                    question_type=_infer_question_type(q_body),
                ))
            continue

        j = 1
        while j < len(sub_split) - 1:
            sub_letter = sub_split[j]
            sub_body   = sub_split[j + 1] if j + 1 < len(sub_split) else ""
            j += 2

            # ── Split sub into sub-sub-questions (i), (ii), (iii) … ──────────
            subsub_split = re.split(r'\n\s*\(([ivxlcdm]+)\)\s+', '\n' + sub_body)
            # Produces: [sub_intro, 'i', 'text-of-(i)', 'ii', 'text-of-(ii)', ...]

            sub_intro = subsub_split[0].strip()  # text before first (i): e.g. "Using Table 1.1:"

            if len(subsub_split) == 1:
                # No sub-sub-questions: (letter) is the leaf
                marks_m = re.findall(r'\[(\d+)\]', sub_body)
                marks = int(marks_m[-1]) if marks_m else 0
                if marks > 0:
                    parts = [p for p in [major_intro, sub_body.strip()] if p]
                    full_text = "\n\n".join(parts)[:4000]
                    questions.append(ExtractedQuestion(
                        number=f"{q_major}({sub_letter})",
                        text=full_text,
                        marks=marks,
                        question_type=_infer_question_type(sub_body),
                    ))
                continue

            k = 1
            while k < len(subsub_split) - 1:
                roman = subsub_split[k]
                leaf_body = subsub_split[k + 1] if k + 1 < len(subsub_split) else ""
                k += 2

                marks_m = re.findall(r'\[(\d+)\]', leaf_body)
                marks = int(marks_m[-1]) if marks_m else 0
                if marks > 0:
                    parts = [p for p in [major_intro, sub_intro, leaf_body.strip()] if p]
                    full_text = "\n\n".join(parts)[:4000]
                    questions.append(ExtractedQuestion(
                        number=f"{q_major}({sub_letter})({roman})",
                        text=full_text,
                        marks=marks,
                        question_type=_infer_question_type(leaf_body),
                    ))

    logger.info("extract_questions_from_text: parsed %d question(s)", len(questions))
    return questions
