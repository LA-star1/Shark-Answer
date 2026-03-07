"""OpenAI provider — supports both Chat Completions and Responses API.

Model routing:
  - "o3-pro" (and any future "o3-pro-*") → Responses API
    (client.responses.create, output via response.output_text)
  - All other models, including o3, o3-mini, o1, gpt-4o → Chat Completions API

Fallback:
  - If the Responses API call for o3-pro fails for any reason, the provider
    automatically retries the same prompt using Chat Completions with model "o3".
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re

import openai

from shark_answer.providers.base import BaseProvider, ModelResponse, TokenUsage

logger = logging.getLogger(__name__)


def _is_o_series(model_name: str) -> bool:
    """Return True for OpenAI o1/o3 reasoning models (Chat Completions path).

    These require ``max_completion_tokens`` instead of ``max_tokens`` and do
    not support a custom ``temperature``.
    Covers: o1, o1-mini, o3, o3-mini — but NOT o3-pro (uses Responses API).
    """
    import re
    return bool(re.match(r"o[13](?!-pro)", model_name))


def _is_responses_api_model(model_name: str) -> bool:
    """Return True for models that require the Responses API (e.g. o3-pro)."""
    return model_name.startswith("o3-pro")


class OpenAIProvider(BaseProvider):
    provider_name = "gpt4o"
    default_model = "gpt-4o"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_name: str | None = None):
        super().__init__(api_key, base_url, model_name)
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7,
                       max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate, prompt=prompt, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def generate_with_image(self, prompt: str, image_data: bytes,
                                  system: str = "",
                                  temperature: float = 0.3,
                                  max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate_image, prompt=prompt, image_data=image_data,
            system=system, temperature=temperature, max_tokens=max_tokens,
        )

    # ── Internal implementation ────────────────────────────────────────────

    async def _do_generate(self, prompt: str, system: str,
                           temperature: float, max_tokens: int) -> ModelResponse:
        if _is_responses_api_model(self.model_name):
            return await self._do_generate_responses(prompt, system, max_tokens)
        return await self._do_generate_chat(
            prompt, system, temperature, max_tokens, model_override=None
        )

    async def _do_generate_responses(
        self, prompt: str, system: str, max_tokens: int,
    ) -> ModelResponse:
        """Call OpenAI Responses API (o3-pro) with a single 429 retry.

        If the first call returns a 429 rate-limit error, we extract the
        ``retry-after`` wait time from the error message (defaulting to 30 s),
        sleep for min(retry_after, 60) seconds, then try exactly once more.
        Any other failure is returned immediately as a clean failure response.
        """
        for attempt in range(2):
            try:
                # Responses API requires max_output_tokens >= 16
                resp = await self.client.responses.create(
                    model=self.model_name,
                    instructions=system or None,
                    input=prompt,
                    max_output_tokens=max(max_tokens, 16),
                )
                content = resp.output_text or ""
                usage = getattr(resp, "usage", None)
                return ModelResponse(
                    content=content,
                    provider=self.provider_name,
                    model_name=self.model_name,
                    usage=TokenUsage(
                        input_tokens=(usage.input_tokens  or 0) if usage else 0,
                        output_tokens=(usage.output_tokens or 0) if usage else 0,
                    ),
                )
            except openai.RateLimitError as exc:
                err_str = str(exc)
                # insufficient_quota = account ran out of credits — no point retrying
                if "insufficient_quota" in err_str:
                    logger.warning(
                        "[%s] o3-pro quota exhausted (insufficient_quota), skipping",
                        self.provider_name,
                    )
                    break   # immediate failure, no retry
                # rate_limit_exceeded = temporary throttle — wait and retry once
                if attempt == 0:
                    m = re.search(r'try again after (\d+(?:\.\d+)?)\s*s', err_str, re.I)
                    wait = float(m.group(1)) if m else 30.0
                    wait = min(wait, 60.0)   # cap at 60 s
                    logger.warning(
                        "[%s] o3-pro rate limited (429), waiting %.1fs before retry…",
                        self.provider_name, wait,
                    )
                    await asyncio.sleep(wait)
                    continue   # retry once
                # Second attempt also hit rate limit — give up gracefully
                logger.warning(
                    "[%s] o3-pro rate limited on retry, returning failure: %s",
                    self.provider_name, exc,
                )
                break
            except Exception as exc:
                # Non-429 error — return clean failure immediately (no retry)
                logger.warning("[%s] o3-pro Responses API failed: %s", self.provider_name, exc)
                break

        from shark_answer.providers.base import ModelResponse as MR
        return MR(
            content="",
            provider=self.provider_name,
            model_name=self.model_name,
            success=False,
            error="rate_limited",
        )

    async def _do_generate_chat(
        self, prompt: str, system: str,
        temperature: float, max_tokens: int,
        model_override: str | None = None,
    ) -> ModelResponse:
        """Standard Chat Completions API path."""
        model = model_override or self.model_name
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # o1/o3 (non-pro) reasoning models: no temperature, different token param
        if _is_o_series(model):
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model_name=model,
            usage=TokenUsage(
                input_tokens=(usage.prompt_tokens     or 0) if usage else 0,
                output_tokens=(usage.completion_tokens or 0) if usage else 0,
            ),
        )

    async def _do_generate_image(self, prompt: str, image_data: bytes,
                                 system: str, temperature: float,
                                 max_tokens: int) -> ModelResponse:
        """Vision call — always uses Chat Completions (Responses API has no vision yet)."""
        # o3-pro doesn't support vision; use gpt-4o for image calls on o3-pro instances.
        model = "gpt-4o" if _is_responses_api_model(self.model_name) else self.model_name

        b64 = base64.standard_b64encode(image_data).decode("utf-8")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        })

        if _is_o_series(model):
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model_name=model,
            usage=TokenUsage(
                input_tokens=(usage.prompt_tokens     or 0) if usage else 0,
                output_tokens=(usage.completion_tokens or 0) if usage else 0,
            ),
        )
