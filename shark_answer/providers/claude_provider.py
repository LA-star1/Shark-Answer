"""Anthropic Claude provider."""

from __future__ import annotations

import base64

import anthropic

from shark_answer.providers.base import BaseProvider, ModelResponse, TokenUsage


class ClaudeProvider(BaseProvider):
    provider_name = "claude"
    default_model = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_name: str | None = None):
        super().__init__(api_key, base_url, model_name)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7,
                       max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate, prompt=prompt, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def _do_generate(self, prompt: str, system: str,
                           temperature: float, max_tokens: int) -> ModelResponse:
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = await self.client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return ModelResponse(
            content=text,
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens  or 0,
                output_tokens=resp.usage.output_tokens or 0,
            ),
        )

    async def generate_with_image(self, prompt: str, image_data: bytes,
                                  system: str = "",
                                  temperature: float = 0.3,
                                  max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate_image, prompt=prompt, image_data=image_data,
            system=system, temperature=temperature, max_tokens=max_tokens,
        )

    async def _do_generate_image(self, prompt: str, image_data: bytes,
                                 system: str, temperature: float,
                                 max_tokens: int) -> ModelResponse:
        b64 = base64.standard_b64encode(image_data).decode("utf-8")
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        if system:
            kwargs["system"] = system

        resp = await self.client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return ModelResponse(
            content=text,
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens  or 0,
                output_tokens=resp.usage.output_tokens or 0,
            ),
        )
