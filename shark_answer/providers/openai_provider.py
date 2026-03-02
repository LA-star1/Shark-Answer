"""OpenAI GPT-4o provider."""

from __future__ import annotations

import base64

import openai

from shark_answer.providers.base import BaseProvider, ModelResponse, TokenUsage


class OpenAIProvider(BaseProvider):
    provider_name = "gpt4o"
    default_model = "gpt-4o"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_name: str | None = None):
        super().__init__(api_key, base_url, model_name)
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7,
                       max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate, prompt=prompt, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def _do_generate(self, prompt: str, system: str,
                           temperature: float, max_tokens: int) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
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
        messages = []
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

        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return ModelResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
