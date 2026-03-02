"""Google Gemini provider."""

from __future__ import annotations

from google import genai
from google.genai import types

from shark_answer.providers.base import BaseProvider, ModelResponse, TokenUsage


class GeminiProvider(BaseProvider):
    provider_name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_name: str | None = None):
        super().__init__(api_key, base_url, model_name)
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7,
                       max_tokens: int = 4096) -> ModelResponse:
        return await self._safe_generate(
            self._do_generate, prompt=prompt, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def _do_generate(self, prompt: str, system: str,
                           temperature: float, max_tokens: int) -> ModelResponse:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system

        resp = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        text = resp.text or ""
        usage_meta = resp.usage_metadata
        return ModelResponse(
            content=text,
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=usage_meta.prompt_token_count if usage_meta else 0,
                output_tokens=usage_meta.candidates_token_count if usage_meta else 0,
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
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system

        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        resp = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[image_part, prompt],
            config=config,
        )
        text = resp.text or ""
        usage_meta = resp.usage_metadata
        return ModelResponse(
            content=text,
            provider=self.provider_name,
            model_name=self.model_name,
            usage=TokenUsage(
                input_tokens=usage_meta.prompt_token_count if usage_meta else 0,
                output_tokens=usage_meta.candidates_token_count if usage_meta else 0,
            ),
        )
