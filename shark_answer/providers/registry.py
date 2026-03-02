"""Provider registry — single point to get provider instances."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider
from shark_answer.providers.base import BaseProvider, ModelResponse

logger = logging.getLogger(__name__)

# Default model names per provider
DEFAULT_MODELS: dict[ModelProvider, str] = {
    ModelProvider.CLAUDE:   "claude-sonnet-4-20250514",
    ModelProvider.GPT4O:    "gpt-4o",
    ModelProvider.DEEPSEEK: "deepseek-chat",
    ModelProvider.GEMINI:   "gemini-2.0-flash",
    ModelProvider.QWEN:     "qwen-plus",
    ModelProvider.GROK:     "grok-2",
    ModelProvider.MINIMAX:  "MiniMax-Text-01",
    ModelProvider.KIMI:     "moonshot-v1-128k",
}

# Default base URLs
DEFAULT_BASE_URLS: dict[ModelProvider, str] = {
    ModelProvider.DEEPSEEK: "https://api.deepseek.com",
    ModelProvider.QWEN:     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ModelProvider.GROK:     "https://api.x.ai/v1",
    ModelProvider.MINIMAX:  "https://api.minimax.chat/v1",
    ModelProvider.KIMI:     "https://api.moonshot.cn/v1",
}


class ProviderRegistry:
    """Creates and caches provider instances."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._providers: dict[ModelProvider, BaseProvider] = {}

    def get(self, provider: ModelProvider) -> Optional[BaseProvider]:
        """Get or create a provider instance. Returns None if not configured."""
        if provider in self._providers:
            return self._providers[provider]

        model_cfg = self.config.models.get(provider)
        if not model_cfg or not model_cfg.api_key:
            logger.warning("Provider %s not configured (no API key)", provider.value)
            return None

        api_key = model_cfg.api_key
        base_url = model_cfg.base_url or DEFAULT_BASE_URLS.get(provider)
        model_name = model_cfg.model_name or DEFAULT_MODELS.get(provider, "")

        instance: BaseProvider

        if provider == ModelProvider.CLAUDE:
            from shark_answer.providers.claude_provider import ClaudeProvider
            instance = ClaudeProvider(api_key=api_key, model_name=model_name)
        elif provider == ModelProvider.GPT4O:
            from shark_answer.providers.openai_provider import OpenAIProvider
            instance = OpenAIProvider(api_key=api_key, model_name=model_name)
        elif provider == ModelProvider.GEMINI:
            from shark_answer.providers.gemini_provider import GeminiProvider
            instance = GeminiProvider(api_key=api_key, model_name=model_name)
        else:
            # DeepSeek, Qwen, Grok, MiniMax, Kimi — all OpenAI-compatible
            from shark_answer.providers.openai_compat_provider import OpenAICompatProvider
            vision_support = provider in {
                ModelProvider.DEEPSEEK, ModelProvider.QWEN, ModelProvider.GROK,
            }
            instance = OpenAICompatProvider(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                provider_name=provider.value,
                supports_vision=vision_support,
            )

        self._providers[provider] = instance
        return instance

    async def call_models_parallel(
        self,
        providers: list[ModelProvider],
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[ModelResponse]:
        """Call multiple models in parallel, return all responses (including failures)."""
        sem = asyncio.Semaphore(self.config.max_concurrent_models)

        async def _call(p: ModelProvider) -> ModelResponse:
            async with sem:
                inst = self.get(p)
                if inst is None:
                    return ModelResponse(
                        content="", provider=p.value, model_name="",
                        success=False, error="Provider not configured",
                    )
                return await inst.generate(
                    prompt=prompt, system=system,
                    temperature=temperature, max_tokens=max_tokens,
                )

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)

    async def call_models_with_image_parallel(
        self,
        providers: list[ModelProvider],
        prompt: str,
        image_data: bytes,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> list[ModelResponse]:
        """Call multiple models with an image in parallel."""
        sem = asyncio.Semaphore(self.config.max_concurrent_models)

        async def _call(p: ModelProvider) -> ModelResponse:
            async with sem:
                inst = self.get(p)
                if inst is None:
                    return ModelResponse(
                        content="", provider=p.value, model_name="",
                        success=False, error="Provider not configured",
                    )
                return await inst.generate_with_image(
                    prompt=prompt, image_data=image_data,
                    system=system, temperature=temperature,
                    max_tokens=max_tokens,
                )

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)
