"""Provider registry — single point to get provider instances."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider
from shark_answer.providers.base import BaseProvider, ModelResponse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default model names per provider  (2026 flagship models)
# TODO: verify exact model strings against each provider's current API docs
#       before deploying to production.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODELS: dict[ModelProvider, str] = {
    # Anthropic — use claude-opus-4-5 if available; fall back to sonnet-4
    # TODO: verify latest Opus string via console.anthropic.com/docs/models
    ModelProvider.CLAUDE:   "claude-opus-4-5",

    # OpenAI  — two separate instances from the same API key
    # GPT4O keeps gpt-4o (vision-safe, known working); override via model_name in .env
    # when gpt-5.2-thinking or newer is confirmed available.
    ModelProvider.GPT4O:    "gpt-4o",
    # TEMP: org-verification pending → using o3-mini as fallback.
    # Swap back to "o3-pro" once OpenAI verifies the organisation.
    ModelProvider.O3PRO:    "o3-mini",          # TODO: revert to "o3-pro"

    # DeepSeek  — V3.2 (deepseek-chat = non-thinking; deepseek-reasoner = CoT/thinking)
    ModelProvider.DEEPSEEK: "deepseek-reasoner",            # deepseek-reasoner → DeepSeek-V3.2 CoT/thinking

    # Google Gemini  — 3.1 Pro
    ModelProvider.GEMINI:   "gemini-3.1-pro-preview",       # preview string confirmed via ListModels

    # Alibaba Qwen  — Qwen3-Max
    ModelProvider.QWEN:     "qwen3-max",                   # TODO: verify availability

    # xAI Grok  — vision-capable variant
    ModelProvider.GROK:     "grok-2-vision-1212",

    # MiniMax  — M2.5
    ModelProvider.MINIMAX:  "minimax-m2.5",                # TODO: verify model string

    # Moonshot Kimi  — K2.5
    ModelProvider.KIMI:     "kimi-k2.5",                   # TODO: verify model string

    # Zhipu AI  — GLM-5
    ModelProvider.GLM:      "glm-5",                       # TODO: verify release
}

# Default base URLs for providers that don't use the canonical OpenAI endpoint
DEFAULT_BASE_URLS: dict[ModelProvider, str] = {
    ModelProvider.DEEPSEEK: "https://api.deepseek.com",
    ModelProvider.QWEN:     "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore international endpoint
    ModelProvider.GROK:     "https://api.x.ai/v1",
    ModelProvider.MINIMAX:  "https://api.minimax.chat/v1",
    ModelProvider.KIMI:     "https://api.moonshot.cn/v1",
    ModelProvider.GLM:      "https://open.bigmodel.cn/api/paas/v4",
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

        elif provider in {ModelProvider.GPT4O, ModelProvider.O3PRO}:
            # Both use the OpenAI provider; differ only in model name.
            from shark_answer.providers.openai_provider import OpenAIProvider
            instance = OpenAIProvider(api_key=api_key, model_name=model_name)

        elif provider == ModelProvider.GEMINI:
            from shark_answer.providers.gemini_provider import GeminiProvider
            instance = GeminiProvider(api_key=api_key, model_name=model_name)

        else:
            # DeepSeek, Qwen, Grok, MiniMax, Kimi, GLM — all OpenAI-compatible REST
            from shark_answer.providers.openai_compat_provider import OpenAICompatProvider
            # Only Grok supports the image_url vision format among compat providers.
            # DeepSeek-chat, Qwen-plus, GLM text models do NOT support image inputs.
            vision_support = provider in {ModelProvider.GROK}
            # kimi-k2.5 is a reasoning model that only accepts temperature=1.
            fixed_temp = 1.0 if provider == ModelProvider.KIMI else None
            instance = OpenAICompatProvider(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                provider_name=provider.value,
                supports_vision=vision_support,
                fixed_temperature=fixed_temp,
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
        """Call multiple models in parallel, return all responses (including failures).

        Each model gets up to 2 attempts: 30 s timeout per attempt, 5 s delay between.
        """
        sem = asyncio.Semaphore(self.config.max_concurrent_models)

        async def _call(p: ModelProvider) -> ModelResponse:
            async with sem:
                inst = self.get(p)
                if inst is None:
                    return ModelResponse(
                        content="", provider=p.value, model_name="",
                        success=False, error="Provider not configured",
                    )
                last: ModelResponse = ModelResponse(
                    content="", provider=p.value,
                    model_name=getattr(inst, "model_name", ""),
                    success=False, error="All attempts failed",
                )
                for attempt in range(2):
                    logger.info("[%s] Calling (attempt %d)...", p.value, attempt + 1)
                    try:
                        last = await asyncio.wait_for(
                            inst.generate(
                                prompt=prompt, system=system,
                                temperature=temperature, max_tokens=max_tokens,
                            ),
                            timeout=30.0,
                        )
                        if last.success:
                            in_tok  = last.usage.input_tokens  if last.usage else 0
                            out_tok = last.usage.output_tokens if last.usage else 0
                            logger.info(
                                "[%s] OK (attempt %d) — %d in / %d out tokens",
                                p.value, attempt + 1, in_tok, out_tok,
                            )
                            return last
                        logger.warning(
                            "[%s] Error (attempt %d): %s", p.value, attempt + 1, last.error
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] Timeout after 30 s (attempt %d)", p.value, attempt + 1
                        )
                        last = ModelResponse(
                            content="", provider=p.value,
                            model_name=getattr(inst, "model_name", ""),
                            success=False, error="Timeout after 30 s",
                        )
                    if attempt == 0:
                        logger.info("[%s] Retrying in 5 s...", p.value)
                        await asyncio.sleep(5.0)
                return last

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
        """Call multiple models with an image in parallel.

        Each model gets up to 2 attempts: 30 s timeout per attempt, 5 s delay between.
        """
        sem = asyncio.Semaphore(self.config.max_concurrent_models)

        async def _call(p: ModelProvider) -> ModelResponse:
            async with sem:
                inst = self.get(p)
                if inst is None:
                    return ModelResponse(
                        content="", provider=p.value, model_name="",
                        success=False, error="Provider not configured",
                    )
                last: ModelResponse = ModelResponse(
                    content="", provider=p.value,
                    model_name=getattr(inst, "model_name", ""),
                    success=False, error="All attempts failed",
                )
                for attempt in range(2):
                    logger.info(
                        "[%s] Calling with image (attempt %d)...", p.value, attempt + 1
                    )
                    try:
                        last = await asyncio.wait_for(
                            inst.generate_with_image(
                                prompt=prompt, image_data=image_data,
                                system=system, temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            timeout=30.0,
                        )
                        if last.success:
                            logger.info(
                                "[%s] Image OK (attempt %d)", p.value, attempt + 1
                            )
                            return last
                        logger.warning(
                            "[%s] Image error (attempt %d): %s",
                            p.value, attempt + 1, last.error,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] Image timeout after 30 s (attempt %d)", p.value, attempt + 1
                        )
                        last = ModelResponse(
                            content="", provider=p.value,
                            model_name=getattr(inst, "model_name", ""),
                            success=False, error="Timeout after 30 s",
                        )
                    if attempt == 0:
                        logger.info("[%s] Retrying in 5 s...", p.value)
                        await asyncio.sleep(5.0)
                return last

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)
