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
    # Anthropic — Opus 4.5 (stable API alias, 4-6 string returns 404)
    ModelProvider.CLAUDE:   "claude-opus-4-5",

    # OpenAI  — two separate instances from the same API key
    # GPT4O keeps gpt-4o (vision-safe, known working)
    ModelProvider.GPT4O:    "gpt-4o",
    # TEMP: org-verification pending → using o3-mini as fallback.
    # Swap back to "o3-pro" once OpenAI verifies the organisation.
    ModelProvider.O3PRO:    "o3-mini",          # TODO: revert to "o3-pro"

    # DeepSeek  — deepseek-chat = V3.2 non-thinking; deepseek-reasoner = CoT/thinking
    # Pipeline A (science/math) uses deepseek-reasoner (CoT/thinking mode)
    ModelProvider.DEEPSEEK: "deepseek-reasoner",

    # Google Gemini  — 2.5 Pro (stable, replaces 3.1-pro-preview)
    ModelProvider.GEMINI:   "gemini-2.5-pro",

    # Alibaba Qwen  — Qwen 3.5 Plus
    ModelProvider.QWEN:     "qwen3.5-plus",

    # xAI Grok  — DISABLED: grok-2-vision-1212 returns 404, not worth debugging
    # ModelProvider.GROK:  "grok-2-vision-1212",

    # MiniMax  — Text-01 (confirmed stable API string)
    ModelProvider.MINIMAX:  "MiniMax-Text-01",

    # Moonshot Kimi  — moonshot-v1-128k (standard 128k model, replaces kimi-k2.5 reasoning)
    ModelProvider.KIMI:     "moonshot-v1-128k",

    # Zhipu AI  — GLM-5
    ModelProvider.GLM:      "glm-5",
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
            # moonshot-v1-128k is a standard model — no fixed temperature needed.
            # (old kimi-k2.5 was a reasoning model requiring temp=1.0; removed)
            fixed_temp = None
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

        Each model gets 1 attempt with a 15 s timeout (fast-fail to avoid blocking
        the whole gather on one sluggish or overloaded provider).
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
                logger.info("[%s] Calling...", p.value)
                try:
                    resp = await asyncio.wait_for(
                        inst.generate(
                            prompt=prompt, system=system,
                            temperature=temperature, max_tokens=max_tokens,
                        ),
                        timeout=15.0,
                    )
                    if resp.success:
                        in_tok  = resp.usage.input_tokens  if resp.usage else 0
                        out_tok = resp.usage.output_tokens if resp.usage else 0
                        logger.info(
                            "[%s] OK — %d in / %d out tokens",
                            p.value, in_tok, out_tok,
                        )
                    else:
                        logger.warning("[%s] Error: %s", p.value, resp.error)
                    return resp
                except asyncio.TimeoutError:
                    logger.warning("[%s] Timeout after 15 s", p.value)
                    return ModelResponse(
                        content="", provider=p.value,
                        model_name=getattr(inst, "model_name", ""),
                        success=False, error="Timeout after 15 s",
                    )

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)

    async def call_with_fallback(
        self,
        providers_chain: list[ModelProvider],
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Try providers sequentially; return the first successful response.

        Used for critical single-call steps (judge, scorer, explanation) where
        quality matters but the pipeline must not stall when one provider is down.
        Each provider gets one attempt with a 15 s timeout before the next is tried.
        """
        for p in providers_chain:
            inst = self.get(p)
            if inst is None:
                logger.debug("[fallback] %s not configured, skipping", p.value)
                continue
            logger.info("[fallback] Trying %s...", p.value)
            try:
                resp = await asyncio.wait_for(
                    inst.generate(
                        prompt=prompt, system=system,
                        temperature=temperature, max_tokens=max_tokens,
                    ),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning("[fallback] %s timed out after 15 s — trying next", p.value)
                continue
            except Exception as exc:
                logger.warning("[fallback] %s raised %s — trying next", p.value, exc)
                continue

            if resp.success:
                logger.info("[fallback] %s succeeded", p.value)
                return resp
            logger.warning("[fallback] %s failed (%s) — trying next", p.value, resp.error)

        # All providers in the chain failed
        providers_str = ", ".join(p.value for p in providers_chain)
        logger.error("[fallback] All providers failed: %s", providers_str)
        return ModelResponse(
            content="", provider="fallback_chain", model_name="",
            success=False,
            error=f"All providers failed ({providers_str})",
        )

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

        Each model gets 1 attempt with a 15 s timeout.
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
                logger.info("[%s] Calling with image...", p.value)
                try:
                    resp = await asyncio.wait_for(
                        inst.generate_with_image(
                            prompt=prompt, image_data=image_data,
                            system=system, temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        timeout=15.0,
                    )
                    if resp.success:
                        logger.info("[%s] Image OK", p.value)
                    else:
                        logger.warning("[%s] Image error: %s", p.value, resp.error)
                    return resp
                except asyncio.TimeoutError:
                    logger.warning("[%s] Image timeout after 15 s", p.value)
                    return ModelResponse(
                        content="", provider=p.value,
                        model_name=getattr(inst, "model_name", ""),
                        success=False, error="Timeout after 15 s",
                    )

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)
