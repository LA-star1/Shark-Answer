"""Provider registry — single point to get provider instances."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from shark_answer.config import AppConfig, ModelProvider
from shark_answer.providers.base import BaseProvider, ModelResponse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Per-call-type timeout constants
# Sending ~29k tokens of KB context per solver call means models need time to
# process the full context before generating.  Different steps have different
# latency profiles:
#
#   SOLVER_TIMEOUT  — default for solver models (non-reasoning)
#   JUDGE_TIMEOUT   — judge / scorer / explain calls
#   SHORT_TIMEOUT   — fast path for 1–3 mark questions (explicit override)
#   HEALTH_TIMEOUT  — lightweight debug / health-check probes
# ──────────────────────────────────────────────────────────────────────────────
SHORT_TIMEOUT:  float = 30.0   # seconds — fast path for 1–3 mark questions
JUDGE_TIMEOUT:  float = 45.0   # seconds — judge / scorer / explain calls
SOLVER_TIMEOUT: float = 60.0   # seconds — default solver timeout (non-reasoning)
HEALTH_TIMEOUT: float = 10.0   # seconds — debug / health-check probes

# ──────────────────────────────────────────────────────────────────────────────
# Per-model solver timeouts  (used when caller does NOT pass an explicit timeout)
# Reasoning models "think" before answering — they need much longer windows.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_TIMEOUTS: dict[ModelProvider, float] = {
    # OpenAI o3-pro: Responses API, can take minutes on hard math
    ModelProvider.O3PRO:    180.0,
    # DeepSeek-reasoner: extended chain-of-thought
    ModelProvider.DEEPSEEK: 120.0,
    # Qwen qwq-plus: step-by-step reasoning model
    ModelProvider.QWEN:     90.0,
    # All others: standard 60 s (set below via .get() default)
}

# ──────────────────────────────────────────────────────────────────────────────
# Default model names per provider  (2026 flagship reasoning models)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODELS: dict[ModelProvider, str] = {
    # Anthropic — claude-opus-4-6 (latest Opus; date suffix not supported by API)
    ModelProvider.CLAUDE:   "claude-opus-4-6",

    # OpenAI  — two separate instances from the same API key
    # GPT4O keeps gpt-4o (vision-safe, general / essay / CS)
    ModelProvider.GPT4O:    "gpt-4o",
    # O3PRO uses the Responses API (see openai_provider.py); fallback → o3
    ModelProvider.O3PRO:    "o3-pro",

    # DeepSeek  — deepseek-reasoner = R1 CoT/thinking model (IMO gold medal level)
    ModelProvider.DEEPSEEK: "deepseek-reasoner",

    # Google Gemini  — 2.5 Pro
    ModelProvider.GEMINI:   "gemini-2.5-pro",

    # Alibaba Qwen  — qwq-plus: dedicated math/reasoning model (replaces qwen3.5-plus)
    ModelProvider.QWEN:     "qwq-plus",

    # xAI Grok  — DISABLED: grok-2-vision-1212 returns 404
    # ModelProvider.GROK:  "grok-2-vision-1212",

    # MiniMax  — Text-01 (confirmed stable)
    ModelProvider.MINIMAX:  "MiniMax-Text-01",

    # Moonshot Kimi  — moonshot-v1-128k
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
        timeout: float | None = None,
    ) -> list[ModelResponse]:
        """Call multiple models in parallel, return all responses (including failures).

        Each model gets 1 attempt.  ``timeout`` defaults to SOLVER_TIMEOUT (60 s)
        but callers can pass a shorter value (e.g. SHORT_TIMEOUT for fast-path
        questions) to fail-fast on slow providers.
        """
        # If the caller passed an explicit timeout, use it for every provider.
        # If None (default), apply per-model timeouts from MODEL_TIMEOUTS so that
        # reasoning models (o3-pro, deepseek-reasoner, qwq-plus) get longer windows.
        _fixed_timeout = timeout  # None means "use per-model defaults"
        sem = asyncio.Semaphore(self.config.max_concurrent_models)

        async def _call(p: ModelProvider) -> ModelResponse:
            async with sem:
                inst = self.get(p)
                if inst is None:
                    return ModelResponse(
                        content="", provider=p.value, model_name="",
                        success=False, error="Provider not configured",
                    )
                # Per-model timeout: explicit caller value wins; otherwise use
                # MODEL_TIMEOUTS (for reasoning models) or SOLVER_TIMEOUT default.
                model_timeout = (
                    _fixed_timeout
                    if _fixed_timeout is not None
                    else MODEL_TIMEOUTS.get(p, SOLVER_TIMEOUT)
                )
                # 2-attempt retry: first attempt → wait 3 s → second attempt.
                # Retry on timeout, API error, OR empty response (success=True but
                # content="", which Gemini can return on ambiguous vision questions).
                last_resp: ModelResponse | None = None
                for attempt in range(2):
                    if attempt > 0:
                        logger.info("[%s] Retrying after 3 s (attempt 2/2)...", p.value)
                        await asyncio.sleep(3)
                    logger.info("[%s] Calling... (timeout=%.0fs, attempt=%d/2)",
                                p.value, model_timeout, attempt + 1)
                    try:
                        resp = await asyncio.wait_for(
                            inst.generate(
                                prompt=prompt, system=system,
                                temperature=temperature, max_tokens=max_tokens,
                            ),
                            timeout=model_timeout,
                        )
                        if resp.success and resp.content.strip():
                            in_tok  = resp.usage.input_tokens  if resp.usage else 0
                            out_tok = resp.usage.output_tokens if resp.usage else 0
                            logger.info(
                                "[%s] OK — %d in / %d out tokens",
                                p.value, in_tok, out_tok,
                            )
                            return resp
                        elif resp.success and not resp.content.strip():
                            logger.warning("[%s] Empty response (attempt %d/2)",
                                           p.value, attempt + 1)
                            last_resp = ModelResponse(
                                content="", provider=p.value,
                                model_name=getattr(inst, "model_name", ""),
                                success=False, error="Empty response",
                            )
                        else:
                            logger.warning("[%s] Error: %s (attempt %d/2)",
                                           p.value, resp.error, attempt + 1)
                            last_resp = resp
                    except asyncio.TimeoutError:
                        logger.warning("[%s] Timeout after %.0f s (attempt %d/2)",
                                       p.value, model_timeout, attempt + 1)
                        last_resp = ModelResponse(
                            content="", provider=p.value,
                            model_name=getattr(inst, "model_name", ""),
                            success=False, error=f"Timeout after {model_timeout:.0f} s",
                        )
                # Both attempts failed — return last known failure
                return last_resp or ModelResponse(
                    content="", provider=p.value,
                    model_name=getattr(inst, "model_name", ""),
                    success=False, error="Both attempts failed",
                )

        results = await asyncio.gather(*[_call(p) for p in providers])
        result_list = list(results)

        # Per-model 2-attempt retry (inside _call above) already handles transient
        # failures — no post-hoc batch retry needed.
        return result_list

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
        Each provider gets one attempt with a JUDGE_TIMEOUT (45 s) before the
        next is tried — enough for ~29k-token judge prompts to be processed.
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
                    timeout=JUDGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("[fallback] %s timed out after %.0f s — trying next",
                               p.value, JUDGE_TIMEOUT)
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

        Each model gets 1 attempt with a SOLVER_TIMEOUT (60 s) timeout.
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
                        timeout=SOLVER_TIMEOUT,
                    )
                    if resp.success:
                        logger.info("[%s] Image OK", p.value)
                    else:
                        logger.warning("[%s] Image error: %s", p.value, resp.error)
                    return resp
                except asyncio.TimeoutError:
                    logger.warning("[%s] Image timeout after %.0f s", p.value, SOLVER_TIMEOUT)
                    return ModelResponse(
                        content="", provider=p.value,
                        model_name=getattr(inst, "model_name", ""),
                        success=False, error=f"Timeout after {SOLVER_TIMEOUT:.0f} s",
                    )

        results = await asyncio.gather(*[_call(p) for p in providers])
        return list(results)
