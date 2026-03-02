"""Base provider interface and shared types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""
    content: str
    provider: str
    model_name: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.input_tokens + self.usage.output_tokens


class BaseProvider(ABC):
    """Abstract base class for all model providers."""

    provider_name: str = "base"
    default_model: str = ""

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name or self.default_model

    @abstractmethod
    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7,
                       max_tokens: int = 4096) -> ModelResponse:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def generate_with_image(self, prompt: str, image_data: bytes,
                                  system: str = "",
                                  temperature: float = 0.3,
                                  max_tokens: int = 4096) -> ModelResponse:
        """Generate text from a prompt + image (vision)."""
        ...

    async def _safe_generate(self, func, **kwargs) -> ModelResponse:
        """Wrap generation with error handling and timing."""
        start = time.time()
        try:
            result = await func(**kwargs)
            result.latency_seconds = time.time() - start
            return result
        except Exception as e:
            return ModelResponse(
                content="",
                provider=self.provider_name,
                model_name=self.model_name,
                latency_seconds=time.time() - start,
                success=False,
                error=str(e),
            )
