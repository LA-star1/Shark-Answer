"""Cost tracking for API calls."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from shark_answer.config import MODEL_COST_PER_M_TOKENS, ModelProvider
from shark_answer.providers.base import ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single API call cost record."""
    provider: str
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    subject: str = ""
    pipeline: str = ""
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """Tracks token usage and estimated cost across all API calls."""

    def __init__(self, budget_warning_usd: float = 5.0):
        self.entries: list[CostEntry] = []
        self.budget_warning_usd = budget_warning_usd
        self._session_total: float = 0.0

    def record(self, response: ModelResponse, subject: str = "",
               pipeline: str = "") -> CostEntry:
        """Record cost for a model response."""
        provider_enum = self._resolve_provider(response.provider)
        rates = MODEL_COST_PER_M_TOKENS.get(
            provider_enum,
            {"input": 1.0, "output": 3.0},  # fallback rates
        )
        cost = (
            response.usage.input_tokens * rates["input"] / 1_000_000
            + response.usage.output_tokens * rates["output"] / 1_000_000
        )

        entry = CostEntry(
            provider=response.provider,
            model_name=response.model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            subject=subject,
            pipeline=pipeline,
        )
        self.entries.append(entry)
        self._session_total += cost

        if self._session_total >= self.budget_warning_usd:
            logger.warning(
                "Cost warning: session total $%.4f exceeds budget $%.2f",
                self._session_total, self.budget_warning_usd,
            )

        return entry

    def record_batch(self, responses: list[ModelResponse], subject: str = "",
                     pipeline: str = "") -> list[CostEntry]:
        """Record cost for multiple responses."""
        return [self.record(r, subject, pipeline) for r in responses if r.success]

    @property
    def total_cost(self) -> float:
        return self._session_total

    def cost_by_provider(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self.entries:
            result[e.provider] = result.get(e.provider, 0.0) + e.cost_usd
        return result

    def cost_by_subject(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self.entries:
            key = e.subject or "unknown"
            result[key] = result.get(key, 0.0) + e.cost_usd
        return result

    def summary(self) -> dict:
        """Return a summary dict suitable for API response."""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_calls": len(self.entries),
            "total_input_tokens": sum(e.input_tokens for e in self.entries),
            "total_output_tokens": sum(e.output_tokens for e in self.entries),
            "by_provider": {
                k: round(v, 6) for k, v in self.cost_by_provider().items()
            },
            "by_subject": {
                k: round(v, 6) for k, v in self.cost_by_subject().items()
            },
        }

    @staticmethod
    def _resolve_provider(name: str) -> ModelProvider:
        try:
            return ModelProvider(name)
        except ValueError:
            return ModelProvider.DEEPSEEK  # fallback for cost calc
