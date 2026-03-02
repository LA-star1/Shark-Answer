"""Configuration management for Shark Answer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Subject(str, Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATH = "math"
    FURTHER_MATH = "further_math"
    ECONOMICS = "economics"
    COMPUTER_SCIENCE = "computer_science"


class Pipeline(str, Enum):
    SCIENCE_MATH = "A"      # Physics, Chemistry, Biology, Math, Further Math
    ESSAY = "B"             # Economics, Biology essay questions
    CS = "C"                # Computer Science
    PRACTICAL = "D"         # Physics Practical Prediction


class Language(str, Enum):
    EN = "en"
    ZH = "zh"


# Subject → Pipeline routing
SUBJECT_PIPELINE_MAP: dict[Subject, Pipeline] = {
    Subject.PHYSICS: Pipeline.SCIENCE_MATH,
    Subject.CHEMISTRY: Pipeline.SCIENCE_MATH,
    Subject.BIOLOGY: Pipeline.SCIENCE_MATH,  # non-essay bio questions
    Subject.MATH: Pipeline.SCIENCE_MATH,
    Subject.FURTHER_MATH: Pipeline.SCIENCE_MATH,
    Subject.ECONOMICS: Pipeline.ESSAY,
    Subject.COMPUTER_SCIENCE: Pipeline.CS,
}


class ModelProvider(str, Enum):
    CLAUDE = "claude"
    GPT4O = "gpt4o"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    QWEN = "qwen"
    GROK = "grok"
    MINIMAX = "minimax"
    KIMI = "kimi"


# Model tier assignments per pipeline — easy to reconfigure
PIPELINE_MODEL_CONFIG: dict[Pipeline, dict[str, list[ModelProvider]]] = {
    Pipeline.SCIENCE_MATH: {
        "primary": [ModelProvider.CLAUDE, ModelProvider.GPT4O, ModelProvider.DEEPSEEK],
        "judge": [ModelProvider.CLAUDE],
    },
    Pipeline.ESSAY: {
        "brainstorm": [
            ModelProvider.CLAUDE, ModelProvider.GPT4O, ModelProvider.GEMINI,
            ModelProvider.DEEPSEEK, ModelProvider.QWEN,
        ],
        "judge": [ModelProvider.CLAUDE],
    },
    Pipeline.CS: {
        "primary": [ModelProvider.CLAUDE, ModelProvider.GPT4O],
        "supplementary": [ModelProvider.DEEPSEEK],
        "judge": [ModelProvider.CLAUDE],
    },
    Pipeline.PRACTICAL: {
        "primary": [ModelProvider.CLAUDE, ModelProvider.GPT4O],
    },
}

# Cost per 1M tokens (input/output) in USD — approximate, update as needed
MODEL_COST_PER_M_TOKENS: dict[ModelProvider, dict[str, float]] = {
    ModelProvider.CLAUDE:   {"input": 3.00, "output": 15.00},
    ModelProvider.GPT4O:    {"input": 2.50, "output": 10.00},
    ModelProvider.DEEPSEEK: {"input": 0.27, "output": 1.10},
    ModelProvider.GEMINI:   {"input": 1.25, "output": 5.00},
    ModelProvider.QWEN:     {"input": 0.50, "output": 2.00},
    ModelProvider.GROK:     {"input": 5.00, "output": 15.00},
    ModelProvider.MINIMAX:  {"input": 0.50, "output": 1.50},
    ModelProvider.KIMI:     {"input": 1.00, "output": 3.00},
}


@dataclass
class ModelConfig:
    """Configuration for a specific model provider."""
    provider: ModelProvider
    api_key: str
    base_url: Optional[str] = None
    model_name: Optional[str] = None  # override default model name
    enabled: bool = True


@dataclass
class AppConfig:
    """Application-wide configuration."""
    log_level: str = "INFO"
    max_concurrent_models: int = 5
    cost_budget_warning_usd: float = 5.0
    default_language: Language = Language.EN
    upload_dir: Path = Path("data/uploads")
    mark_scheme_dir: Path = Path("data/mark_schemes")
    examiner_profile_dir: Path = Path("data/examiner_profiles")
    max_answer_versions: int = 5
    models: dict[ModelProvider, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_concurrent_models=int(os.getenv("MAX_CONCURRENT_MODELS", "5")),
            cost_budget_warning_usd=float(os.getenv("COST_BUDGET_WARNING_USD", "5.0")),
            default_language=Language(os.getenv("DEFAULT_LANGUAGE", "en")),
            upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")),
        )

        # Load model configurations
        model_env_map: dict[ModelProvider, tuple[str, Optional[str], Optional[str]]] = {
            ModelProvider.CLAUDE:   ("ANTHROPIC_API_KEY", None, None),
            ModelProvider.GPT4O:    ("OPENAI_API_KEY", None, None),
            ModelProvider.DEEPSEEK: ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", None),
            ModelProvider.GEMINI:   ("GOOGLE_GEMINI_API_KEY", None, None),
            ModelProvider.QWEN:     ("QWEN_API_KEY", "QWEN_BASE_URL", None),
            ModelProvider.GROK:     ("GROK_API_KEY", "GROK_BASE_URL", None),
            ModelProvider.MINIMAX:  ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", None),
            ModelProvider.KIMI:     ("KIMI_API_KEY", "KIMI_BASE_URL", None),
        }

        for provider, (key_env, url_env, _) in model_env_map.items():
            api_key = os.getenv(key_env, "")
            if api_key:
                base_url = os.getenv(url_env) if url_env else None
                config.models[provider] = ModelConfig(
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                )

        config.upload_dir.mkdir(parents=True, exist_ok=True)
        config.mark_scheme_dir.mkdir(parents=True, exist_ok=True)
        config.examiner_profile_dir.mkdir(parents=True, exist_ok=True)

        return config

    def get_available_models(self, requested: list[ModelProvider]) -> list[ModelProvider]:
        """Return only models that have API keys configured."""
        return [m for m in requested if m in self.models and self.models[m].enabled]

    def get_pipeline_models(self, pipeline: Pipeline, tier: str) -> list[ModelProvider]:
        """Get available models for a pipeline tier."""
        requested = PIPELINE_MODEL_CONFIG.get(pipeline, {}).get(tier, [])
        return self.get_available_models(requested)
