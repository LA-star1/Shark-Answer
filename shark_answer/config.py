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
    CLAUDE   = "claude"    # Anthropic  — claude-opus-4-6-20250901
    GPT4O    = "gpt4o"     # OpenAI     — gpt-5.2-thinking  (essays / general)
    O3PRO    = "o3pro"     # OpenAI     — o3-pro            (science / math)
    DEEPSEEK = "deepseek"  # DeepSeek   — deepseek-v3.2-speciale
    GEMINI   = "gemini"    # Google     — gemini-3.1-pro
    QWEN     = "qwen"      # Alibaba    — qwen3-max
    GROK     = "grok"      # xAI        — grok-2-vision-1212
    MINIMAX  = "minimax"   # MiniMax    — minimax-m2.5
    KIMI     = "kimi"      # Moonshot   — kimi-k2.5
    GLM      = "glm"       # Zhipu AI   — glm-5


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE_CONFIG — single source of truth for which models run in each pipeline
# ──────────────────────────────────────────────────────────────────────────────
#
# Pipeline A (Science & Math):  5 solvers  +  Claude as judge
# Pipeline B (Essay):           7 angle-writers + Claude as judge; MiniMax backup
# Pipeline C (Computer Science):4 solvers  +  Claude as judge
#
PIPELINE_CONFIG: dict[str, dict] = {
    "A_science": {
        # o3-pro: strongest math reasoning (top AIME scores)
        # deepseek: strong STEM competitor (IMO/IOI-level)
        # gemini: top GPQA science benchmark
        # claude: solver + dispute judge
        # glm: strong all-rounder supplement
        "solvers": [
            ModelProvider.O3PRO,
            ModelProvider.DEEPSEEK,
            ModelProvider.GEMINI,
            ModelProvider.CLAUDE,
            ModelProvider.GLM,
        ],
        "judge": ModelProvider.CLAUDE,
    },
    "B_essay": {
        # Each model is assigned a FIXED argument angle to prevent generic outputs.
        # Claude also acts as the scoring judge for all drafts.
        "angles": {
            ModelProvider.CLAUDE:   "orthodox/mainstream textbook view",
            ModelProvider.GPT4O:    "critical/contrarian perspective",
            ModelProvider.GEMINI:   "case-study driven (real-world examples)",
            ModelProvider.DEEPSEEK: "theoretical/academic framework",
            ModelProvider.QWEN:     "comparative analysis (cross-country/cross-policy)",
            ModelProvider.GLM:      "policy-oriented perspective",
            ModelProvider.KIMI:     "data/evidence-based empirical approach",
        },
        "judge": ModelProvider.CLAUDE,
        "backup": [ModelProvider.MINIMAX],
    },
    "C_cs": {
        # claude: primary + judge (SWE-bench leader)
        # minimax: strong software engineering
        # glm: high HumanEval score
        # deepseek: strong coder, low cost
        "solvers": [
            ModelProvider.CLAUDE,
            ModelProvider.MINIMAX,
            ModelProvider.GLM,
            ModelProvider.DEEPSEEK,
        ],
        "judge": ModelProvider.CLAUDE,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE_MODEL_CONFIG — legacy format consumed by config.get_pipeline_models()
# Derived automatically from PIPELINE_CONFIG so there is only ONE place to edit.
# ──────────────────────────────────────────────────────────────────────────────
PIPELINE_MODEL_CONFIG: dict[Pipeline, dict[str, list[ModelProvider]]] = {
    Pipeline.SCIENCE_MATH: {
        "primary": PIPELINE_CONFIG["A_science"]["solvers"],
        "judge":   [PIPELINE_CONFIG["A_science"]["judge"]],
    },
    Pipeline.ESSAY: {
        "brainstorm": list(PIPELINE_CONFIG["B_essay"]["angles"].keys()),
        "judge":      [PIPELINE_CONFIG["B_essay"]["judge"]],
        "backup":     PIPELINE_CONFIG["B_essay"]["backup"],
    },
    Pipeline.CS: {
        "primary": PIPELINE_CONFIG["C_cs"]["solvers"],
        "judge":   [PIPELINE_CONFIG["C_cs"]["judge"]],
    },
    Pipeline.PRACTICAL: {
        "primary": [ModelProvider.CLAUDE, ModelProvider.O3PRO],
    },
}

# Cost per 1M tokens (input/output) in USD — approximate, update as pricing changes
MODEL_COST_PER_M_TOKENS: dict[ModelProvider, dict[str, float]] = {
    ModelProvider.CLAUDE:   {"input": 15.00, "output": 75.00},   # Opus 4.6 pricing estimate
    ModelProvider.GPT4O:    {"input": 10.00, "output": 30.00},   # GPT-5.2 pricing estimate
    ModelProvider.O3PRO:    {"input": 20.00, "output": 80.00},   # o3-pro pricing estimate
    ModelProvider.DEEPSEEK: {"input":  0.27, "output":  1.10},
    ModelProvider.GEMINI:   {"input":  1.25, "output":  5.00},
    ModelProvider.QWEN:     {"input":  0.50, "output":  2.00},
    ModelProvider.GROK:     {"input":  5.00, "output": 15.00},
    ModelProvider.MINIMAX:  {"input":  0.50, "output":  1.50},
    ModelProvider.KIMI:     {"input":  1.00, "output":  3.00},
    ModelProvider.GLM:      {"input":  0.70, "output":  2.80},   # GLM-5 pricing estimate
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

        # Maps ModelProvider → (api_key_env, base_url_env)
        # O3PRO reuses the same OPENAI_API_KEY — two instances, different models.
        model_env_map: dict[ModelProvider, tuple[str, Optional[str]]] = {
            ModelProvider.CLAUDE:   ("ANTHROPIC_API_KEY",   None),
            ModelProvider.GPT4O:    ("OPENAI_API_KEY",      None),
            ModelProvider.O3PRO:    ("OPENAI_API_KEY",      None),   # same key, diff model
            ModelProvider.DEEPSEEK: ("DEEPSEEK_API_KEY",    "DEEPSEEK_BASE_URL"),
            ModelProvider.GEMINI:   ("GOOGLE_GEMINI_API_KEY", None),
            ModelProvider.QWEN:     ("QWEN_API_KEY",        "QWEN_BASE_URL"),
            ModelProvider.GROK:     ("GROK_API_KEY",        "GROK_BASE_URL"),
            ModelProvider.MINIMAX:  ("MINIMAX_API_KEY",     "MINIMAX_BASE_URL"),
            ModelProvider.KIMI:     ("KIMI_API_KEY",        "KIMI_BASE_URL"),
            ModelProvider.GLM:      ("GLM_API_KEY",         "GLM_BASE_URL"),
        }

        for provider, (key_env, url_env) in model_env_map.items():
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
