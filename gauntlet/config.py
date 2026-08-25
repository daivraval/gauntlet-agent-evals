"""Runtime configuration, loaded from .env / environment variables.

GAUNTLET talks to any OpenAI-compatible endpoint, so one config switches the
whole harness between Groq, OpenAI, the Hugging Face router, or local Ollama.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

PROVIDER_PRESETS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "hf":     {"base_url": "https://router.huggingface.co/v1", "model": "Qwen/Qwen2.5-72B-Instruct"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Config:
    base_url: str = field(default_factory=lambda: _env("GAUNTLET_BASE_URL", PROVIDER_PRESETS["groq"]["base_url"]))
    api_key: str = field(default_factory=lambda: _env("GAUNTLET_API_KEY") or _env("GROQ_API_KEY") or _env("OPENAI_API_KEY"))
    agent_model: str = field(default_factory=lambda: _env("GAUNTLET_AGENT_MODEL", PROVIDER_PRESETS["groq"]["model"]))
    judge_model: str = field(default_factory=lambda: _env("GAUNTLET_JUDGE_MODEL"))  # defaults to agent_model

    temperature: float = field(default_factory=lambda: _env_float("GAUNTLET_TEMPERATURE", 0.0))
    max_steps: int = 10                # max LLM calls per scenario (loop guard)
    request_timeout_s: float = 60.0
    max_retries: int = 3               # transport-level retries (429 / 5xx)
    scenario_timeout_s: float = 180.0
    concurrency: int = 4
    pass_threshold: float = field(default_factory=lambda: _env_float("GAUNTLET_PASS_THRESHOLD", 0.7))
    judge_enabled: bool = True

    # Optional $ per 1M tokens, for the cost column in reports.
    price_input_per_1m: float = field(default_factory=lambda: _env_float("GAUNTLET_PRICE_INPUT_PER_1M", 0.0))
    price_output_per_1m: float = field(default_factory=lambda: _env_float("GAUNTLET_PRICE_OUTPUT_PER_1M", 0.0))

    def __post_init__(self) -> None:
        if not self.judge_model:
            self.judge_model = self.agent_model

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.price_input_per_1m
            + completion_tokens * self.price_output_per_1m
        ) / 1_000_000

    def require_api_key(self) -> None:
        if not self.api_key:
            raise SystemExit(
                "No API key configured. Copy .env.example to .env and set "
                "GAUNTLET_API_KEY (a free Groq key from https://console.groq.com works), "
                "or run the offline demo: python run_evals.py run --agent scripted --no-judge"
            )
