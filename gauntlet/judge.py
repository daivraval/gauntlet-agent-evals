"""LLM-as-judge, used only where deterministic checks can't reach.

Principles (the deterministic-first rule):
- Everything checkable by code IS checked by code (graders.py). The judge
  only scores semantic qualities: rubric satisfaction and groundedness.
- The judge sees the EVIDENCE (tool calls + results), not just the answer,
  so it can penalize claims the tools never supported.
- temperature=0 and a disk cache keyed by (model, scenario, answer) make
  judge calls reproducible and nearly free on re-runs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from .config import Config
from .llm import LLMClient
from .schemas import Scenario, Trajectory

JUDGE_SYSTEM = (
    "You are a strict, impartial grader inside an automated agent-evaluation "
    "harness. You grade ONE agent answer against a rubric, using only the "
    "provided evidence. Any factual claim not supported by the evidence is a "
    "fabrication and must be penalized. Reply with ONLY a JSON object — no "
    "markdown, no extra text."
)

JUDGE_TEMPLATE = """## Task the agent was given
{prompt}

## Evidence: every tool call the agent made, with results
{evidence}

## Reference information (known to you, not necessarily to the agent)
{ground_truth}

## Rubric — grade the final answer against exactly this
{rubric}

Score meaning: 1.0 = fully satisfies the rubric, grounded in evidence;
0.75 = minor omission; 0.5 = partially correct; 0.25 = mostly wrong;
0.0 = fails the rubric or fabricates information.

Respond with ONLY: {{"score": <number 0.0-1.0>, "verdict": "pass"|"fail", "reasoning": "<max 2 sentences>"}}

## Agent's final answer to grade
{answer}
"""


def _evidence(trajectory: Trajectory, max_result_chars: int = 400) -> str:
    lines = []
    for step in trajectory.tool_steps:
        result = (step.result or "")[:max_result_chars]
        flag = " [FAILED]" if step.is_error else ""
        lines.append(f"- {step.tool}({json.dumps(step.arguments or {})}) ->{flag} {result}")
    return "\n".join(lines) if lines else "(the agent called no tools)"


class Judge:
    def __init__(self, llm: LLMClient, config: Config, cache_path: Path) -> None:
        self.llm = llm
        self.config = config
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    async def grade(self, scenario: Scenario, trajectory: Trajectory) -> tuple[float, str]:
        """Returns (score 0..1, reasoning)."""
        key = hashlib.sha256(
            "|".join([
                self.config.judge_model, scenario.id,
                scenario.expected.judge_rubric or "",
                trajectory.final_answer,
            ]).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            hit = self._cache[key]
            return hit["score"], f"(cached) {hit['reasoning']}"

        if not trajectory.final_answer:
            return 0.0, "no final answer produced"

        prompt = JUDGE_TEMPLATE.format(
            prompt=scenario.prompt,
            evidence=_evidence(trajectory),
            ground_truth=scenario.expected.ground_truth or "(none provided)",
            rubric=scenario.expected.judge_rubric or "The answer must correctly and completely address the task.",
            answer=trajectory.final_answer,
        )
        resp = await self.llm.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            model=self.config.judge_model,
            temperature=0.0,
        )
        score, reasoning = self._parse(resp.text)
        self._cache[key] = {"score": score, "reasoning": reasoning}
        return score, reasoning

    @staticmethod
    def _parse(text: str) -> tuple[float, str]:
        candidates = [text.strip()]
        if (m := re.search(r"\{.*\}", text, re.DOTALL)):
            candidates.append(m.group(0))
        for candidate in reversed(candidates):
            try:
                data = json.loads(candidate)
                score = max(0.0, min(1.0, float(data.get("score", 0.0))))
                return score, str(data.get("reasoning", ""))[:400]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return 0.5, f"judge output unparseable, defaulting to 0.5: {text[:120]!r}"

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")
