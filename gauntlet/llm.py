"""Async LLM client for any OpenAI-compatible endpoint.

One thin wrapper used by both the agent under test and the LLM judge:
- exponential backoff with jitter on 429 / 5xx / timeouts
- per-call latency and token accounting (fed into the trajectory)
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .config import Config

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_message: Optional[dict[str, Any]] = None  # to append back into the transcript


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-set",
            timeout=config.request_timeout_s,
            max_retries=0,  # we handle retries ourselves so they show up in logs
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._chat_once(messages, tools, model, temperature)
            except (RateLimitError, APITimeoutError, APIConnectionError) as err:
                last_err = err
            except APIStatusError as err:
                if err.status_code not in RETRYABLE_STATUS:
                    raise
                last_err = err
            # exponential backoff with jitter: ~1s, 2s, 4s
            await asyncio.sleep((2**attempt) + random.uniform(0, 0.5))
        raise RuntimeError(f"LLM call failed after {self.config.max_retries + 1} attempts: {last_err}")

    async def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        model: Optional[str],
        temperature: Optional[float],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.config.agent_model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        start = time.perf_counter()
        resp = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        msg = resp.choices[0].message
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            raw_message={
                "role": "assistant",
                "content": msg.content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            },
        )
