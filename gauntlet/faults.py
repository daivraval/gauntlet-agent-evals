"""Fault injection: makes tools fail on purpose, per scenario config.

This is chaos engineering for agents. A FaultSpec can make a tool return a
503, a timeout, corrupted bytes, an empty result set, or a 429 — either for
the first N calls (transient) or forever (permanent). The trajectory marks
injected steps `faulted=True`, giving graders ground truth about which
failures were ours vs. the agent's own doing.
"""
from __future__ import annotations

import json
from typing import Optional

from .schemas import FaultMode, FaultSpec
from .tools import ToolRegistry, ToolResult

_PAYLOADS = {
    FaultMode.ERROR: ("HTTP 503 Service Unavailable: upstream service failed to respond", True),
    FaultMode.TIMEOUT: ("TimeoutError: request to internal service timed out after 10000ms", True),
    FaultMode.RATE_LIMIT: ("HTTP 429 Too Many Requests: rate limit exceeded, slow down", True),
}
_GARBAGE = "xK9#▓▓GARBLED-RESPONSE▓▓<<0xF3 0x00 0x9C>>corrupted stream##"


class FaultingToolExecutor:
    """Wraps a ToolRegistry; intercepts calls that match an active FaultSpec."""

    def __init__(self, registry: ToolRegistry, faults: list[FaultSpec]) -> None:
        self.registry = registry
        self.faults = faults
        self.call_counts: dict[str, int] = {}

    def schemas(self):
        return self.registry.schemas()

    def execute(self, name: str, args: dict) -> tuple[ToolResult, bool]:
        """Returns (result, faulted). `faulted` means the harness injected it."""
        count = self.call_counts.get(name, 0) + 1
        self.call_counts[name] = count

        for spec in self.faults:
            if spec.tool != name:
                continue
            if spec.always or count <= spec.times:
                return self._inject(spec), True
        return self.registry.execute(name, args), False

    def _inject(self, spec: FaultSpec) -> ToolResult:
        if spec.mode == FaultMode.GARBAGE:
            # Not an error at the transport level — the tool "succeeded" but
            # returned junk. The agent has to notice that on its own.
            return ToolResult(spec.message or _GARBAGE, False)
        if spec.mode == FaultMode.EMPTY:
            payload = {"results": [], "note": spec.message or "no records matched"}
            return ToolResult(json.dumps(payload), False)
        message, is_error = _PAYLOADS[spec.mode]
        return ToolResult(json.dumps({"error": spec.message or message}), is_error)
