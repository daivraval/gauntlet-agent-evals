"""Scenario loading + validation.

Scenarios live in scenarios/*.yaml, one file per category. Validation is
strict and runs before anything is executed: broken expectations are a bug
in the eval, and a silently-broken eval is worse than no eval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .schemas import Scenario
from .tools import TOOLS

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def load_scenarios(
    directory: Optional[Path] = None,
    categories: Optional[list[str]] = None,
    ids: Optional[list[str]] = None,
) -> list[Scenario]:
    directory = directory or SCENARIOS_DIR
    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not files:
        raise SystemExit(f"no scenario files found in {directory}")

    scenarios: list[Scenario] = []
    for file in files:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        for raw in data.get("scenarios", []):
            try:
                scenarios.append(Scenario(**raw))
            except Exception as exc:  # pydantic ValidationError, with file context
                raise SystemExit(f"invalid scenario in {file.name} "
                                 f"(id={raw.get('id', '?')}): {exc}") from exc

    _validate(scenarios)

    if categories:
        wanted = {c.lower() for c in categories}
        scenarios = [s for s in scenarios if s.category.value in wanted]
    if ids:
        wanted_ids = {i.upper() for i in ids}
        scenarios = [s for s in scenarios if s.id.upper() in wanted_ids]
    if not scenarios:
        raise SystemExit("no scenarios match the given filters")
    return scenarios


def _validate(scenarios: list[Scenario]) -> None:
    problems: list[str] = []

    seen: set[str] = set()
    for s in scenarios:
        if s.id in seen:
            problems.append(f"{s.id}: duplicate scenario id")
        seen.add(s.id)

        offered = set(TOOLS) if s.tools == "all" else set(s.tools)
        unknown_offered = offered - set(TOOLS)
        if unknown_offered:
            problems.append(f"{s.id}: offers unknown tools {sorted(unknown_offered)}")

        exp = s.expected
        for label, names in [
            ("required_tools", exp.required_tools),
            ("forbidden_tools", exp.forbidden_tools),
            ("allowed_tools", exp.allowed_tools or []),
            ("required_args tools", list(exp.required_args)),
            ("min_calls_to tools", list(exp.min_calls_to)),
        ]:
            unknown = set(names) - set(TOOLS)
            if unknown:
                problems.append(f"{s.id}: {label} references unknown tools {sorted(unknown)}")

        missing_offer = set(exp.required_tools) - offered
        if missing_offer:
            problems.append(f"{s.id}: requires tools the agent is never offered: {sorted(missing_offer)}")

        for fault in s.faults:
            if fault.tool not in TOOLS:
                problems.append(f"{s.id}: fault targets unknown tool {fault.tool!r}")

        if exp.expect_retry_of and not any(f.tool == exp.expect_retry_of for f in s.faults):
            problems.append(f"{s.id}: expect_retry_of={exp.expect_retry_of!r} but no fault targets it")

    if problems:
        raise SystemExit("scenario validation failed:\n  - " + "\n  - ".join(problems))
