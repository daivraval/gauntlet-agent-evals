"""The runner: executes scenarios against an agent and grades trajectories.

Each scenario gets a fresh world (isolation), its own fault schedule, and a
hard timeout. Scenarios run concurrently under a semaphore; every step the
agent takes is recorded, graded, and persisted to the run directory:

    reports/run_<timestamp>_<agent>/
        results.json        full RunReport (grades + trajectories)
        trajectories.jsonl  one trajectory per line, grep-friendly
        report.html         self-contained visual report
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from . import graders
from .agents import Agent, build_agent
from .config import Config
from .faults import FaultingToolExecutor
from .judge import Judge
from .llm import LLMClient
from .schemas import RunReport, Scenario, ScenarioResult, TrajectoryRecorder
from .tools import ToolRegistry
from .world import World

console = Console()


async def _run_one(
    scenario: Scenario,
    trial: int,
    agent: Agent,
    config: Config,
    judge: Judge | None,
) -> ScenarioResult:
    world = World(scenario.world_overrides)
    registry = ToolRegistry(world, None if scenario.tools == "all" else list(scenario.tools))
    executor = FaultingToolExecutor(registry, scenario.faults)
    recorder = TrajectoryRecorder(scenario.id, agent.name, getattr(config, "agent_model", ""))

    start = time.perf_counter()
    try:
        await asyncio.wait_for(
            agent.run(scenario, executor, recorder),
            timeout=config.scenario_timeout_s,
        )
    except asyncio.TimeoutError:
        recorder.fail(f"scenario timed out after {config.scenario_timeout_s:.0f}s")
    except Exception as exc:  # the agent crashing is a result, not a harness error
        recorder.fail(f"agent crashed: {type(exc).__name__}: {exc}")
    duration = time.perf_counter() - start

    trajectory = recorder.trajectory
    grades = graders.run_deterministic(scenario, trajectory, world)
    if judge is not None and graders.needs_judge(scenario):
        judge_score, reasoning = await judge.grade(scenario, trajectory)
        grades.append(graders.judge_grade_result(scenario, judge_score, reasoning))

    score, passed, hard_fails = graders.compose(grades, config.pass_threshold)
    prompt_tokens, completion_tokens = trajectory.total_tokens
    return ScenarioResult(
        scenario_id=scenario.id,
        name=scenario.name,
        category=scenario.category,
        agent=agent.name,
        trial=trial,
        grades=grades,
        score=score,
        passed=passed,
        hard_fails=hard_fails,
        duration_s=round(duration, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(config.cost(prompt_tokens, completion_tokens), 6),
        trajectory=trajectory,
    )


async def run_suite(
    scenarios: list[Scenario],
    agent_name: str,
    config: Config,
    judge_enabled: bool = True,
    trials: int = 1,
    out_root: Path = Path("reports"),
    quiet: bool = False,
) -> tuple[RunReport, Path]:
    agent = build_agent(agent_name, config)

    judge: Judge | None = None
    if judge_enabled:
        config.require_api_key()
        judge = Judge(LLMClient(config), config, out_root / ".judge_cache.json")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{agent_name}"
    report = RunReport(
        run_id=run_id,
        started_at=dt.datetime.now().isoformat(timespec="seconds"),
        agent=agent_name,
        model=config.agent_model if agent_name != "scripted" else "(none)",
        judge_model=config.judge_model if judge else "(disabled)",
        judge_enabled=judge is not None,
        pass_threshold=config.pass_threshold,
        trials=trials,
    )

    semaphore = asyncio.Semaphore(config.concurrency)
    total = len(scenarios) * trials
    done = 0

    async def _guarded(scenario: Scenario, trial: int) -> ScenarioResult:
        nonlocal done
        async with semaphore:
            result = await _run_one(scenario, trial, agent, config, judge)
        done += 1
        if not quiet:
            mark = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            hard = " [bold red](hard fail)[/bold red]" if result.hard_fails else ""
            console.print(
                f"[dim]{done:>3}/{total}[/dim] {mark} {result.scenario_id:<6} "
                f"score={result.score:.2f}{hard}  [dim]{escape(result.name)}[/dim]"
            )
        return result

    wall_start = time.perf_counter()
    tasks = [
        _guarded(scenario, trial)
        for scenario in scenarios
        for trial in range(1, trials + 1)
    ]
    results = await asyncio.gather(*tasks)
    report.wall_time_s = round(time.perf_counter() - wall_start, 2)
    report.results = sorted(results, key=lambda r: (r.scenario_id, r.trial))

    if judge is not None:
        judge.save_cache()

    out_dir = _persist(report, out_root)
    return report, out_dir


def _persist(report: RunReport, out_root: Path) -> Path:
    out_dir = out_root / f"run_{report.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as fh:
        for result in report.results:
            if result.trajectory is not None:
                fh.write(result.trajectory.model_dump_json() + "\n")

    from .report import to_html  # local import to keep runner importable without report deps

    (out_dir / "report.html").write_text(to_html(report), encoding="utf-8")
    return out_dir
