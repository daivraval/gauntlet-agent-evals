"""End-to-end pipeline test: scenarios -> scripted agent -> faults -> graders
-> artifacts, fully offline (no API key, no network)."""
import asyncio
import json

from gauntlet.config import Config
from gauntlet.loader import load_scenarios
from gauntlet.report import aggregate, to_html
from gauntlet.runner import run_suite


def run(tmp_path, ids, trials=1):
    scenarios = load_scenarios(ids=ids)
    config = Config()
    config.concurrency = 2
    return asyncio.run(run_suite(
        scenarios, agent_name="scripted", config=config,
        judge_enabled=False, trials=trials, out_root=tmp_path, quiet=True,
    ))


def test_scripted_run_produces_results_and_artifacts(tmp_path):
    report, out_dir = run(tmp_path, ids=["TS-01", "AR-01", "ER-01"])
    assert len(report.results) == 3
    assert (out_dir / "results.json").exists()
    assert (out_dir / "trajectories.jsonl").exists()
    assert (out_dir / "report.html").exists()

    saved = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["agent"] == "scripted" and len(saved["results"]) == 3

    lines = (out_dir / "trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 and all(json.loads(line)["steps"] for line in lines)


def test_scripted_agent_recovers_through_injected_fault(tmp_path):
    report, _ = run(tmp_path, ids=["ER-01"])
    result = report.results[0]
    recovery = next(g for g in result.grades if g.grader == "recovery")
    assert recovery.score == 1.0, recovery.details
    tool_steps = [s for s in result.trajectory.steps if s.kind == "tool"]
    assert tool_steps[0].faulted and not tool_steps[-1].faulted


def test_trials_multiply_results(tmp_path):
    report, _ = run(tmp_path, ids=["TS-01"], trials=3)
    assert len(report.results) == 3
    assert [r.trial for r in report.results] == [1, 2, 3]


def test_aggregate_and_html_render(tmp_path):
    report, _ = run(tmp_path, ids=["TS-01", "HA-01"])
    agg = aggregate(report)
    assert agg["n"] == 2 and 0.0 <= agg["overall_score"] <= 1.0
    html = to_html(report)
    assert "GAUNTLET" in html and "TS-01" in html
