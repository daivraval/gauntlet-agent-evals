"""GAUNTLET CLI — run, inspect, and compare agent evaluations.

    python run_evals.py run --agent scripted --no-judge     # offline demo, no API key
    python run_evals.py run --agent baseline                # evaluate the naive agent
    python run_evals.py run --agent hardened                # evaluate the hardened agent
    python run_evals.py run --category error_recovery --id ER-01
    python run_evals.py list
    python run_evals.py show reports/run_... --id HA-03
    python run_evals.py compare reports/run_A reports/run_B --fail-on-regression 0.05
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from gauntlet.config import Config
from gauntlet.loader import load_scenarios
from gauntlet.report import compare, load_report, print_summary, print_trajectory
from gauntlet.runner import run_suite

console = Console()


def cmd_run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(categories=args.category, ids=args.id)
    config = Config()
    if args.concurrency:
        config.concurrency = args.concurrency
    if args.threshold:
        config.pass_threshold = args.threshold

    console.print(f"[cyan]Running {len(scenarios)} scenario(s) × {args.trials} trial(s) "
                  f"against agent [bold]{args.agent}[/bold] "
                  f"(judge: {'off' if args.no_judge else 'on'})[/cyan]\n")

    report, out_dir = asyncio.run(run_suite(
        scenarios,
        agent_name=args.agent,
        config=config,
        judge_enabled=not args.no_judge,
        trials=args.trials,
        out_root=Path(args.out),
        quiet=args.quiet,
    ))
    console.print()
    print_summary(report)
    console.print(f"\n[dim]Artifacts:[/dim] {out_dir}\\results.json · trajectories.jsonl · report.html")

    if args.fail_under is not None:
        from gauntlet.report import aggregate
        overall = aggregate(report)["overall_score"]
        if overall < args.fail_under:
            console.print(f"[bold red]GATE FAILED: overall score {overall:.3f} "
                          f"< --fail-under {args.fail_under}[/bold red]")
            return 2
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(categories=args.category)
    table = Table(title=f"{len(scenarios)} scenarios", header_style="bold cyan")
    for col in ("ID", "Category", "Name", "Faults", "Judge"):
        table.add_column(col)
    for s in scenarios:
        faults = ", ".join(f"{f.tool}:{f.mode.value}" + ("(always)" if f.always else f"×{f.times}")
                           for f in s.faults) or "-"
        table.add_row(s.id, s.category.value, s.name, faults,
                      "yes" if s.expected.judge_rubric else "-")
    console.print(table)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    report = load_report(Path(args.run_dir))
    matches = [r for r in report.results if r.scenario_id.upper() == args.id.upper()]
    if not matches:
        console.print(f"[red]scenario {args.id} not found in {args.run_dir}[/red]")
        return 1
    for result in matches:
        print_trajectory(result)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    delta = compare(load_report(Path(args.run_a)), load_report(Path(args.run_b)))
    if args.fail_on_regression is not None and delta < -args.fail_on_regression:
        console.print(f"[bold red]REGRESSION GATE FAILED: overall score dropped "
                      f"{-delta:.3f} (> {args.fail_on_regression})[/bold red]")
        return 3
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="gauntlet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the eval suite")
    p_run.add_argument("--agent", default="baseline", help="baseline | hardened | scripted")
    p_run.add_argument("--category", action="append", help="filter by category (repeatable)")
    p_run.add_argument("--id", action="append", help="filter by scenario id (repeatable)")
    p_run.add_argument("--trials", type=int, default=1, help="runs per scenario")
    p_run.add_argument("--concurrency", type=int, help="parallel scenarios (default 4)")
    p_run.add_argument("--threshold", type=float, help="pass threshold (default 0.7)")
    p_run.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge grading")
    p_run.add_argument("--out", default="reports", help="output root directory")
    p_run.add_argument("--quiet", action="store_true", help="no per-scenario progress lines")
    p_run.add_argument("--fail-under", type=float, help="exit 2 if overall score below this (CI gate)")
    p_run.set_defaults(fn=cmd_run)

    p_list = sub.add_parser("list", help="list scenarios")
    p_list.add_argument("--category", action="append")
    p_list.set_defaults(fn=cmd_list)

    p_show = sub.add_parser("show", help="inspect one scenario's trajectory from a run")
    p_show.add_argument("run_dir")
    p_show.add_argument("--id", required=True)
    p_show.set_defaults(fn=cmd_show)

    p_cmp = sub.add_parser("compare", help="diff two runs (A = before, B = after)")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--fail-on-regression", type=float,
                       help="exit 3 if overall score drops more than this")
    p_cmp.set_defaults(fn=cmd_compare)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
