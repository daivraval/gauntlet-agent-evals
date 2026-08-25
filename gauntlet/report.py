"""Reporting: console scorecard, self-contained HTML report, run comparison.

The same RunReport JSON drives everything, so a run can be re-rendered,
diffed, or gated in CI long after it finished.
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .schemas import RunReport, ScenarioResult

console = Console()


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(report: RunReport) -> dict:
    results = report.results
    by_cat: dict[str, list[ScenarioResult]] = defaultdict(list)
    by_grader: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r.category.value].append(r)
        for g in r.grades:
            by_grader[g.grader].append(g.score)

    return {
        "n": len(results),
        "overall_score": round(mean([r.score for r in results]), 4) if results else 0.0,
        "pass_rate": round(sum(r.passed for r in results) / len(results), 4) if results else 0.0,
        "hard_fails": sum(bool(r.hard_fails) for r in results),
        "categories": {
            cat: {
                "n": len(rs),
                "avg_score": round(mean([r.score for r in rs]), 4),
                "pass_rate": round(sum(r.passed for r in rs) / len(rs), 4),
                "hard_fails": sum(bool(r.hard_fails) for r in rs),
            }
            for cat, rs in sorted(by_cat.items())
        },
        "graders": {g: round(mean(scores), 4) for g, scores in sorted(by_grader.items())},
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
        "cost_usd": round(sum(r.cost_usd for r in results), 4),
        "avg_duration_s": round(mean([r.duration_s for r in results]), 2) if results else 0.0,
    }


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------

def print_summary(report: RunReport) -> None:
    agg = aggregate(report)
    grade_color = "green" if agg["pass_rate"] >= 0.8 else ("yellow" if agg["pass_rate"] >= 0.5 else "red")

    lines = [
        f"[bold]Agent:[/bold] {report.agent}   [bold]Model:[/bold] {report.model}   "
        f"[bold]Judge:[/bold] {report.judge_model}",
        f"[bold]Scenarios:[/bold] {agg['n']}   [bold]Trials:[/bold] {report.trials}   "
        f"[bold]Wall time:[/bold] {report.wall_time_s:.1f}s",
        "",
        f"[bold {grade_color}]Overall score : {agg['overall_score']:.3f}[/bold {grade_color}]",
        f"[bold {grade_color}]Pass rate     : {agg['pass_rate']:.0%}[/bold {grade_color}]"
        f"   (threshold {report.pass_threshold})",
        f"[bold red]Hard failures : {agg['hard_fails']}[/bold red]",
        f"Tokens: {agg['prompt_tokens']:,} in / {agg['completion_tokens']:,} out"
        + (f"   Est. cost: ${agg['cost_usd']:.4f}" if agg["cost_usd"] else ""),
    ]
    console.print(Panel("\n".join(lines), title="⚔ GAUNTLET RESULTS", border_style="cyan"))

    cat_table = Table(title="By category", header_style="bold cyan")
    for col in ("Category", "N", "Avg score", "Pass rate", ""):
        cat_table.add_column(col)
    for cat, stats in agg["categories"].items():
        cat_table.add_row(
            cat, str(stats["n"]), f"{stats['avg_score']:.3f}",
            f"{stats['pass_rate']:.0%}", _bar(stats["avg_score"]),
        )
    console.print(cat_table)

    grader_table = Table(title="By grader (where it loses points)", header_style="bold cyan")
    for col in ("Grader", "Avg score", ""):
        grader_table.add_column(col)
    for grader, score in sorted(agg["graders"].items(), key=lambda kv: kv[1]):
        grader_table.add_row(grader, f"{score:.3f}", _bar(score))
    console.print(grader_table)

    failures = [r for r in report.results if not r.passed]
    if failures:
        console.print(f"\n[bold red]{len(failures)} failed scenario(s):[/bold red]")
        for r in failures:
            hard = f" [bold red]HARD FAIL: {escape(', '.join(r.hard_fails))}[/bold red]" if r.hard_fails else ""
            console.print(f"\n[red]✗[/red] [bold]{r.scenario_id}[/bold] {escape(r.name)} "
                          f"[dim]({r.category.value})[/dim] score={r.score:.2f}{hard}")
            for g in r.grades:
                if g.score < 1.0 or g.hard_fail:
                    console.print(f"    [yellow]{g.grader}[/yellow]={g.score:.2f}  [dim]{escape(g.details)}[/dim]")
            if r.trajectory is not None:
                if r.trajectory.error:
                    console.print(f"    [red]agent error:[/red] {escape(r.trajectory.error)}")
                answer = (r.trajectory.final_answer or "").replace("\n", " ")[:220]
                if answer:
                    console.print(f"    [dim]answer: {escape(answer)}[/dim]")
    else:
        console.print("\n[bold green]All scenarios passed.[/bold green]")


def print_trajectory(result: ScenarioResult) -> None:
    console.print(Panel(
        f"[bold]{result.scenario_id}[/bold] {result.name}\n"
        f"category={result.category.value}  agent={result.agent}  "
        f"score={result.score:.2f}  passed={result.passed}",
        border_style="cyan",
    ))
    traj = result.trajectory
    if traj is None:
        console.print("[dim](trajectory not stored)[/dim]")
        return
    table = Table(header_style="bold cyan")
    for col in ("#", "kind", "tool", "detail", "flags"):
        table.add_column(col, overflow="fold")
    for s in traj.steps:
        if s.kind == "llm":
            detail = (s.assistant_text or "")[:120] or f"requested {len(s.tool_calls_requested)} tool call(s)"
        elif s.kind == "tool":
            detail = f"args={json.dumps(s.arguments or {})[:80]} -> {(s.result or '')[:120]}"
        else:
            detail = (s.result or "")[:200]
        flags = " ".join(filter(None, [
            "[red]error[/red]" if s.is_error else "",
            "[magenta]INJECTED FAULT[/magenta]" if s.faulted else "",
        ]))
        table.add_row(str(s.index), s.kind, s.tool or "", escape(detail), flags)
    console.print(table)
    for g in result.grades:
        mark = "[red]✗[/red]" if g.hard_fail else ("[green]✓[/green]" if g.passed else "[yellow]~[/yellow]")
        console.print(f" {mark} {g.grader}: {g.score:.2f} (w={g.weight})  [dim]{escape(g.details)}[/dim]")


# ---------------------------------------------------------------------------
# comparison / regression gate
# ---------------------------------------------------------------------------

def load_report(run_dir: Path) -> RunReport:
    path = run_dir / "results.json" if run_dir.is_dir() else run_dir
    return RunReport.model_validate_json(path.read_text(encoding="utf-8"))


def compare(a: RunReport, b: RunReport) -> float:
    """Prints A vs B; returns the overall score delta (b - a)."""
    agg_a, agg_b = aggregate(a), aggregate(b)
    table = Table(title=f"Comparison — A: {a.run_id} ({a.agent})  vs  B: {b.run_id} ({b.agent})",
                  header_style="bold cyan")
    for col in ("Metric", "A", "B", "Δ"):
        table.add_column(col)

    def row(label: str, va: float, vb: float, pct: bool = False) -> None:
        delta = vb - va
        color = "green" if delta > 0.0001 else ("red" if delta < -0.0001 else "dim")
        fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.3f}")
        table.add_row(label, fmt(va), fmt(vb), f"[{color}]{delta:+.3f}[/{color}]")

    row("overall score", agg_a["overall_score"], agg_b["overall_score"])
    row("pass rate", agg_a["pass_rate"], agg_b["pass_rate"], pct=True)
    for cat in sorted(set(agg_a["categories"]) | set(agg_b["categories"])):
        va = agg_a["categories"].get(cat, {}).get("avg_score", 0.0)
        vb = agg_b["categories"].get(cat, {}).get("avg_score", 0.0)
        row(f"  {cat}", va, vb)
    console.print(table)

    status_a = {r.scenario_id: r.passed for r in a.results}
    regressions = [r.scenario_id for r in b.results
                   if not r.passed and status_a.get(r.scenario_id, False)]
    if regressions:
        console.print(f"[bold red]Regressions (passed in A, failed in B):[/bold red] "
                      f"{', '.join(sorted(set(regressions)))}")
    improvements = [r.scenario_id for r in b.results
                    if r.passed and status_a.get(r.scenario_id) is False]
    if improvements:
        console.print(f"[bold green]Newly passing in B:[/bold green] "
                      f"{', '.join(sorted(set(improvements)))}")
    return agg_b["overall_score"] - agg_a["overall_score"]


# ---------------------------------------------------------------------------
# HTML report (single self-contained file)
# ---------------------------------------------------------------------------

def to_html(report: RunReport) -> str:
    agg = aggregate(report)
    esc = html.escape

    cat_rows = "".join(
        f"""<div class="cat"><span class="cat-name">{esc(cat)}</span>
        <div class="track"><div class="fill" style="width:{stats['avg_score'] * 100:.0f}%"></div></div>
        <span class="cat-num">{stats['avg_score']:.2f} · {stats['pass_rate']:.0%} pass · n={stats['n']}</span></div>"""
        for cat, stats in agg["categories"].items()
    )

    def result_block(r: ScenarioResult) -> str:
        chip = ('<span class="chip pass">PASS</span>' if r.passed
                else '<span class="chip fail">FAIL</span>')
        hard = (f'<span class="chip hard">HARD: {esc(", ".join(r.hard_fails))}</span>'
                if r.hard_fails else "")
        grade_rows = "".join(
            f"<tr><td>{esc(g.grader)}</td><td>{g.score:.2f}</td><td>{g.weight}</td>"
            f"<td class='dim'>{esc(g.details)}</td></tr>"
            for g in r.grades
        )
        tool_lines = ""
        if r.trajectory:
            for s in r.trajectory.tool_steps:
                flags = ("<b class='inj'>⚡injected fault</b> " if s.faulted else "") + \
                        ("<b class='errf'>error</b> " if s.is_error else "")
                tool_lines += (f"<div class='tl'>{flags}<code>{esc(s.tool or '')}"
                               f"({esc(json.dumps(s.arguments or {})[:100])})</code> → "
                               f"<span class='dim'>{esc((s.result or '')[:180])}</span></div>")
            answer = esc((r.trajectory.final_answer or "")[:600])
            err = (f"<div class='tl errf'>agent error: {esc(r.trajectory.error)}</div>"
                   if r.trajectory.error else "")
        else:
            answer, err = "", ""
        return f"""<details class="{'ok' if r.passed else 'bad'}">
<summary><b>{esc(r.scenario_id)}</b> {esc(r.name)} <span class="dim">({esc(r.category.value)})</span>
 — {r.score:.2f} {chip}{hard}</summary>
<table><tr><th>grader</th><th>score</th><th>w</th><th>details</th></tr>{grade_rows}</table>
<h4>Tool calls</h4>{tool_lines or "<div class='tl dim'>(none)</div>"}{err}
<h4>Final answer</h4><div class="ans">{answer or "<span class='dim'>(none)</span>"}</div>
</details>"""

    blocks = "".join(result_block(r) for r in report.results)
    cost = f"<div class='stat'><div class='k'>${agg['cost_usd']:.4f}</div><div>est. cost</div></div>" \
        if agg["cost_usd"] else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>GAUNTLET · {esc(report.run_id)}</title><style>
:root {{ color-scheme: dark; }}
body {{ background:#0b0e14; color:#dbe2ef; font:15px/1.5 'Segoe UI',system-ui,sans-serif;
       max-width:1000px; margin:2rem auto; padding:0 1rem; }}
h1 {{ background:linear-gradient(90deg,#7dd3fc,#c084fc,#f472b6); -webkit-background-clip:text;
     background-clip:text; color:transparent; letter-spacing:.12em; }}
.meta {{ color:#8b93a7; margin-bottom:1.5rem; }}
.stats {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.2rem 0; }}
.stat {{ background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
        border-radius:12px; padding:.9rem 1.4rem; text-align:center; }}
.stat .k {{ font-size:1.7rem; font-weight:700; color:#7dd3fc; }}
.cat {{ display:flex; align-items:center; gap:.8rem; margin:.35rem 0; }}
.cat-name {{ width:150px; color:#c9d3e8; }}
.cat-num {{ color:#8b93a7; font-size:.85rem; }}
.track {{ flex:1; height:10px; background:rgba(255,255,255,.06); border-radius:6px; overflow:hidden; }}
.fill {{ height:100%; background:linear-gradient(90deg,#38bdf8,#a78bfa); }}
details {{ background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
          border-radius:10px; margin:.5rem 0; padding:.55rem .9rem; }}
details.bad {{ border-left:3px solid #f87171; }}
details.ok {{ border-left:3px solid #34d399; }}
summary {{ cursor:pointer; }}
.chip {{ font-size:.7rem; padding:.1rem .5rem; border-radius:99px; margin-left:.5rem; font-weight:700; }}
.chip.pass {{ background:#052e1f; color:#34d399; }}
.chip.fail {{ background:#3f1120; color:#f87171; }}
.chip.hard {{ background:#450a0a; color:#fca5a5; }}
table {{ border-collapse:collapse; margin:.8rem 0; width:100%; font-size:.85rem; }}
th,td {{ text-align:left; padding:.25rem .6rem; border-bottom:1px solid rgba(255,255,255,.06); }}
.dim {{ color:#8b93a7; }} .inj {{ color:#c084fc; }} .errf {{ color:#f87171; }}
.tl {{ font-size:.83rem; margin:.2rem 0; }} code {{ color:#7dd3fc; }}
.ans {{ background:rgba(255,255,255,.04); border-radius:8px; padding:.6rem .8rem;
       white-space:pre-wrap; font-size:.9rem; }}
h4 {{ margin:.8rem 0 .3rem; color:#c9d3e8; }}
</style></head><body>
<h1>⚔ GAUNTLET</h1>
<div class="meta">run <b>{esc(report.run_id)}</b> · agent <b>{esc(report.agent)}</b> ·
model <b>{esc(report.model)}</b> · judge <b>{esc(report.judge_model)}</b> ·
{esc(report.started_at)} · {report.wall_time_s:.1f}s</div>
<div class="stats">
<div class="stat"><div class="k">{agg['overall_score']:.3f}</div><div>overall score</div></div>
<div class="stat"><div class="k">{agg['pass_rate']:.0%}</div><div>pass rate</div></div>
<div class="stat"><div class="k">{agg['hard_fails']}</div><div>hard failures</div></div>
<div class="stat"><div class="k">{agg['n']}</div><div>scenarios</div></div>
<div class="stat"><div class="k">{agg['prompt_tokens'] + agg['completion_tokens']:,}</div><div>tokens</div></div>
{cost}</div>
<h3>Categories</h3>{cat_rows}
<h3>Scenarios</h3>{blocks}
</body></html>"""
