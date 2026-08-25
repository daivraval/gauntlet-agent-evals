"""Reporting: console scorecard, self-contained HTML report, run comparison.

The same RunReport JSON drives everything, so a run can be re-rendered,
diffed, or gated in CI long after it finished.
"""
from __future__ import annotations

import base64
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

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Sorted scenario ids, matching the row-major tile order of the grid sheet
# (docs/prep_grid.py cuts tile N for the N-th id in this list).
_GRID_ORDER = (
    [f"AD-{i:02d}" for i in range(1, 6)] + [f"AR-{i:02d}" for i in range(1, 9)]
    + [f"ER-{i:02d}" for i in range(1, 11)] + [f"HA-{i:02d}" for i in range(1, 11)]
    + [f"MS-{i:02d}" for i in range(1, 8)] + [f"TS-{i:02d}" for i in range(1, 11)]
)
_GRID_INDEX = {sid: i for i, sid in enumerate(_GRID_ORDER)}


def _grid_tile_count() -> int:
    return len(list((_ASSETS_DIR / "grid").glob("grid_*.jpg"))) if _ASSETS_DIR.exists() else 0

# Tiny repeating noise swatch layered over every grid tile (print grain).
_NOISE_URI = (
    "data:image/svg+xml;utf8,<svg%20xmlns='http://www.w3.org/2000/svg'%20width='160'%20height='160'>"
    "<filter%20id='gn'><feTurbulence%20type='fractalNoise'%20baseFrequency='0.85'%20numOctaves='3'/>"
    "<feColorMatrix%20type='saturate'%20values='0'/></filter>"
    "<rect%20width='160'%20height='160'%20filter='url(%23gn)'%20opacity='0.42'/></svg>"
)


def _asset_uri(rel: str) -> str:
    """Inline one image asset as a data URI, keeping the report a single
    self-contained file. Missing files simply skip that element."""
    path = _ASSETS_DIR / rel
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def to_html(report: RunReport) -> str:
    agg = aggregate(report)
    esc = html.escape

    cat_rows = "".join(
        f"""<div class="cat"><span class="cat-name">{esc(cat.replace('_', ' '))}</span>
        <span class="cat-score">{stats['avg_score']:.2f}</span>
        <span class="cat-num">{stats['pass_rate']:.0%} pass · {stats['n']} trials</span></div>"""
        for cat, stats in agg["categories"].items()
    )

    def result_block(r: ScenarioResult) -> str:
        chip = ('<span class="chip pass">pass</span>' if r.passed
                else '<span class="chip fail">fail</span>')
        hard = (f'<span class="chip hard">hard · {esc(", ".join(r.hard_fails))}</span>'
                if r.hard_fails else "")
        grade_rows = "".join(
            f"<tr><td>{esc(g.grader)}</td><td>{g.score:.2f}</td><td>{g.weight}</td>"
            f"<td class='dim'>{esc(g.details)}</td></tr>"
            for g in r.grades
        )
        tool_lines = ""
        if r.trajectory:
            for s in r.trajectory.tool_steps:
                flags = ("<b class='inj'>sabotaged</b> " if s.faulted else "") + \
                        ("<b class='errf'>error</b> " if s.is_error else "")
                tool_lines += (f"<div class='tl'>{flags}<code>{esc(s.tool or '')}"
                               f"({esc(json.dumps(s.arguments or {})[:100])})</code> → "
                               f"<span class='dim'>{esc((s.result or '')[:180])}</span></div>")
            answer = esc((r.trajectory.final_answer or "")[:600])
            err = (f"<div class='tl errf'>agent error: {esc(r.trajectory.error)}</div>"
                   if r.trajectory.error else "")
        else:
            answer, err = "", ""
        # Sheets may hold fewer tiles than scenarios; cycle through what we have.
        order_idx = _GRID_INDEX.get(r.scenario_id)
        tile_count = _grid_tile_count()
        tile_uri = (_asset_uri(f"grid/grid_{order_idx % tile_count:02d}.jpg")
                    if order_idx is not None and tile_count else "")
        img_div = (f"""<div class="gimg" style="background-image:url(&quot;{_NOISE_URI}&quot;), """
                   f"""url(&quot;{tile_uri}&quot;)"></div>""" if tile_uri else "")
        return f"""<details class="{'ok' if r.passed else 'bad'}">
<summary>
<div class="tkick">{esc(r.scenario_id)} · {esc(r.category.value.replace('_', ' '))}</div>
<div class="tname">{esc(r.name)}</div>
{img_div}
<div class="tmeta">score {r.score:.2f} {chip}{hard}</div>
</summary>
<table><tr><th>grader</th><th>score</th><th>w</th><th>details</th></tr>{grade_rows}</table>
<h4>Tool calls</h4>{tool_lines or "<div class='tl dim'>(none)</div>"}{err}
<h4>Final answer</h4><div class="ans">{answer or "<span class='dim'>(none)</span>"}</div>
</details>"""

    # The grid runs as alternating blocks of two rows: one hugging the left,
    # the next hugging the right. A figure fills the gap each block leaves,
    # stretched to that space and tucked under the nearest tile.
    # The opening block takes the warrior from the same source and in the same
    # blue as the pair beside the wordmark, run off the right edge of the page.
    # It is the untouched crop, not the painted one, because that version's
    # ground is true black and so dissolves completely under screen blending.
    edge_art = {0: (_asset_uri("figures/fig3.jpg"), "g-edge g-right"),
                1: (_asset_uri("figures/caduceus.jpg"), "g-edge g-turn")}
    gap_art = [(uri, cls) for asset, cls in (
        ("painted/gap3_torch.jpg", "g-torch"),
        ("painted/gap4_clouds.jpg", ""),
        ("painted/gap5_moonwing.jpg", ""),
        ("painted/gap6_pegasus.jpg", ""),
        ("painted/gap7_worlds.jpg", ""),
    ) if (uri := _asset_uri(asset))]
    group_size = 6
    groups = []
    for index, start in enumerate(range(0, len(report.results), group_size)):
        tiles = "".join(result_block(r) for r in report.results[start:start + group_size])
        side = "left" if index % 2 == 0 else "right"
        if index in edge_art and edge_art[index][0]:
            uri, cls = edge_art[index]
            art = f'<div class="gfig {cls}"><img src="{uri}" alt=""></div>'
        elif gap_art:
            uri, cls = gap_art[(index - len(edge_art)) % len(gap_art)]
            art = f'<div class="gfig {cls}"><img src="{uri}" alt=""></div>'
        else:
            art = ""
        groups.append(f'<div class="tgroup {side}"><div class="tgrid">{tiles}</div>{art}</div>')
    blocks = "".join(groups)
    cost = f"<div class='stat'><div class='k'>${agg['cost_usd']:.4f}</div><div>est. cost</div></div>" \
        if agg["cost_usd"] else ""

    # (asset, css class) — the warrior crop (fig3) is deliberately unplaced;
    # the reaching hands took over the foot of the page.
    backdrop = [
        ("figures/fig1.jpg", "f-bust"),
        ("figures/damn.jpg", "f-flame"),
        ("figures/hands.jpg", "f-hands"),
    ]
    fig_layers = "".join(
        f'<div class="fig {cls}"><img src="{uri}" alt=""></div>'
        for asset, cls in backdrop
        if (uri := _asset_uri(asset))
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAUNTLET · {esc(report.run_id)}</title><style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Courier+Prime:ital,wght@0,400;0,700;1,400&display=swap');
:root {{ color-scheme: dark;
  --red-bright:#c22626; --red-hi:#a81d1d; --red:#8d1a1a; --red-mid:#761616;
  --red-dim:#5a1111; --red-faint:#2e0a0a; --ink:#070404;
  /* Every piece of artwork is red in the source files. This one rotation
     recolours all of it; set it to 0deg to go back to red. */
  --art-hue:203deg; }}
.fig img, .gfig img, .gimg {{ filter:hue-rotate(var(--art-hue)); }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--ink); color:var(--red); position:relative;
  font:15px/1.7 'Courier Prime', 'Courier New', monospace; }}
.layer {{ position:absolute; inset:0; pointer-events:none; }}
.layer .hold {{ position:sticky; top:0; height:100vh; overflow:hidden; }}
.layer svg {{ width:100%; height:100vh; display:block; }}
.grain {{ z-index:-1; }}
.grain .hold {{ opacity:.16; }}
/* Background figures sit in staggered adjacent pairs. The art is pre-cleared
   to true black, and `screen` blending drops that black out entirely, so no
   panel edge or halo survives against the clay-black page. */
.fig {{ position:absolute; z-index:-2; pointer-events:none; mix-blend-mode:screen;
  -webkit-mask-image:radial-gradient(ellipse 60% 58% at 50% 50%, #000 52%, transparent 84%);
  mask-image:radial-gradient(ellipse 60% 58% at 50% 50%, #000 52%, transparent 84%); }}
.fig img {{ width:100%; height:auto; display:block; opacity:.34; }}
/* The wordmark is centred and its size is capped, so anchoring these two off
   the page centre keeps them locked to specific letters: the bust's right
   edge meets the G, the bound figure's left edge meets the closing T. */
.f-bust {{ top:0; right:calc(50% + 389px); width:302px; }}
.f-flame {{ top:0.4%; left:45%; width:min(32vw,500px); }}
/* the pair flanking the wordmark carry a slight lift over the rest */
.f-bust img, .f-flame img {{ filter:hue-rotate(var(--art-hue)) brightness(1.06); }}
@media (max-width:1330px) {{
  .f-bust {{ right:auto; left:2%; top:1%; width:min(22vw,280px); }}
}}
/* The reaching hands close the page. Anchored to the foot and bled past the
   right edge so the right arm's cut falls outside the viewport, while the
   left arm's cut sits back under the final trial tiles. It is composited on
   the page ink, so normal blending keeps its bounding box invisible, and the
   left-edge fade covers layouts where no tile happens to sit over the seam. */
.f-hands {{ bottom:4.5%; right:-4%; width:min(92vw,1600px);
  mix-blend-mode:normal;
  -webkit-mask-image:linear-gradient(90deg, transparent 0, #000 12%, #000 100%);
  mask-image:linear-gradient(90deg, transparent 0, #000 12%, #000 100%); }}
.f-hands img {{ opacity:.44; }}
main {{ width:100%; margin:0; padding:0 clamp(1.1rem,3.2vw,3.4rem) 6rem; }}
nav {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  padding:1.2rem 0 1rem; border-bottom:1px solid var(--red-faint); }}
.brand {{ font-family:'Bodoni Moda','Didot','Playfair Display',serif; font-weight:900;
  font-size:1.15rem; letter-spacing:.34em; text-transform:uppercase; color:var(--red-hi); }}
.navr {{ font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.2em;
  color:var(--red-dim); }}
/* Hero: the wordmark sits centred with the bound figure behind it, the way a
   product shot sits behind a masthead. */
header {{ margin:3.4rem 0 3rem; }}
.hero {{ position:relative; display:flex; align-items:center; justify-content:center;
  padding:1.5rem 0 1rem; }}
h1 {{ position:relative; z-index:1; margin:0; text-align:center;
  font-family:'Bodoni Moda','Didot','Playfair Display',serif; font-weight:900;
  font-size:clamp(3.6rem, 10vw, 8.2rem); line-height:.92; letter-spacing:.01em;
  text-transform:uppercase; color:var(--red-bright);
  text-shadow:0 0 26px rgba(7,4,4,.85), 0 0 8px rgba(7,4,4,.9); }}
.hero-meta {{ display:flex; justify-content:space-between; gap:2rem; flex-wrap:wrap;
  font-size:10.5px; font-weight:700; line-height:1.9; color:var(--red);
  text-transform:uppercase; letter-spacing:.2em; }}
.hm-right {{ text-align:right; }}
.stats {{ display:flex; gap:clamp(1.8rem,5vw,4.5rem); flex-wrap:wrap;
  justify-content:flex-start; margin:3.2rem 0 1rem; }}
.stat {{ text-align:left; }}
.stat .k {{ font-family:'Bodoni Moda','Didot',serif; font-weight:900; font-size:2.9rem;
  color:var(--red-bright); line-height:1.05; }}
.stat div:last-child {{ font-size:9.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:.24em; color:var(--red-mid); margin-top:.45rem; }}
h3 {{ font-family:'Bodoni Moda','Didot','Playfair Display',serif; font-weight:900;
  letter-spacing:.02em; color:var(--red-bright); font-size:1.6rem;
  margin:4rem 0 1.3rem; padding-bottom:.5rem; border-bottom:1px solid var(--red-faint); }}
.cats {{ display:grid; gap:0;
  grid-template-columns:repeat(auto-fit,minmax(min(100%,215px),1fr)); }}
.cat {{ display:flex; flex-direction:column; gap:.3rem; padding:1rem 1.4rem 1.1rem 0;
  border-bottom:1px solid var(--red-faint); }}
.cat-name {{ font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.2em;
  color:var(--red); }}
.cat-score {{ font-family:'Bodoni Moda','Didot',serif; font-weight:900; font-size:2rem;
  line-height:1; color:var(--red-bright); }}
.cat-num {{ font-size:10px; font-weight:700; color:var(--red-dim); letter-spacing:.1em; }}
.gridnote {{ font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.28em;
  color:var(--red-dim); margin:-.7rem 0 1.8rem; }}
/* Blocks of six small tiles alternate left and right down the page; the
   figure beside each block stretches to fill whatever gap the block left. */
.trials {{ --tile:215px; --tgap:1.3rem; --overlap:108px;
  --gridw:calc(3 * var(--tile) + 2 * var(--tgap));
  display:flex; flex-direction:column; gap:2.6rem; }}
.tgroup {{ position:relative; display:flex; }}
.tgroup.right {{ justify-content:flex-end; }}
.tgrid {{ flex:0 0 auto; position:relative; z-index:1; display:grid; gap:1.9rem var(--tgap);
  grid-template-columns:repeat(3,var(--tile)); align-content:start; }}
/* Absolute, pinned top and bottom, so the block's height comes from the tiles
   and the art is squeezed to fit it rather than the other way round. */
.gfig {{ position:absolute; top:0; bottom:0; z-index:0;
  display:flex; align-items:center; justify-content:center;
  pointer-events:none; mix-blend-mode:screen;
  -webkit-mask-image:radial-gradient(ellipse 74% 72% at 50% 50%, #000 58%, transparent 92%);
  mask-image:radial-gradient(ellipse 74% 72% at 50% 50%, #000 58%, transparent 92%); }}
/* half a tile of overlap, so the art tucks under the block it sits beside */
.tgroup.left  .gfig {{ left:calc(var(--gridw) - var(--overlap)); right:0; }}
.tgroup.right .gfig {{ right:calc(var(--gridw) - var(--overlap)); left:0; }}
/* Painted in the source files, so no hue filter here. Contained rather than
   cropped, and held well inside the gap so each figure reads whole and small
   instead of filling the space and losing its edges. */
.gfig img {{ width:auto; height:74%; max-width:58%; display:block;
  object-fit:contain; opacity:.9; filter:none; }}
/* Edge figures: run off the side of the page at 1.4x the other gap art, and
   carry the wordmark figures' exact blue, brightness and fade. No mask needed
   - their ground is true black, so screen blending alone dissolves it, and a
   radial falloff would eat the very edge that is meant to bleed. */
.gfig.g-edge {{ -webkit-mask-image:none; mask-image:none; }}
.gfig.g-edge img {{ height:104%; width:auto; max-width:none; opacity:.14;
  filter:hue-rotate(var(--art-hue)) brightness(1.06); }}
.gfig.g-right {{ justify-content:flex-end; }}
.gfig.g-right img {{ margin-right:-20%; }}
/* Turned a quarter clockwise. Unlike the bleeding warrior this one is shown
   whole and centred in its gap, sized so its turned width stays well short of
   the tiles, and sat further back: the plain fade rather than the lifted one,
   and a brightness cut instead of the lift. */
/* Torch: twice the height of the other gap art, so it overruns its block, and
   its foot dissolves into the page so it reads as rising out of the dark. The
   fade sits on the image, not the wrapper, so it covers the overrun too. */
.gfig.g-torch {{ align-items:flex-start;
  -webkit-mask-image:none; mask-image:none; }}
.gfig.g-torch img {{ height:99%; max-width:none; opacity:.45;
  filter:brightness(.45);
  -webkit-mask-image:linear-gradient(to top, transparent 0, #000 44%);
  mask-image:linear-gradient(to top, transparent 0, #000 44%); }}
.gfig.g-turn {{ justify-content:flex-start; }}
/* Stretched off its natural proportions: the quarter turn swaps the axes, so
   the box's height drives the on-screen horizontal span (1.5x) and its width
   drives the vertical (1.3x) - hence the ratio below and object-fit:fill.
   The turn also makes the visual box wider than the laid-out one, overhanging
   left by half the difference; translateX cancels that so the figure lands
   flush against the browser edge. */
.gfig.g-turn img {{ transform:translateX(94px) rotate(90deg);
  height:144%; aspect-ratio:.59; object-fit:fill;
  opacity:.09; filter:hue-rotate(var(--art-hue)) brightness(.6); }}
@media (max-width:1080px) {{ .gfig {{ display:none; }} }}
@media (max-width:680px) {{ .tgrid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
details {{ background:transparent; border:none; margin:0; padding:0; }}
details[open] {{ grid-column:1/-1; background:rgba(10,4,4,.92);
  border:1px solid var(--red-faint); padding:1.3rem 1.5rem; }}
details[open] .tname {{ font-size:1.35rem; min-height:0; }}
details[open].bad {{ border-left:3px solid var(--red-hi); }}
details[open].ok {{ border-left:3px solid var(--red-dim); }}
summary {{ cursor:pointer; list-style:none; }}
summary::-webkit-details-marker {{ display:none; }}
.tkick {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.26em;
  color:var(--red-mid); }}
.tname {{ margin:.4rem 0 0; font-family:'Bodoni Moda','Didot','Playfair Display',serif;
  font-weight:700; font-size:1.02rem; line-height:1.25; letter-spacing:.01em;
  color:var(--red-hi); min-height:2.6rem; transition:color .2s ease; }}
summary:hover .tname {{ color:var(--red-bright); }}
.gimg {{ margin-top:.9rem; aspect-ratio:16/9; opacity:.62;
  background-size:150px auto, cover; background-repeat:repeat, no-repeat;
  background-position:center; border:1px solid rgba(168,29,29,.22);
  transition:opacity .25s ease; }}
summary:hover .gimg {{ opacity:.82; }}
details[open] .gimg {{ max-width:340px; }}
.tmeta {{ margin-top:.75rem; font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:.2em; color:var(--red-mid); }}
.chip {{ font-size:9.5px; letter-spacing:.2em; text-transform:uppercase;
  padding:.18rem .6rem; margin-left:.45rem; border:1px solid; white-space:nowrap; }}
.chip.pass {{ color:var(--red); border-color:var(--red-dim); }}
.chip.fail {{ color:var(--ink); border-color:var(--red-hi); background:var(--red-hi); font-weight:700; }}
.chip.hard {{ color:#ffd9d4; border-color:var(--red-hi); background:#5c0f0f; font-weight:700; }}
table {{ border-collapse:collapse; width:100%; margin:.9rem 0; font-size:12px; }}
th {{ text-transform:uppercase; letter-spacing:.18em; font-weight:400; color:var(--red-dim); }}
th,td {{ text-align:left; padding:.32rem .6rem; border-bottom:1px solid rgba(160,31,31,.14); }}
.dim {{ color:#7c2020; }} .inj {{ color:#ff5544; font-style:italic; }} .errf {{ color:var(--red-hi); }}
.tl {{ font-size:11.5px; line-height:1.75; margin:.3rem 0; color:#963c3c; }}
code {{ color:#c22323; }}
.ans {{ background:rgba(160,31,31,.06); border:1px solid var(--red-faint);
  border-left:2px solid var(--red-dim); padding:.75rem 1rem; white-space:pre-wrap;
  font-family:'Bodoni Moda','Didot',serif; font-style:italic; font-size:1.1rem; color:#bf4a4a; }}
h4 {{ font-size:11px; text-transform:uppercase; letter-spacing:.3em; color:var(--red-mid);
  margin:1.2rem 0 .4rem; font-weight:700; }}
footer {{ margin-top:6rem; padding-top:1.6rem; border-top:1px solid var(--red-faint);
  display:flex; flex-wrap:wrap; align-items:baseline; gap:.9rem 2.2rem; }}
.fname {{ font-family:'Bodoni Moda','Didot','Playfair Display',serif; font-weight:700;
  font-size:1.5rem; color:var(--red-hi); letter-spacing:.02em; }}
.flinks {{ display:flex; flex-wrap:wrap; gap:.5rem 1.8rem; }}
.flinks a {{ font-size:11px; text-transform:uppercase; letter-spacing:.18em;
  color:var(--red); text-decoration:none; border-bottom:1px solid var(--red-faint);
  padding-bottom:.15rem; transition:color .2s ease, border-color .2s ease; }}
.flinks a:hover {{ color:#c22525; border-color:var(--red-dim); }}
.fnote {{ margin-left:auto; font-size:10px; text-transform:uppercase;
  letter-spacing:.2em; color:var(--red-dim); }}
::selection {{ background:var(--red-hi); color:var(--ink); }}
::-webkit-scrollbar {{ width:10px; }} ::-webkit-scrollbar-track {{ background:var(--ink); }}
::-webkit-scrollbar-thumb {{ background:var(--red-faint); border:2px solid var(--ink); }}
@supports (animation-timeline: view()) {{
  .fig, .gfig.g-edge {{ view-timeline:--figt block; }}
  .fig img, .gfig.g-edge img {{ animation:figfade linear both;
    animation-timeline:--figt; animation-duration:auto; }}
  @keyframes figfade {{
    0% {{ opacity:0; }}
    14% {{ opacity:.17; }}
    62% {{ opacity:.09; }}
    100% {{ opacity:.05; }} }}
  /* The pair flanking the wordmark ride a lifted curve. Opacity is what has
     to change here: these render around 0.1, and a brightness filter on
     something that transparent is invisible. */
  .f-bust img, .f-flame img, .gfig.g-right img {{ animation-name:figfade-hero; }}
  @keyframes figfade-hero {{
    0% {{ opacity:0; }}
    14% {{ opacity:.25; }}
    62% {{ opacity:.14; }}
    100% {{ opacity:.08; }} }}
}}
@media (prefers-reduced-motion: no-preference) {{
  @supports (animation-timeline: view()) {{
    details, .cat, .stat {{ animation:rise both; animation-timeline:view();
      animation-duration:auto; animation-range:entry 0% entry 55%; }}
    @keyframes rise {{ from {{ opacity:.08; transform:translateY(22px); }}
                       to {{ opacity:1; transform:none; }} }}
  }}
}}
</style></head><body>
{fig_layers}
<div class="layer grain"><div class="hold"><svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
<filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3"/>
<feColorMatrix type="saturate" values="0"/></filter>
<rect width="100%" height="100%" filter="url(#n)"/></svg></div></div>
<main>
<nav><span class="brand">Gauntlet</span><span class="navr">agent evaluation harness · run {esc(report.run_id)}</span></nav>
<header>
<div class="hero"><h1>Gauntlet</h1></div>
<div class="hero-meta">
<div>open harness · {esc(report.agent)} on trial<br>judge {esc(report.judge_model)}</div>
<div class="hm-right">fifty trials · every step recorded<br>model {esc(report.model)} · wall {report.wall_time_s:.1f}s</div>
</div>
</header>
<div class="stats">
<div class="stat"><div class="k">{agg['overall_score']:.3f}</div><div>overall score</div></div>
<div class="stat"><div class="k">{agg['pass_rate']:.0%}</div><div>pass rate</div></div>
<div class="stat"><div class="k">{agg['hard_fails']}</div><div>hard failures</div></div>
<div class="stat"><div class="k">{agg['n']}</div><div>scenarios</div></div>
<div class="stat"><div class="k">{agg['prompt_tokens'] + agg['completion_tokens']:,}</div><div>tokens</div></div>
{cost}</div>
<h3>Trials by Category</h3>
<div class="cats">{cat_rows}</div>
<h3>The Trials</h3>
<div class="gridnote">click a trial to unseal its full record</div>
<div class="trials">{blocks}</div>
<footer>
<div class="fname">Daiv Raval</div>
<div class="flinks">
<a href="https://github.com/daivraval">github.com/daivraval</a>
<a href="https://www.linkedin.com/in/daiv-raval">linkedin.com/in/daiv-raval</a>
<a href="tel:+919484446112">+91 94844 46112</a>
</div>
<div class="fnote">Gauntlet · an evaluation harness for tool-calling agents</div>
</footer>
</main>
</body></html>"""
