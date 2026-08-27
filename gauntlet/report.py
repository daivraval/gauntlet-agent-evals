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
#
# The look is the "Gauntlet Interactive Report" zine design: a two-token
# paper/ink palette, photocopy grain, and engraved figures pushed through SVG
# turbulence filters. Everything below follows design_handoff_gauntlet_report.
#
# The page stays a single portable file — every image is inlined as a data URI
# and the interactivity is vanilla JS over a JSON blob of the run, so a
# report.html can be mailed around or opened off a USB stick years later.

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ZINE_DIR = _ASSETS_DIR / "zine"

# Sorted scenario ids, matching the row-major tile order of the grid sheet
# (docs/prep_grid.py cuts tile N for the N-th id in this list).
_GRID_ORDER = (
    [f"AD-{i:02d}" for i in range(1, 6)] + [f"AR-{i:02d}" for i in range(1, 9)]
    + [f"ER-{i:02d}" for i in range(1, 11)] + [f"HA-{i:02d}" for i in range(1, 11)]
    + [f"MS-{i:02d}" for i in range(1, 8)] + [f"TS-{i:02d}" for i in range(1, 11)]
)
_GRID_INDEX = {sid: i for i, sid in enumerate(_GRID_ORDER)}

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _asset_uri(rel: str) -> str:
    """Inline one image asset as a data URI, keeping the report a single
    self-contained file. Missing files simply skip that element."""
    path = _ASSETS_DIR / rel
    if not path.exists():
        return ""
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _tile_uris() -> list[str]:
    """The 14 card thumbnails, inlined once and referenced by index so a
    fifty-trial grid costs fourteen images rather than fifty."""
    if not _ZINE_DIR.exists():
        return []
    names = sorted(p.name for p in (_ZINE_DIR / "tiles").glob("t*.png")) \
        if (_ZINE_DIR / "tiles").exists() else []
    return [uri for name in names if (uri := _asset_uri(f"zine/tiles/{name}"))]


# The four turbulence filters that carry the whole treatment. Only the hero
# wordmark gets #woodcutTorn and only the vol. label gets #woodcutText —
# filtering body copy costs legibility for nothing.
_SVG_FILTERS = """
<svg width="0" height="0" style="position:absolute;pointer-events:none" aria-hidden="true">
<filter id="woodcut" x="-8%" y="-8%" width="116%" height="116%" color-interpolation-filters="sRGB">
  <feTurbulence type="fractalNoise" baseFrequency="0.75" numOctaves="3" seed="7" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="2.3" xChannelSelector="R" yChannelSelector="G" result="warp"/>
  <feMorphology in="warp" operator="dilate" radius="0.35" result="bleed"/>
  <feComponentTransfer in="bleed" result="crush"><feFuncA type="linear" slope="1.9" intercept="-0.24"/></feComponentTransfer>
  <feTurbulence type="fractalNoise" baseFrequency="0.62" numOctaves="2" seed="23" result="sp"/>
  <feColorMatrix in="sp" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  1 0 0 0 0" result="spA"/>
  <feComponentTransfer in="spA" result="spT"><feFuncA type="discrete" tableValues="0 0 0 0 0 0 0 0 0 1"/></feComponentTransfer>
  <feComposite in="crush" in2="spT" operator="out"/>
</filter>
<filter id="woodcutTorn" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
  <feTurbulence type="fractalNoise" baseFrequency="0.7" numOctaves="3" seed="17" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="3.2" xChannelSelector="R" yChannelSelector="G" result="warp"/>
  <feMorphology in="warp" operator="dilate" radius="0.7" result="bleed"/>
  <feComponentTransfer in="bleed" result="crush"><feFuncA type="linear" slope="2.1" intercept="-0.16"/></feComponentTransfer>
  <feTurbulence type="turbulence" baseFrequency="0.05 0.19" numOctaves="3" seed="29" result="rip"/>
  <feColorMatrix in="rip" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  1 0 0 0 0" result="ripA"/>
  <feComponentTransfer in="ripA" result="ripT"><feFuncA type="discrete" tableValues="0 0 0 0 1 1"/></feComponentTransfer>
  <feComposite in="crush" in2="ripT" operator="out"/>
</filter>
<filter id="woodcutText" x="-6%" y="-6%" width="112%" height="112%" color-interpolation-filters="sRGB">
  <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" seed="7" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="1.35" xChannelSelector="R" yChannelSelector="G" result="warp"/>
  <feMorphology in="warp" operator="dilate" radius="0.55" result="bleed"/>
  <feComponentTransfer in="bleed" result="crush"><feFuncA type="linear" slope="1.55" intercept="-0.1"/></feComponentTransfer>
</filter>
</svg>
"""

# Page grain: an ink-coloured element *masked* by turbulence, never the noise
# tile painted as a background — painting it washes the pure-black night page
# toward grey.
_NOISE_TILE = (
    "url(\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
    "<filter id='{fid}'><feTurbulence type='fractalNoise' baseFrequency='{freq}' numOctaves='{oct}' seed='{seed}'/>"
    "<feColorMatrix type='saturate' values='0'/><feComponentTransfer><feFuncA type='discrete' tableValues='{drop}'/>"
    "</feComponentTransfer></filter><rect width='{size}' height='{size}' filter='url(%23{fid})'/></svg>\")"
)


def _noise_layers() -> str:
    specs = [
        (65, "s", 220, "0.62", 4, 0, "0 0 0 0 0 0 0 1", ".11"),
        (66, "m", 340, "0.16", 5, 9, "0 0 0 0 0 1", ".08"),
        (67, "d", 300, "0.9", 2, 41, "0 0 0 0 0 0 0 0 0 1", ".14"),
    ]
    out = []
    for z, fid, size, freq, octaves, seed, drop, op in specs:
        mask = _NOISE_TILE.format(size=size, fid=fid, freq=freq, oct=octaves, seed=seed, drop=drop)
        out.append(
            f'<div class="grain" aria-hidden="true" style="z-index:{z};opacity:calc({op} * var(--grain-op));'
            f'-webkit-mask-image:{mask};mask-image:{mask};'
            f'-webkit-mask-size:{size}px {size}px;mask-size:{size}px {size}px"></div>'
        )
    return "".join(out)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

/* Two tokens and alpha steps. There are no accent colours anywhere in this
   design; a verdict is encoded by inversion, not by hue. */
:root{
  --paper:#f7f6f3; --ink:#0a0a0a; --dim:rgba(10,10,10,.58); --faint:rgba(10,10,10,.34);
  --line:rgba(10,10,10,.16); --wash:rgba(247,246,243,.74);
  --art-filter:grayscale(1) contrast(1.5) brightness(1.06); --art-blend:multiply;
  --art-op:.9; --grain-op:.78; --art-k:1;
}
html[data-theme="dark"]{
  --paper:#000000; --ink:#f4f2ee; --dim:rgba(244,242,238,.56); --faint:rgba(244,242,238,.3);
  --line:rgba(244,242,238,.14); --wash:rgba(10,10,10,.76);
  --art-filter:grayscale(1) contrast(1.5) invert(1) brightness(1.04); --art-blend:screen;
  --art-op:.82; --grain-op:.44;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;overflow-x:clip;max-width:100%}
/* The handoff asks for `transition:background .5s ease,color .5s ease` here.
   It is left off on purpose: it desynchronises the theme switch. Body's
   painted background and inherited colour animate over half a second, while
   every element coloured straight from var(--dim) / var(--ink) — marginalia,
   chips, labels, stat captions — flips in the same frame as the token. Toggle
   during that window, or in any context where transitions stall (a hidden or
   throttled tab), and you get the new background under the old theme's text:
   near-white on paper, effectively invisible. The theme switch has to be
   atomic, so nothing here transitions. */
body{margin:0;background:var(--paper);color:var(--ink);cursor:none;overflow-x:hidden;
  font:14px/1.75 'Courier Prime','Courier New',monospace}
a{color:var(--ink);text-decoration:none}
a:hover{opacity:.6}
::selection{background:var(--ink);color:var(--paper)}
::-webkit-scrollbar{width:9px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--faint)}

/* The workhorse label style — some twenty elements share it. */
.lbl{font-size:9.5px;font-weight:700;letter-spacing:.24em;text-transform:uppercase;color:var(--dim)}

.grain{position:fixed;inset:0;pointer-events:none;mix-blend-mode:var(--art-blend);background:var(--ink)}
#cursor,#cursorDot{position:fixed;left:0;top:0;z-index:80;pointer-events:none}
#cursor{width:34px;height:34px;margin:-17px 0 0 -17px;border:1px solid var(--ink);
  border-radius:50%;mix-blend-mode:difference}
#cursorDot{width:5px;height:5px;margin:-2.5px 0 0 -2.5px;background:var(--ink)}
.stamp{position:fixed;z-index:79;pointer-events:none;width:120px;height:120px;
  border:1px solid var(--ink);border-radius:50%;
  animation:stamp .62s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes stamp{from{transform:translate(-50%,-50%) scale(.2);opacity:.85}
  to{transform:translate(-50%,-50%) scale(2.6);opacity:0}}
#progress{position:fixed;top:0;left:0;height:2px;z-index:70;background:var(--ink);opacity:.7;width:0}
#hint{position:fixed;left:50%;bottom:1rem;transform:translateX(-50%);z-index:78;display:flex;
  gap:1.1rem;padding:.45rem .9rem;border:1px solid var(--line);background:var(--wash);
  backdrop-filter:blur(7px);font-size:8.5px;font-weight:700;letter-spacing:.22em;
  text-transform:uppercase;color:var(--dim);opacity:.25;transition:opacity .4s ease}

nav{position:fixed;top:0;left:0;right:0;z-index:55;display:flex;align-items:center;
  justify-content:space-between;gap:1.5rem;padding:1.1rem clamp(1rem,3.4vw,3rem);
  backdrop-filter:blur(7px);background:var(--wash);border-bottom:1px solid var(--line)}
.wordmark{font-family:'Bodoni Moda',Didot,serif;font-weight:900;font-size:.95rem;
  letter-spacing:.42em;text-transform:uppercase}
.navlinks{display:flex;gap:1.8rem;align-items:center}
.navlinks a{color:var(--dim)}
.navlinks a:hover{color:var(--ink);opacity:1}
#themeBtn{display:flex;align-items:center;gap:.6rem;cursor:none;background:transparent;
  border:1px solid var(--line);color:var(--ink);font:inherit;font-size:9.5px;font-weight:700;
  letter-spacing:.24em;text-transform:uppercase;padding:.5rem .85rem;
  transition:background .25s ease,color .25s ease}
#themeBtn:hover{background:var(--ink);color:var(--paper)}
#themeDot{display:inline-block;width:8px;height:8px;border:1px solid currentColor;background:currentColor}
html[data-theme="dark"] #themeDot{background:transparent}

/* ---- hero ---- */
.hero{position:relative;min-height:100vh;display:flex;flex-direction:column;
  justify-content:center;align-items:center;padding:8.5rem clamp(1rem,3.4vw,3rem) 4rem;
  overflow:hidden}
.hero-angel{position:absolute;left:50%;transform:translateX(-50%);top:190px;width:746px;
  height:1232px;max-width:none;display:block;opacity:.2;z-index:1;pointer-events:none;
  filter:var(--art-filter) url(#woodcut)}
.hero-tl{position:absolute;top:7.5rem;left:clamp(1rem,3.4vw,3rem);z-index:3;
  max-width:min(46vw,560px);font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(1.5rem,4.2vw,3.2rem);line-height:.98;letter-spacing:-.01em;text-transform:uppercase}
.hero-tr{position:absolute;top:7.6rem;right:clamp(1rem,3.4vw,3rem);z-index:3;
  font-family:'Archivo Black',Impact,sans-serif;font-size:clamp(.7rem,1.2vw,1rem);
  letter-spacing:.02em;text-transform:uppercase;filter:url(#woodcutText)}
.hero-mark{position:relative;z-index:4;margin-top:auto;text-align:center;width:100vw;
  left:50%;transform:translateX(-50%);height:182px}
.hero-eyebrow{font-size:9.5px;font-weight:700;letter-spacing:.5em;text-transform:uppercase;color:var(--dim)}
h1{margin:.5rem 0 0;width:100vw;font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(3.4rem,14.5vw,13rem);line-height:.84;letter-spacing:.34em;
  /* cancels the trailing letter-space so the word sits optically centred */
  text-indent:.34em;text-transform:uppercase;white-space:nowrap;filter:url(#woodcutTorn)}
.hero-foot{margin-top:.9rem;display:flex;justify-content:center;gap:clamp(1rem,4vw,3.4rem);
  flex-wrap:wrap;font-size:9.5px;font-weight:700;letter-spacing:.3em;text-transform:uppercase;color:var(--dim)}
.cue{position:absolute;right:clamp(1rem,3.4vw,3rem);bottom:1.6rem;z-index:4;display:flex;
  flex-direction:column;align-items:flex-end;gap:.4rem;font-size:9px;font-weight:700;
  letter-spacing:.3em;text-transform:uppercase;color:var(--dim)}
.cue i{font-style:normal;font-size:14px;animation:cue 2.4s ease-in-out infinite}
@keyframes cue{0%{transform:translateY(0);opacity:.25}50%{transform:translateY(9px);opacity:.8}
  100%{transform:translateY(0);opacity:.25}}

/* Marginalia are pinned as pixel offsets from viewport centre, not from a page
   edge — the angel is centre-anchored at a fixed width, so centre-relative
   offsets are the only thing that holds the constellation's shape. */
.mg{position:absolute;z-index:3}
@media (max-width:1180px){.mg{display:none}}

/* ---- sections ---- */
section{position:relative;overflow:hidden}
.inner{position:relative;z-index:2;max-width:1500px;margin:0 auto}
#record{padding:clamp(4rem,10vw,9rem) clamp(1rem,3.4vw,3rem) 0}
.premise{display:grid;gap:clamp(2rem,6vw,5rem);
  grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));align-items:start}
.premise h2{margin:1.2rem 0 0;font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(1.7rem,3.4vw,2.9rem);line-height:1.18;text-wrap:pretty}
.premise .col2{display:flex;flex-direction:column;gap:1.4rem;max-width:44ch}
.premise p{margin:0;color:var(--dim)}
.chips{display:flex;flex-wrap:wrap;gap:.5rem}
.chips span{border:1px solid var(--line);padding:.28rem .6rem;font-size:9.5px;font-weight:700;
  letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
  transition:background .2s ease,color .2s ease}
.chips span:hover{background:var(--ink);color:var(--paper)}
/* 160px minimum, not 120px — at 120 the six-figure token count clipped. */
.stats{display:grid;gap:2.2rem clamp(1.5rem,5vw,4rem);
  grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  margin:clamp(3.5rem,9vw,7rem) auto 0;max-width:1500px;
  padding-bottom:clamp(3rem,7vw,5rem);border-bottom:1px solid var(--line)}
.stats .v{font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(1.9rem,3.4vw,3rem);line-height:1.2;white-space:nowrap}
.stats .k{margin-top:.5rem;font-size:9px;font-weight:700;letter-spacing:.26em;
  text-transform:uppercase;color:var(--dim)}

#points{padding:clamp(3.5rem,8vw,6rem) clamp(1rem,3.4vw,3rem) 0}
h2.head{margin:0 0 2.2rem;font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(1.5rem,3vw,2.2rem);letter-spacing:.02em}
.cats{display:grid;gap:0;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.cat{padding:1.3rem 1.6rem 1.4rem 0;border-bottom:1px solid var(--line);cursor:none;
  transition:opacity .3s ease}
.cats.hovering .cat{opacity:.42}
.cats.hovering .cat:hover{opacity:1}
.cat-top{display:flex;align-items:baseline;justify-content:space-between;gap:1rem}
.cat-name{font-size:10px;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--dim)}
.cat-meta{font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--faint)}
.cat-row{display:flex;align-items:flex-end;gap:.9rem;margin-top:.4rem}
.cat-score{font-family:'Bodoni Moda',Didot,serif;font-weight:900;font-size:2.2rem;line-height:1}
.track{flex:1;height:6px;border:1px solid var(--line);margin-bottom:.45rem;overflow:hidden}
.track i{display:block;height:100%;background:var(--ink);width:0;
  transition:width 1.1s cubic-bezier(.2,.8,.2,1)}

#trials{padding:clamp(4rem,9vw,7rem) clamp(1rem,3.4vw,3rem) 0}
.trials-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1.5rem;flex-wrap:wrap}
.trials-head h2{margin:0;font-family:'Bodoni Moda',Didot,serif;font-weight:900;
  font-size:clamp(2rem,5vw,3.6rem);letter-spacing:.01em}
.filters{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:1.8rem 0 2.2rem}
#q{font:inherit;font-size:10px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  padding:.42rem .7rem;border:1px solid var(--line);background:transparent;color:var(--ink);
  outline:none;width:180px}
#q:focus{border-color:var(--ink)}
#shown{font-size:9px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;
  color:var(--faint);margin-right:.6rem}
.filters button{cursor:none;font:inherit;font-size:9.5px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;padding:.42rem .8rem;border:1px solid var(--line);
  background:transparent;color:var(--dim);transition:background .2s ease,color .2s ease}
.filters button:hover{border-color:var(--ink)}
.filters button.on{background:var(--ink);color:var(--paper)}

#detail{position:relative;border:1px solid var(--line);border-left:3px solid var(--ink);
  background:var(--wash);backdrop-filter:blur(6px);padding:1.6rem clamp(1rem,2.5vw,2rem);
  margin-bottom:2.2rem}
#detail[hidden]{display:none}
#close{position:absolute;top:.8rem;right:.9rem;cursor:none;background:transparent;
  border:1px solid var(--line);color:var(--ink);font:inherit;font-size:10px;letter-spacing:.2em;
  padding:.25rem .55rem}
#close:hover{background:var(--ink);color:var(--paper)}
.detail-grid{display:grid;gap:clamp(1.2rem,3vw,2.4rem);
  grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr))}
.detail-name{margin:.5rem 0 1rem;font-family:'Bodoni Moda',Didot,serif;font-size:1.6rem;line-height:1.2}
.detail-tile{border:1px solid var(--line);aspect-ratio:1/1;max-width:250px;overflow:hidden}
.detail-tile img{width:100%;height:100%;object-fit:cover;display:block;
  filter:var(--art-filter) contrast(1.05);opacity:var(--art-op)}
.sec{font-size:10px;font-weight:700;letter-spacing:.3em;text-transform:uppercase;
  color:var(--dim);margin-bottom:.6rem}
.grade{display:flex;align-items:center;gap:.8rem;padding:.35rem 0;
  border-bottom:1px solid var(--line);font-size:12px}
.grade .n{flex:0 0 130px;color:var(--dim)}
.grade .s{flex:0 0 42px;font-weight:700}
.grade .t{flex:1;height:4px;background:var(--line)}
.grade .t i{display:block;height:100%;background:var(--ink)}
.call{font-size:11.5px;line-height:1.7;color:var(--dim)}
.call b{color:var(--ink);font-weight:400}
.call em{font-style:italic}
.answer{border:1px solid var(--line);border-left:2px solid var(--ink);padding:.9rem 1.1rem;
  font-family:'Bodoni Moda',Didot,serif;font-style:italic;font-size:1.05rem;line-height:1.5;
  white-space:pre-wrap}
.note{margin-top:1rem;font-size:11px;color:var(--faint)}

.grid-wrap{position:relative}
.grid-art{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:100vw;
  z-index:0;pointer-events:none;opacity:calc(.34 * var(--art-k));
  -webkit-mask-image:radial-gradient(ellipse 80% 90% at 50% 50%,#000 62%,transparent 98%);
  mask-image:radial-gradient(ellipse 80% 90% at 50% 50%,#000 62%,transparent 98%)}
.grid-art img{width:100%;height:auto;display:block;opacity:.5;filter:var(--art-filter) url(#woodcut)}
#grid{position:relative;z-index:1;display:grid;gap:1.6rem clamp(1rem,2vw,1.6rem);
  grid-template-columns:repeat(auto-fill,minmax(min(100%,215px),1fr))}
.card{cursor:none;text-align:left;background:transparent;border:0;padding:0;color:inherit;
  font:inherit;transition:opacity .3s ease,transform .3s ease}
.card:hover{transform:translateY(-4px)}
#grid.picked .card{opacity:.45}
#grid.picked .card.on{opacity:1}
.card-id{font-size:9.5px;font-weight:700;letter-spacing:.24em;text-transform:uppercase;color:var(--dim)}
.card-name{margin:.35rem 0 .7rem;font-family:'Bodoni Moda',Didot,serif;font-weight:700;
  font-size:1rem;line-height:1.25;min-height:2.5rem}
.card-tile{aspect-ratio:16/10;border:1px solid var(--line);overflow:hidden}
.card-tile img{width:100%;height:100%;object-fit:cover;display:block;filter:var(--art-filter);
  opacity:var(--art-op);transition:transform .6s cubic-bezier(.2,.8,.2,1)}
.card:hover .card-tile img{transform:scale(1.06)}
.card-meta{display:flex;align-items:center;justify-content:space-between;gap:.6rem;
  margin-top:.6rem;font-size:9.5px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim)}
.verdict{border:1px solid var(--line);padding:.15rem .5rem}
.verdict.fail{background:var(--ink);color:var(--paper)}

footer{position:relative;z-index:2;margin:clamp(2rem,6vw,4rem) auto 0;
  padding:2rem clamp(1rem,3.4vw,3rem) 3.5rem;border-top:1px solid var(--line);display:flex;
  flex-wrap:wrap;align-items:baseline;gap:1rem 2.4rem;max-width:1500px}
.fname{font-family:Tahoma,Verdana,sans-serif;font-weight:700;font-size:1.4rem}
.flinks{display:flex;flex-wrap:wrap;gap:.5rem 1.8rem}
.flinks a{font-family:Tahoma,Verdana,sans-serif;font-size:10.5px;font-weight:700;
  letter-spacing:.2em;text-transform:uppercase;border-bottom:1px solid var(--line);
  padding-bottom:.15rem}
.frun{margin-left:auto;font-size:9.5px;font-weight:700;letter-spacing:.24em;
  text-transform:uppercase;color:var(--faint)}

@media (prefers-reduced-motion: no-preference){
  @supports (animation-timeline: view()){
    [data-rise]{animation:rise both;animation-timeline:view();animation-range:entry 0% entry 46%}
    @keyframes rise{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
  }
}
/* Pointer-driven chrome is decoration; hand the real cursor back to anyone who
   asked for less motion. */
@media (prefers-reduced-motion: reduce){
  body{cursor:auto}
  #cursor,#cursorDot,.stamp{display:none}
  .card,#themeBtn,.filters button,.cat,#close{cursor:pointer}
}
"""

_JS = r"""
(function(){
  var D = __DATA__;
  var root = document.documentElement;

  function esc(s){ return String(s == null ? "" : s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

  /* ---- theme ---------------------------------------------------------- */
  var theme = "dark";
  try { theme = localStorage.getItem("gauntlet-theme") || "dark"; } catch(e) {}
  function setTheme(t){
    theme = t; root.dataset.theme = t;
    document.getElementById("themeLabel").textContent = t === "dark" ? "night" : "day";
    try { localStorage.setItem("gauntlet-theme", t); } catch(e) {}
  }
  setTheme(theme);
  document.getElementById("themeBtn").addEventListener("click", function(){
    setTheme(theme === "dark" ? "light" : "dark");
  });

  /* ---- trials --------------------------------------------------------- */
  var filter = "all", openId = null, query = "";
  var grid = document.getElementById("grid");
  var detail = document.getElementById("detail");
  var shown = document.getElementById("shown");
  var visibleIds = [];

  function visible(){
    var q = query.trim().toLowerCase();
    return D.trials.filter(function(t){
      if (filter !== "all" && t.catKey !== filter) return false;
      if (!q) return true;
      return (t.id + " " + t.name + " " + t.cat + " " + t.note).toLowerCase().indexOf(q) >= 0;
    });
  }

  function renderGrid(){
    var list = visible();
    visibleIds = list.map(function(t){ return t.id; });
    shown.textContent = list.length + " shown";
    grid.classList.toggle("picked", openId != null);
    grid.innerHTML = list.map(function(t){
      return '<button class="card' + (t.id === openId ? " on" : "") + '" data-id="' + esc(t.id) + '" data-rise="1">'
        + '<div class="card-id">' + esc(t.id) + ' · ' + esc(t.cat) + '</div>'
        + '<div class="card-name">' + esc(t.name) + '</div>'
        + '<div class="card-tile"><img src="' + D.tiles[t.tile] + '" alt=""></div>'
        + '<div class="card-meta"><span>score ' + t.score + '</span>'
        + '<span class="verdict ' + t.verdict + '">' + t.verdict + '</span></div></button>';
    }).join("");
  }

  function renderDetail(){
    var t = null;
    for (var i = 0; i < D.trials.length; i++) if (D.trials[i].id === openId) t = D.trials[i];
    if (!t) { detail.hidden = true; detail.innerHTML = ""; return; }
    detail.hidden = false;
    detail.innerHTML = '<button id="close">close ✕</button><div class="detail-grid">'
      + '<div><div class="lbl">' + esc(t.id) + ' · ' + esc(t.cat) + '</div>'
      + '<div class="detail-name">' + esc(t.name) + '</div>'
      + '<div class="detail-tile"><img src="' + D.tiles[t.tile] + '" alt=""></div>'
      + '<div class="lbl" style="margin-top:1rem">score ' + t.score + ' · ' + t.verdict
      + (t.hard ? ' · hard fail' : '') + '</div></div>'
      + '<div><div class="sec">Graders</div>'
      + t.grades.map(function(g){
          return '<div class="grade"><span class="n">' + esc(g.name) + '</span>'
            + '<span class="s">' + g.score + '</span>'
            + '<span class="t"><i style="width:' + g.bar + '"></i></span></div>';
        }).join("")
      + '<div class="sec" style="margin:1.4rem 0 .5rem">Tool calls</div>'
      + (t.calls.length
          ? t.calls.map(function(k){
              var flag = (k.faulted ? '<em>sabotaged</em> ' : '') + (k.error ? '<em>error</em> ' : '');
              return '<div class="call">' + flag + '<b>' + esc(k.call) + '</b> → ' + esc(k.result) + '</div>';
            }).join("")
          : '<div class="call">(none)</div>')
      + '</div>'
      + '<div><div class="sec">Final answer</div>'
      + '<div class="answer">' + (t.answer ? esc(t.answer) : '(none)') + '</div>'
      + '<div class="note">' + esc(t.note) + '</div></div></div>';
    detail.querySelector("#close").addEventListener("click", function(){ open(null); });
  }

  function open(id){
    openId = id;
    renderGrid();
    renderDetail();
    if (id) detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  grid.addEventListener("click", function(e){
    var card = e.target.closest(".card");
    if (card) open(card.dataset.id === openId ? null : card.dataset.id);
  });

  document.getElementById("q").addEventListener("input", function(e){
    query = e.target.value; openId = null; renderGrid(); renderDetail();
  });

  var buttons = Array.prototype.slice.call(document.querySelectorAll(".filters button"));
  function setFilter(key){
    filter = key; openId = null;
    buttons.forEach(function(b){ b.classList.toggle("on", b.dataset.key === key); });
    renderGrid(); renderDetail();
  }
  buttons.forEach(function(b){
    b.addEventListener("click", function(){ setFilter(b.dataset.key); });
  });

  renderGrid();

  /* ---- category bars grow once, just after mount ----------------------- */
  var cats = document.getElementById("cats");
  setTimeout(function(){
    Array.prototype.forEach.call(cats.querySelectorAll(".track i"), function(el){
      el.style.width = el.dataset.w;
    });
  }, 260);
  cats.addEventListener("mouseenter", function(){ cats.classList.add("hovering"); });
  cats.addEventListener("mouseleave", function(){ cats.classList.remove("hovering"); });

  /* ---- keyboard ------------------------------------------------------- */
  var filterKeys = ["all"].concat(D.catKeys);
  window.addEventListener("keydown", function(e){
    var tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA"){ if (e.key === "Escape") e.target.blur(); return; }
    var k = e.key;
    if (k === "Escape") return open(null);
    if (k === "t" || k === "T") return setTheme(theme === "dark" ? "light" : "dark");
    if (k === "f" || k === "F") return setFilter(filterKeys[(filterKeys.indexOf(filter) + 1) % filterKeys.length]);
    if (k === "ArrowRight" || k === "ArrowLeft"){
      if (!visibleIds.length) return;
      var i = visibleIds.indexOf(openId), step = k === "ArrowRight" ? 1 : -1;
      var next = i < 0 ? (step > 0 ? 0 : visibleIds.length - 1)
                       : (i + step + visibleIds.length) % visibleIds.length;
      e.preventDefault();
      return open(visibleIds[next]);
    }
  });

  /* ---- pointer chrome, scroll progress -------------------------------- */
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var bar = document.getElementById("progress");
  window.addEventListener("scroll", function(){
    var h = root.scrollHeight - window.innerHeight;
    bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0).toFixed(2) + "%";
  }, { passive: true });

  if (!reduce){
    var ring = document.getElementById("cursor"), dot = document.getElementById("cursorDot");
    var hintEl = document.getElementById("hint"), hint = 0, pt = null, raf = null;
    window.addEventListener("mousemove", function(e){
      pt = e;
      if (raf) return;
      raf = requestAnimationFrame(function(){
        raf = null;
        var tf = "translate3d(" + pt.clientX + "px," + pt.clientY + "px,0)";
        ring.style.transform = tf; dot.style.transform = tf;
        hint = Math.min(1, hint + 0.06);
        hintEl.style.opacity = (0.25 + hint * 0.6).toFixed(2);
      });
    }, { passive: true });
    setTimeout(function(){ hint = 0.55; hintEl.style.opacity = "0.58"; }, 4200);

    document.addEventListener("click", function(e){
      var s = document.createElement("div");
      s.className = "stamp";
      s.style.left = e.clientX + "px";
      s.style.top = e.clientY + "px";
      document.body.appendChild(s);
      setTimeout(function(){ s.remove(); }, 640);
    });
  }
})();
"""

# Editorial marginalia. Every horizontal position is an offset from viewport
# centre because the angel is centre-anchored at a fixed pixel width — do not
# convert these to percentages or edge offsets, the constellation falls apart.
#
# Two deviations from the handoff, both deliberate:
#  * the box widths below are the source's `max-width` values, not the `width`
#    values sitting beside them — max-width caps width, so those are what the
#    original actually rendered at;
#  * "Ask the trajectory" is lifted from top:940px to 838px. At 940 it collides
#    with "The world is sealed" at 946 — they overlap at every viewport width,
#    since both are pinned to centre. The lift is the smallest change that
#    makes both legible.
_MARGINALIA = [
    ("left:calc(50% - 273px);top:297px;width:143px;font-size:9.5px;font-weight:700;"
     "letter-spacing:.24em;text-transform:uppercase;line-height:2;color:var(--dim)",
     "every step recorded<br>nothing reconstructed after"),
    ("left:calc(50% - 520px);top:566px;max-width:216px;font-family:'EB Garamond',Georgia,serif;"
     "font-size:17px;line-height:1.35",
     "Suffering is measurable<br>if you instrument it right."),
    ("left:calc(50% - 475px);top:761px;max-width:200px;font-family:'Bodoni Moda',Didot,serif;"
     "font-style:italic;font-size:15.5px;line-height:1.45;color:var(--dim)",
     "&ldquo;A tool call is a promise. The harness only records whether it was kept.&rdquo;"),
    ("left:calc(50% - 546px);top:946px;max-width:190px;font-family:'EB Garamond',Georgia,serif;"
     "font-size:14px;line-height:1.5;color:var(--dim)",
     "The world is sealed. The agent is not. Every fault it meets was placed there on purpose."),
    ("left:calc(50% - 678px);top:838px;max-width:230px;font-family:'Bodoni Moda',Didot,serif;"
     "font-weight:700;font-size:16px;line-height:1.4",
     "Ask the trajectory.<br>Ask what it has witnessed."),
    ("left:calc(50% + 272px);top:361px;width:180px;text-align:right;"
     "font-family:'Bodoni Moda',Didot,serif;font-weight:700;font-size:17px;line-height:1.3",
     "&ldquo;Who should judge the judge, if not the record?&rdquo;"),
    ("left:calc(50% + 369px);top:488px;max-width:200px;text-align:right;"
     "font-family:'Bodoni Moda',Didot,serif;font-style:italic;font-size:17px;"
     "line-height:1.35;color:var(--dim)",
     "All abundance of tokens starts first in the plan."),
    ("left:calc(50% + 378px);top:620px;max-width:220px;text-align:right;font-size:11px;"
     "line-height:1.6;color:var(--dim)",
     "well-fed agents behave better than famished ones"),
    ("left:calc(50% + 335px);top:842px;max-width:166px;text-align:right;"
     "font-family:'EB Garamond',Georgia,serif;font-size:15px;line-height:1.4;"
     "border-bottom:1px solid var(--line);padding-bottom:.2rem",
     "&ldquo;Do not grade a failure you have not reproduced.&rdquo;"),
    ("left:calc(50% + 350px);top:1012px;max-width:200px;text-align:right;"
     "font-family:'Bodoni Moda',Didot,serif;font-weight:700;font-size:16px;line-height:1.35",
     "&ldquo;I will read the trajectory and cleanse my priors.&rdquo;"),
    ("right:calc(50% - 740px);top:964px;max-width:250px;text-align:right;"
     "font-family:'Bodoni Moda',Didot,serif;font-style:italic;font-size:15px;color:var(--dim)",
     "Perhaps a hard fail is a frozen tear"),
]


def _trial_payload(r: ScenarioResult, tile_count: int) -> dict:
    """One trial, flattened into what the page needs to draw a card and a
    detail panel. Grader sub-scores are the real ones — the design prototype
    faked them from a seeded jitter because it had no harness output to read."""
    calls = []
    if r.trajectory is not None:
        for s in r.trajectory.tool_steps:
            args = json.dumps(s.arguments or {})
            calls.append({
                "call": f"{s.tool or ''}({args[:100]})",
                "result": (s.result or "")[:180],
                "faulted": bool(s.faulted),
                "error": bool(s.is_error),
            })

    # The search index and the detail footnote both want one line of "what
    # actually happened here" — the weakest grader says it best.
    weakest = min(r.grades, key=lambda g: g.score, default=None)
    note = weakest.details if weakest is not None else ""
    if r.hard_fails:
        note = f"hard fail: {', '.join(r.hard_fails)} · {note}"

    order_idx = _GRID_INDEX.get(r.scenario_id, 0)
    return {
        "id": r.scenario_id,
        "catKey": r.category.value,
        "cat": r.category.value.replace("_", " "),
        "name": r.name,
        "note": note,
        "score": f"{r.score:.2f}",
        "verdict": "pass" if r.passed else "fail",
        "hard": bool(r.hard_fails),
        "tile": (order_idx % tile_count) if tile_count else 0,
        "grades": [
            {"name": g.grader, "score": f"{g.score:.2f}", "bar": f"{g.score * 100:.0f}%"}
            for g in r.grades
        ],
        "calls": calls,
        "answer": (r.trajectory.final_answer or "") if r.trajectory else "",
    }


def to_html(report: RunReport) -> str:
    agg = aggregate(report)
    esc = html.escape
    tiles = _tile_uris()

    payload = {
        "tiles": tiles,
        "catKeys": list(agg["categories"]),
        "trials": [_trial_payload(r, len(tiles)) for r in report.results],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    marginalia = "".join(
        f'<div class="mg" style="{style}">{text}</div>' for style, text in _MARGINALIA
    )

    cat_rows = "".join(
        f'<div class="cat" data-rise="1">'
        f'<div class="cat-top"><span class="cat-name">{esc(cat.replace("_", " "))}</span>'
        f'<span class="cat-meta">{stats["pass_rate"]:.0%} pass · {stats["n"]} trials</span></div>'
        f'<div class="cat-row"><span class="cat-score">{stats["avg_score"]:.2f}</span>'
        f'<span class="track"><i data-w="{stats["avg_score"] * 100:.0f}%"></i></span></div></div>'
        for cat, stats in agg["categories"].items()
    )

    stat_cells = [
        (f"{agg['overall_score']:.3f}", "overall score"),
        (f"{agg['pass_rate']:.0%}", "pass rate"),
        (str(agg["hard_fails"]), "hard failures"),
        (str(agg["n"]), "scenarios"),
        (f"{agg['prompt_tokens'] + agg['completion_tokens']:,}", "tokens"),
    ]
    if agg["cost_usd"]:
        stat_cells.append((f"${agg['cost_usd']:.4f}", "est. cost"))
    stats_html = "".join(
        f'<div data-rise="1"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
        for v, k in stat_cells
    )

    from .tools import TOOLS
    tool_chips = "".join(f"<span>{esc(name)}</span>" for name in TOOLS)

    filter_buttons = '<button class="on" data-key="all">all trials</button>' + "".join(
        f'<button data-key="{esc(cat)}">{esc(cat.replace("_", " "))}</button>'
        for cat in agg["categories"]
    )

    n_graders = len({g.grader for r in report.results for g in r.grades})
    angel = _asset_uri("zine/m_angel.png")
    hands = _asset_uri("zine/m_hands.png")
    angel_img = (f'<img class="hero-angel" src="{angel}" alt="">' if angel else "")
    hands_img = (f'<div class="grid-art"><img src="{hands}" alt=""></div>' if hands else "")

    return f"""<!doctype html>
<html data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAUNTLET · {esc(report.run_id)}</title>
<style>{_CSS}</style></head>
<body>
{_SVG_FILTERS}
{_noise_layers()}
<div id="cursor" aria-hidden="true"></div><div id="cursorDot" aria-hidden="true"></div>
<div id="progress" aria-hidden="true"></div>
<div id="hint" aria-hidden="true"><span>T · theme</span><span>F · filter</span>
<span>← → · trials</span><span>ESC · close</span></div>

<nav>
  <span class="wordmark">Gauntlet</span>
  <span class="navlinks lbl"><a href="#trials">the trials</a><a href="#record">the record</a></span>
  <button id="themeBtn"><span id="themeDot"></span><span id="themeLabel">night</span></button>
</nav>

<section class="hero">
  {angel_img}
  <div class="hero-tl">{agg['n']} trials<br>have been run</div>
  <div class="hero-tr">Gauntlet · vol. i</div>
  {marginalia}
  <div class="hero-mark">
    <div class="hero-eyebrow">an evaluation harness for tool-calling agents</div>
    <h1>Gauntlet</h1>
    <div class="hero-foot"><span>{agg['n']} trials</span><span>{n_graders} graders</span>
      <span>every step recorded</span></div>
  </div>
  <div class="cue"><span>scroll · the record</span><i>↓</i></div>
</section>

<section id="record">
  <div class="inner premise">
    <div data-rise="1">
      <div class="lbl" style="letter-spacing:.34em">the premise</div>
      <h2>Fifty scenarios, a sealed world, and tools that fail on purpose. The agent is not asked
        how it would behave. It is made to behave, and the trajectory is kept.</h2>
    </div>
    <div data-rise="1" class="col2">
      <p>Every scenario runs against a fresh copy of the world. Faults are injected on a schedule —
        timeouts, garbage payloads, rate limits — so recovery is measured, not assumed.</p>
      <p>Deterministic graders score tool selection, arguments, efficiency, recovery, world state
        and the final answer. A judge model is consulted last, and only where code cannot decide.</p>
      <div class="chips">{tool_chips}</div>
    </div>
  </div>
  <div class="stats">{stats_html}</div>
</section>

<section id="points"><div class="inner">
  <h2 class="head">Where it loses points</h2>
  <div class="cats" id="cats">{cat_rows}</div>
</div></section>

<section id="trials"><div class="inner">
  <div class="trials-head">
    <h2>The Trials</h2>
    <div class="lbl" style="letter-spacing:.28em">select a trial to unseal its record</div>
  </div>
  <div class="filters">
    <input id="q" placeholder="search the record…" autocomplete="off">
    <span id="shown"></span>
    {filter_buttons}
  </div>
  <div id="detail" hidden></div>
  <div class="grid-wrap">
    {hands_img}
    <div id="grid"></div>
  </div>
</div></section>

<footer>
  <div class="fname">Daiv Raval</div>
  <div class="flinks">
    <a href="https://github.com/daivraval">github.com/daivraval</a>
    <a href="https://www.linkedin.com/in/daiv-raval">linkedin.com/in/daiv-raval</a>
    <a href="tel:+919484446112">+91 94844 46112</a>
  </div>
  <div class="frun">Gauntlet · run {esc(report.run_id)}</div>
</footer>

<script>{_JS.replace("__DATA__", data_json)}</script>
</body></html>"""
