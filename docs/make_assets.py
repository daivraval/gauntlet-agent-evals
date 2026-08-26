"""Regenerates the README images in docs/assets/ and the demo page docs/index.html.

Run from the project root:

    python docs/make_assets.py

It runs the offline demo suite (no API key, no cost), then saves REAL
terminal output as SVG images using rich's recording feature, and writes the
same run out as the browsable report at docs/index.html. Nothing here is
mocked up: if the code changes, rerun this and every picture and the demo
page update to match.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "docs" / "assets"

from gauntlet import report as report_mod  # noqa: E402
from gauntlet.config import Config  # noqa: E402
from gauntlet.loader import load_scenarios  # noqa: E402
from gauntlet.runner import run_suite  # noqa: E402


def fresh_console(width: int) -> Console:
    console = Console(record=True, width=width, force_terminal=True)
    report_mod.console = console
    return console


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    config = Config()
    with tempfile.TemporaryDirectory() as tmp:
        report, _ = asyncio.run(run_suite(
            scenarios, agent_name="scripted", config=config,
            judge_enabled=False, out_root=Path(tmp), quiet=True,
        ))

    # 1. the scorecard, exactly as the terminal shows it
    console = fresh_console(width=100)
    report_mod.print_summary(report)
    console.save_svg(str(ASSETS / "scorecard.svg"),
                     title="gauntlet · full 50 scenario run (offline demo)")

    # 2. one trajectory replay with the injected fault visible
    console = fresh_console(width=110)
    result = next(r for r in report.results if r.scenario_id == "ER-01")
    report_mod.print_trajectory(result)
    console.save_svg(str(ASSETS / "trajectory.svg"),
                     title="gauntlet show · scenario ER-01 (a sabotaged tool, then the retry)")

    # 3. the report itself, as the landing page for GitHub Pages. It is a
    # single self-contained file (art is inlined), so it needs no server.
    page = ROOT / "docs" / "index.html"
    page.write_text(report_mod.to_html(report), encoding="utf-8")

    print(f"wrote {ASSETS / 'scorecard.svg'}")
    print(f"wrote {ASSETS / 'trajectory.svg'}")
    print(f"wrote {page} ({page.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
