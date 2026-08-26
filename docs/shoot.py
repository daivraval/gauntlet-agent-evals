"""Re-shoots the README screenshots from the live report.

Usage:
    python -m http.server 8931 --directory docs      # in another shell
    python docs/shoot.py [http://localhost:8931/index.html]

Every shot is taken at the exact pixel size it ships at, so the README never
has to scale one down. The hero is captured at the full height of the section
(1480px) rather than a viewport height, because the angel hangs to 1422px and
a shorter frame crops its feet - which is the bug these shots exist to show
is gone.

Needs selenium and a local Chrome; Selenium Manager fetches the driver.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8931/index.html"
OUT = Path(__file__).resolve().parent / "assets"

NAV = 62  # the fixed masthead, which must not sit on top of a section heading

# Interaction affordances. They live at the bottom of the viewport and would
# print on top of the wordmark in a still, where they read as artefacts rather
# than as the live hints they are.
HIDE_HUD = """
#hint, #cursor, #cursorDot, .cue { display: none !important; }
"""


def driver():
    opts = Options()
    for flag in ("--headless=new", "--hide-scrollbars", "--disable-gpu",
                 "--force-device-scale-factor=1", "--font-render-hinting=none"):
        opts.add_argument(flag)
    return webdriver.Chrome(options=opts)


def viewport(d, w, h):
    """Exact CSS viewport. Window sizing alone leaves room for browser chrome."""
    d.execute_cdp_cmd("Emulation.setDeviceMetricsOverride",
                      {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False})


def settle(d, seconds=1.4):
    d.execute_script("return document.fonts.ready")
    time.sleep(seconds)


def shoot(d, name, w, h, *, top=0.0, crop=None):
    viewport(d, w, h)
    d.execute_script("window.scrollTo(0, arguments[0]);", top)
    settle(d, 0.6)
    path = OUT / name
    d.save_screenshot(str(path))
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
        if im.size != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
    im.save(path, optimize=True)
    print(f"  {name:24} {im.size[0]}x{im.size[1]}  {path.stat().st_size // 1024}KB")
    return im


def section_top(d, selector):
    return d.execute_script(
        "var e=document.querySelector(arguments[0]);"
        "return e.getBoundingClientRect().top + window.scrollY - arguments[1];",
        selector, NAV)


def main():
    d = driver()
    try:
        viewport(d, 1920, 1480)
        d.get(URL)
        d.execute_script(
            "var s=document.createElement('style');s.textContent=arguments[0];"
            "document.head.appendChild(s);", HIDE_HUD)
        settle(d, 2.0)

        print("dark:")
        # The whole hero: angel uncropped, wordmark seated beneath it.
        hero = shoot(d, "report-hero.png", 1920, 1480)

        # The banner is the wordmark band lifted straight out of the hero, so
        # the two can never drift apart. Cropped 1:1 at the shipped height
        # rather than resized, so the letterforms keep the page's proportions.
        # .hero-mark carries a fixed height its contents overflow, so the band
        # is measured from the strapline that actually sits lowest.
        foot = d.execute_script(
            "return document.querySelector('.hero-foot').getBoundingClientRect().bottom;")
        bot = min(1480, int(foot) + 12)
        crop = hero.crop((0, bot - 350, 1920, bot))
        crop.save(OUT / "report-banner.png", optimize=True)
        print(f"  {'report-banner.png':24} 1920x350  "
              f"{(OUT / 'report-banner.png').stat().st_size // 1024}KB")

        # 960 rather than the old 800: the section runs 865px and the summary
        # statistics sit on its last row, which the shorter frame sliced off.
        shoot(d, "report-record.png", 1780, 960, top=section_top(d, "#record"))
        shoot(d, "report-categories.png", 1180, 440, top=section_top(d, "#points"))

        viewport(d, 1780, 1100)
        d.execute_script("window.scrollTo(0, arguments[0]);", section_top(d, "#trials"))
        settle(d, 0.6)
        shoot(d, "report-trials.png", 1780, 1100, top=section_top(d, "#trials"))

        # One trial unsealed. ER-01 is the retry story the README describes.
        d.execute_script(
            "var c=[].find.call(document.querySelectorAll('#grid .card'),"
            "  function(e){return /ER-01/.test(e.textContent)}) "
            "  || document.querySelector('#grid .card');"
            "c.click();")
        settle(d, 0.8)
        shoot(d, "report-detail.png", 1780, 790, top=section_top(d, "#detail"))

        print("day:")
        # Reseal the record; the day shot is of the grid, not the detail panel.
        d.execute_script("var b=document.getElementById('close');if(b)b.click();")
        settle(d, 0.5)
        d.find_element("id", "themeBtn").click()
        settle(d, 1.0)
        shoot(d, "report-day.png", 1780, 1100, top=section_top(d, "#trials"))
    finally:
        d.quit()


if __name__ == "__main__":
    main()
