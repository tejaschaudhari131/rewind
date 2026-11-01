"""Capture the three v3 looks (?look=a|b|c) at 1920x1080 against a running server (default :8077).
All states are real fixture responses. CHECKING is captured by delaying the verify route in Playwright (no fake latency in the UI)."""
import asyncio, sys
from pathlib import Path
from playwright.async_api import async_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
OUT = Path(__file__).resolve().parent.parent / "docs/screenshots/ui_v3"
LOOKS = {"a": "paper", "b": "hud", "c": "poster"}
STATES = [("01_stop_holes_out", "ikea_fail_holes_out", "stop"), ("02_provisional_fixture", "ikea_pass_step3", "provisional"),
          ("03_unknown_blur", "ikea_unmeasurable_blur", "unknown")]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        for look, name in LOOKS.items():
            d = OUT / f"{look}_{name}"; d.mkdir(parents=True, exist_ok=True)
            for fn, sc, ph in STATES:
                pg = await b.new_page(viewport={"width": 1920, "height": 1080})
                await pg.goto(f"{BASE}/?look={look}&scenario={sc}"); await pg.wait_for_timeout(600)
                await pg.keyboard.press("Space")
                await pg.wait_for_selector(f'[data-phase="{ph}"]'); await pg.wait_for_timeout(1400)
                await pg.screenshot(path=str(d / f"{fn}.png")); await pg.close()
            pg = await b.new_page(viewport={"width": 1920, "height": 1080})
            async def slow(route):
                await asyncio.sleep(5); await route.continue_()
            await pg.route("**/api/furniture/checkpoint/verify", slow)
            await pg.goto(f"{BASE}/?look={look}&scenario=ikea_fail_holes_out"); await pg.wait_for_timeout(600)
            await pg.keyboard.press("Space"); await pg.wait_for_timeout(900)
            await pg.screenshot(path=str(d / "04_checking_route_delayed.png")); await pg.close()
            print("captured", d)
        await b.close()
asyncio.run(main())
