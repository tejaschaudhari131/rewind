"""Capture REWIND v2 UI states at 1920x1080 into docs/screenshots/ui_v2/.
Usage: python3 scripts/capture_ui_v2.py [base_url]. All states use the real fixture API, except
'step_complete_UI_MOCK.png', whose response is injected by Playwright and visibly labelled UI MOCK
(no accepted live run exists yet on synthetic frames)."""
import asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8077"
OUT = Path("docs/screenshots/ui_v2"); OUT.mkdir(parents=True, exist_ok=True)
MOCK_ACCEPT = {
    "stage1": {"provisional_verdict": "provisional_accepted", "provisional_reason_codes": [], "panel_silhouette_iou": None,
               "hole_pattern_witness": {"status": "pass", "reason": "hole_face_matches_reference", "reference_hole_count": None, "live_hole_count": None, "live_holes_px": [], "producer": "UI_MOCK"},
               "tape_edge_witness": {"status": "pass", "reason": "tape_edge_matches_reference", "polarity_cosine": None, "producer": "UI_MOCK"},
               "sam": {"called": False, "measured": False, "concept_prompt": "wooden panel"}, "stage1_elapsed_ms": None},
    "stage2": {"measured": False}, "decision": "accepted", "mode": "fixture", "fixture_only": True,
    "manual": {"status": "updated", "measured": False, "label": "UI MOCK: tick animation preview, not a real Muse Wonder run", "reason": "ui_mock_screenshot_only"},
}
async def shot(p, name, scenario_btn=None, check=True, delay=None, mock=None, wait=1600):
    b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = await b.new_page(viewport={"width": 1920, "height": 1080})
    if delay or mock:
        async def handle(route):
            if mock: await route.fulfill(status=200, content_type="application/json", body=json.dumps(mock)); return
            await asyncio.sleep(delay); await route.continue_()
        await pg.route("**/api/furniture/checkpoint/verify", handle)
    await pg.goto(BASE + "/"); await pg.wait_for_timeout(1200)
    if scenario_btn: await pg.get_by_role("button", name=scenario_btn).click()
    if check: await pg.get_by_role("button", name="Check this step").click()
    await pg.wait_for_timeout(wait)
    if mock:
        await pg.evaluate("""() => { const d=document.createElement('div');
          d.textContent='UI MOCK · injected response · not a real run';
          d.style.cssText='position:fixed;left:50%;top:46%;transform:translate(-50%,-50%) rotate(-8deg);font:900 64px system-ui;color:rgba(251,191,36,.9);border:6px solid rgba(251,191,36,.9);padding:12px 28px;border-radius:18px;background:rgba(2,6,23,.55);z-index:99;pointer-events:none';
          document.body.appendChild(d); }""")
    await pg.screenshot(path=str(OUT / name)); await b.close()
async def main():
    async with async_playwright() as p:
        await shot(p, "01_ready.png", check=False, wait=300)
        await shot(p, "02_checking.png", "Flipped: holes out", delay=5, wait=700)
        await shot(p, "03_stop_holes_out.png", "Flipped: holes out")
        await shot(p, "04_provisional_holes_in_fixture.png", "Holes facing in")
        await shot(p, "05_cant_tell_blur.png", "Dim / blurred")
        await shot(p, "06_step_complete_UI_MOCK.png", "Holes facing in", mock=MOCK_ACCEPT, wait=2200)
asyncio.run(main())
