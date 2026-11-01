"""Final captures: default look A (no param), A at 1280 wide, A OK-state UI MOCK (stamped; never product footage),
B in record mode, and the C STOP image for X at 1600x900. Real fixture responses except the stamped mock."""
import asyncio, json, sys, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "docs/screenshots/ui_v3/a_paper"
STAMP = """const d=document.createElement('div');d.textContent='UI MOCK · not a real run';
d.style.cssText='position:fixed;left:50%;top:46%;transform:translateX(-50%) rotate(-8deg);z-index:9999;font:900 44px sans-serif;color:#d71920;border:6px solid #d71920;padding:6px 18px;background:rgba(255,255,255,.85)';document.body.appendChild(d);"""

def real_pass():
    req = urllib.request.Request(f"{BASE}/api/furniture/checkpoint/verify", method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"scenario": "ikea_pass_step3", "mode": "fixture", "step_id": "step-3-left-side-panel", "check_off_manual": True}).encode())
    return json.load(urllib.request.urlopen(req))

async def shot(b, url, path, w=1920, h=1080, phase=None, route=None, stamp=False):
    pg = await b.new_page(viewport={"width": w, "height": h})
    if route: await pg.route("**/api/furniture/checkpoint/verify", route)
    await pg.goto(BASE + url); await pg.wait_for_timeout(600)
    if phase:
        await pg.keyboard.press("Space"); await pg.wait_for_selector(f'[data-phase="{phase}"]'); await pg.wait_for_timeout(1400)
    if stamp: await pg.evaluate(STAMP)
    await pg.screenshot(path=str(path)); await pg.close(); print(path.relative_to(ROOT))

async def main():
    m = real_pass()  # start from a real fixture response, then mark it as a mock pass
    m.update(decision="accepted", mode="live", ui_mock=True)
    m["stage2"] = {**m.get("stage2", {}), "measured": True, "wall_ms": None, "model_finding": "UI_MOCK", "error": None}
    m["manual"] = {**m.get("manual", {}), "status": "updated", "label": "UI MOCK: manual ticked", "measured": True, "wall_ms": None}
    async def mock(route): await route.fulfill(status=200, content_type="application/json", body=json.dumps(m))
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        await shot(b, "/?scenario=ikea_fail_holes_out", A / "00_default_stop.png", phase="stop")
        await shot(b, "/?scenario=ikea_pass_step3", A / "00_default_provisional.png", phase="provisional")
        await shot(b, "/?scenario=ikea_fail_holes_out", A / "05_stop_1280w.png", 1280, 800, phase="stop")
        await shot(b, "/?scenario=ikea_pass_step3", A / "06_ok_step_complete_UI_MOCK.png", phase="ok", route=mock, stamp=True)
        await shot(b, "/?look=b&record=1&scenario=ikea_fail_holes_out", ROOT / "docs/screenshots/ui_v3/b_hud/05_record_mode_stop.png", phase="stop")
        await shot(b, "/?look=c&record=1&scenario=ikea_fail_holes_out", ROOT / "docs/launch/x_post_image_c.png", 1600, 900, phase="stop")
        await b.close()
asyncio.run(main())
