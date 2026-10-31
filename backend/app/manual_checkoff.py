"""Muse Wonder manual check-off: a real computer-use action on a local HTML checklist.

Flow: render checklist -> headless Chrome screenshot -> Wonder `run_cua_step` -> execute the returned click
-> re-read the DOM and verify ONLY the target step's checkbox flipped to checked. Anything else (no model call,
no click, click missed, wrong box) => status `not_updated`, UI label "Manual not updated". Never faked.
"""
from __future__ import annotations

import html
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checkpoints import CheckpointSet

logger = logging.getLogger("rewind.manual_checkoff")
VIEW_W, VIEW_H = 1200, 680
CHROME = "/usr/bin/google-chrome"
NOT_UPDATED = "Manual not updated"


def render_manual_html(cs: CheckpointSet, step_titles: Optional[Dict[int, str]] = None) -> str:
    titles = {c.step_number: c.title for c in cs.steps}
    titles.update(step_titles or {})
    rows = []
    for n in range(1, cs.total_steps + 1):
        t = html.escape(titles.get(n, f"Step {n}"))
        rows.append(f'<label class="row" for="step-{n}"><input type="checkbox" id="step-{n}" data-step="{n}">'
                    f'<span class="num">Step {n}</span><span class="t">{t}</span></label>')
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Assembly manual checklist</title><style>"
            "body{font-family:sans-serif;margin:24px;background:#fff}h1{font-size:22px}"
            ".row{display:flex;align-items:center;gap:14px;padding:10px 12px;border-bottom:1px solid #ddd;font-size:18px}"
            "input{width:28px;height:28px}.num{font-weight:bold;width:80px}</style></head><body>"
            f"<h1>Assembly manual &middot; checklist ({cs.total_steps} steps)</h1>" + "".join(rows) + "</body></html>")


def _clicks(calls: List[dict]) -> List[tuple[int, int]]:
    out = []
    for c in calls:
        acts = c.get("actions") or ([c["action"]] if isinstance(c.get("action"), dict) else [])
        for a in acts:
            if a.get("type") in ("click", "double_click", "left_click") and "x" in a and "y" in a:
                out.append((int(a["x"]), int(a["y"])))
    return out


def _only_screenshot(calls: List[dict]) -> Optional[dict]:
    for c in calls:
        acts = c.get("actions") or ([c["action"]] if isinstance(c.get("action"), dict) else [])
        if acts and all(a.get("type") == "screenshot" for a in acts):
            return c
    return None


async def run_checkoff(cs: CheckpointSet, step_number: int, *, mode: str = "fixture", wonder: Any = None,
                       out_dir: Path = Path("/tmp/rewind_manual"), max_turns: int = 3) -> Dict[str, Any]:
    result: Dict[str, Any] = {"step_number": step_number, "mode": mode, "status": "not_updated",
                              "label": NOT_UPDATED, "measured": False, "clicks_executed": [], "turns": []}
    if mode != "live":
        result["reason"] = "fixture_mode_no_wonder_call"
        return result
    from playwright.async_api import async_playwright
    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = out_dir / "manual.html"
    page_path.write_text(render_manual_html(cs), encoding="utf-8")
    if wonder is None:
        from .providers.meta_wonder import MetaWonderClient
        wonder = MetaWonderClient()
    task = (f"This is an assembly manual checklist. Step {step_number} has just been verified as complete. "
            f"Click the checkbox for 'Step {step_number}' to tick it. Do not click any other checkbox. "
            f"The current screenshot ({VIEW_W}x{VIEW_H}) is attached; act on it directly with a click.")
    t0 = time.perf_counter()
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, headless=True)
        try:
            page = await browser.new_page(viewport={"width": VIEW_W, "height": VIEW_H})
            await page.goto(page_path.as_uri())
            before = await page.eval_on_selector_all("input[type=checkbox]", "els=>els.map(e=>e.checked)")
            shot = await page.screenshot()
            (out_dir / "before.png").write_bytes(shot)
            prev_id = call_id = receipt = None
            for turn in range(max_turns):
                r, last_exc = None, None
                for _attempt in range(2):  # one retry for transient proxy/network errors
                    try:
                        r = await wonder.run_cua_step(screenshot_source=shot, task_prompt=task, display_width=VIEW_W,
                                                  display_height=VIEW_H, raw_instruction=True,
                                                  previous_response_id=prev_id, call_id=call_id,
                                                  safety_replay_receipt=receipt)
                        break
                    except Exception as exc:
                        last_exc = exc
                if r is None:
                    result["reason"] = "wonder_error"
                    result["error"] = f"{type(last_exc).__name__}: {last_exc}"[:300]
                    break
                result["measured"] = True
                result["turns"].append({"response_id": r.provider_response_id, "model": r.model_used,
                                        "latency_ms": r.latency_ms, "computer_calls": r.computer_calls,
                                        "text": r.text_summary[:300], "usage": r.usage})
                clicks = _clicks(r.computer_calls)
                if clicks:
                    for x, y in clicks:
                        await page.mouse.click(x, y)
                        result["clicks_executed"].append({"x": x, "y": y})
                    break
                sid = _only_screenshot(r.computer_calls)
                if not sid:
                    result["reason"] = "wonder_returned_no_click"
                    break
                prev_id, call_id = r.provider_response_id, sid.get("call_id")
                receipt = sid.get("meta_safety_replay_receipt")
            after = await page.eval_on_selector_all("input[type=checkbox]", "els=>els.map(e=>e.checked)")
            (out_dir / "after.png").write_bytes(await page.screenshot())
        finally:
            await browser.close()
    changed = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b]
    result.update(before=before, after=after, changed_steps=changed,
                  wall_ms=round((time.perf_counter() - t0) * 1000, 1),
                  screenshots={"before": str(out_dir / "before.png"), "after": str(out_dir / "after.png")})
    if changed == [step_number] and after[step_number - 1]:
        result.update(status="updated", label=f"Manual updated: Step {step_number} ticked by Muse Wonder (verified in DOM)")
        result.pop("reason", None)
    else:
        result.setdefault("reason", "checkbox_not_changed" if not changed else "wrong_checkbox_changed")
        logger.warning("manual check-off not updated: %s", {k: result.get(k) for k in ("reason", "changed_steps", "clicks_executed")})
    return result
