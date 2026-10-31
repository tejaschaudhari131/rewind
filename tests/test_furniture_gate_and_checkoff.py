import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.checkpoints import default_checkpoint_set
from backend.app.furniture_gate import provisional_from_witnesses, verify_furniture_checkpoint
from backend.app.main import app
from backend.app.manual_checkoff import _clicks, render_manual_html, run_checkoff
from backend.app.providers.baseten_vision import AUDIT_PROMPTS
from backend.app.semantic_validator import SemanticValidationError, assert_glimmer_input_blinded

CS = default_checkpoint_set()
CFG = CS.step()


def run(c):
    return asyncio.get_event_loop().run_until_complete(c) if False else asyncio.run(c)


@pytest.mark.parametrize("scenario,prov,decision", [
    ("ikea_pass_step3", "provisional_accepted", "unknown"),
    ("ikea_fail_holes_out", "provisional_rejected", "rejected"),
    ("ikea_unmeasurable_blur", "unknown", "unknown"),
])
def test_fixture_gate_never_auto_accepts(scenario, prov, decision):
    out = run(verify_furniture_checkpoint(CS.scenarios[scenario], CFG, mode="fixture"))
    assert out["stage1"]["provisional_verdict"] == prov
    assert out["decision"] == decision
    assert out["fixture_only"] is True and out["stage2"]["measured"] is False


def test_provisional_rules_fail_closed():
    p = {"status": "pass", "check": "h"}
    i = {"status": "insufficient_evidence", "check": "t", "reason": "x"}
    assert provisional_from_witnesses(None, p, p, 0.8)[0] == "unknown"
    assert provisional_from_witnesses(0.9, p, i, 0.8)[0] == "unknown"
    assert provisional_from_witnesses(0.5, p, p, 0.8)[0] == "provisional_rejected"


class FakeSam:
    def __init__(self, n=1, exc=None):
        self.n, self.exc, self.prompts = n, exc, []

    async def segment_image(self, image, prompt_text, part_id):
        self.prompts.append(prompt_text)
        if self.exc:
            raise self.exc
        return SimpleNamespace(instances=[object()] * self.n, latency_ms=5.0)


class FakeAuditor:
    def __init__(self, finding):
        self.finding, self.profile = finding, None

    async def audit_checkpoint(self, current_image, reference_image, blinded_context, audit_profile):
        assert_glimmer_input_blinded(blinded_context)
        self.profile = audit_profile
        return SimpleNamespace(audit={"finding": self.finding}, usage={"total_tokens": 10}, latency_ms=12.0)


def test_live_uses_config_concept_prompt_and_both_stages_must_agree():
    sam, aud = FakeSam(), FakeAuditor("consistent")
    out = run(verify_furniture_checkpoint(CS.scenarios["ikea_pass_step3"], CFG, mode="live", sam_client=sam, auditor=aud))
    assert sam.prompts == ["wooden panel"] and aud.profile == "furniture_panel"
    assert out["decision"] == "accepted"
    out = run(verify_furniture_checkpoint(CS.scenarios["ikea_pass_step3"], CFG, mode="live", sam_client=FakeSam(),
                                          auditor=FakeAuditor("mismatch")))
    assert out["decision"] != "accepted"


def test_live_sam_failure_is_unknown_not_accept():
    out = run(verify_furniture_checkpoint(CS.scenarios["ikea_pass_step3"], CFG, mode="live",
                                          sam_client=FakeSam(exc=RuntimeError("boom")), auditor=FakeAuditor("consistent")))
    assert out["stage1"]["provisional_verdict"] == "unknown"
    assert out["decision"] != "accepted"
    out = run(verify_furniture_checkpoint(CS.scenarios["ikea_pass_step3"], CFG, mode="live",
                                          sam_client=FakeSam(n=0), auditor=FakeAuditor("consistent")))
    assert out["decision"] != "accepted"


def test_furniture_prompt_is_neutral_and_blinding_catches_fixture_names():
    p = AUDIT_PROMPTS["furniture_panel"].lower()
    for leak in ("flipped", "wrong", "holes facing out", "should fail"):
        assert leak not in p
    with pytest.raises(SemanticValidationError):
        assert_glimmer_input_blinded({"frame_id": "checkpoint_fail_holes_out"})


def test_manual_html_and_click_parsing():
    h = render_manual_html(CS)
    assert h.count('type="checkbox"') == CS.total_steps
    assert _clicks([{"actions": [{"type": "click", "x": 40, "y": 100}]}, {"action": {"type": "screenshot"}}]) == [(40, 100)]


def test_checkoff_fixture_mode_reports_not_updated():
    r = run(run_checkoff(CS, 3, mode="fixture"))
    assert r["status"] == "not_updated" and r["label"] == "Manual not updated" and r["measured"] is False


class FakeWonder:
    def __init__(self, xy):
        self.xy = xy

    async def run_cua_step(self, **kw):
        acts = [{"type": "click", "x": self.xy[0], "y": self.xy[1]}] if self.xy else []
        return SimpleNamespace(provider_response_id="resp_fake", model_used="fake", latency_ms=1.0,
                               computer_calls=[{"type": "computer_call", "call_id": "c1", "actions": acts}],
                               text_summary="", usage={})


def _chrome_available():
    import os
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return os.path.exists("/usr/bin/google-chrome")


@pytest.mark.skipif(not _chrome_available(), reason="playwright/chrome not installed")
def test_checkoff_verifies_dom_change(tmp_path):
    from backend.app.manual_checkoff import render_manual_html  # noqa: F401
    import asyncio as _a
    from playwright.async_api import async_playwright

    async def locate():
        (tmp_path / "m.html").write_text(render_manual_html(CS))
        async with async_playwright() as p:
            b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
            pg = await b.new_page(viewport={"width": 1200, "height": 680})
            await pg.goto((tmp_path / "m.html").as_uri())
            boxes = [await pg.locator(f"#step-{n}").bounding_box() for n in (2, 3)]
            await b.close()
            return [(int(x["x"] + x["width"] / 2), int(x["y"] + x["height"] / 2)) for x in boxes]

    (b2, b3) = _a.run(locate())
    ok = run(run_checkoff(CS, 3, mode="live", wonder=FakeWonder(b3), out_dir=tmp_path))
    assert ok["status"] == "updated" and ok["changed_steps"] == [3]
    wrong = run(run_checkoff(CS, 3, mode="live", wonder=FakeWonder(b2), out_dir=tmp_path))
    assert wrong["status"] == "not_updated" and wrong["reason"] == "wrong_checkbox_changed"
    none = run(run_checkoff(CS, 3, mode="live", wonder=FakeWonder(None), out_dir=tmp_path))
    assert none["status"] == "not_updated" and none["label"] == "Manual not updated"


def test_api_routes():
    c = TestClient(app)
    cfg = c.get("/api/furniture/config").json()
    assert cfg["steps"][0]["label"].startswith("Step 3 of 8") and cfg["steps"][0]["sam_concept_prompt"] == "wooden panel"
    r = c.post("/api/furniture/checkpoint/verify", json={"scenario": "ikea_fail_holes_out", "check_off_manual": True}).json()
    assert r["decision"] == "rejected" and r["manual"]["label"] == "Manual not updated"
    assert c.post("/api/furniture/checkpoint/verify", json={"scenario": "nope"}).status_code == 404
