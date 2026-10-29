"""Two-stage gate: provisional (stage 1) -> async model confirmation (stage 2). Fail closed throughout."""
import asyncio
from pathlib import Path

import pytest

from backend.app.providers.modal_glimmer import GlimmerAuditResult
from backend.app.state_machine import SessionStateMachine
from backend.app.two_stage import final_from_stage2, run_two_stage_gate

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "golden_rig"
PASS = FIX / "checkpoint_pass_step2.png"
FLIP = FIX / "checkpoint_fail_flipped_rod.png"


class FakeAuditor:
    def __init__(self, finding, delay=0.0, raise_exc=None):
        self.finding, self.delay, self.raise_exc = finding, delay, raise_exc
        self.calls = 0

    async def audit_checkpoint(self, *, current_image, reference_image, blinded_context):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        obs = blinded_context["observation_id"]
        return GlimmerAuditResult(
            model_used="meta-models/muse-spark-open-1026",
            audit={"observation_id": obs, "finding": self.finding,
                   "observations": [{"frame_id": blinded_context["frame_id"], "region_xyxy_px": [0, 0, 1, 1],
                                     "statement": f"fake says {self.finding}"}],
                   "contradicted_predicate_ids": [], "requires_new_view": False},
            latency_ms=12.0,
            usage={"prompt_tokens": 475, "completion_tokens": 800, "request_id": "req-fake",
                   "raw_verdict": {"finding": self.finding, "statement": f"fake says {self.finding}", "confidence": 0.9}},
        )


class FakeArbiter:
    def __init__(self, verdict="MISMATCH"):
        self.verdict = verdict

    async def arbitrate_ambiguous_checkpoint(self, **kw):
        return {"triggered": True, "verdict": self.verdict, "consensus": "X", "latency_ms": 5.0, "measured": True}


def _sm():
    sm = SessionStateMachine()
    sm.current_step_index = 1
    return sm


@pytest.mark.asyncio
async def test_agree_pass_confirms_accept():
    sm = _sm()
    s1, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent"),
                                        arbiter_client=FakeArbiter())
    assert s1["final"] is False and s1["decision"] == "pending"
    assert s1["provisional_verdict"] == "provisional_accepted"
    assert sm.state == "CHECKING"  # stage 1 never commits
    r = await task
    assert r["decision"] == "accepted" and r["resolution"] == "confirmed" and r["status"] == "committed"
    assert r["auditor"]["request_id"] == "req-fake" and r["auditor"]["prompt_tokens"] == 475
    assert sm.state == "ACCEPTED"


@pytest.mark.asyncio
async def test_agree_flip_confirms_reject():
    sm = _sm()
    s1, task = await run_two_stage_gate(FLIP, state_machine=sm, audit_client=FakeAuditor("mismatch"),
                                        arbiter_client=FakeArbiter("MISMATCH"))
    assert s1["provisional_verdict"] == "provisional_rejected"
    r = await task
    assert r["decision"] == "rejected" and r["resolution"] == "confirmed"
    assert r["model_reason"] == "fake says mismatch"


@pytest.mark.asyncio
async def test_disagree_model_overrides_provisional_accept_to_needs_review():
    sm = _sm()
    _, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("mismatch"),
                                       arbiter_client=FakeArbiter())
    r = await task
    assert r["decision"] != "accepted"
    assert r["resolution"] == "overridden" and r["needs_review"] is True
    assert "model_override_needs_review" in r["reason_codes"]
    assert r["model_reason"] == "fake says mismatch"


@pytest.mark.asyncio
async def test_model_consistent_never_auto_accepts_a_provisional_reject():
    sm = _sm()
    _, task = await run_two_stage_gate(FLIP, state_machine=sm, audit_client=FakeAuditor("consistent"),
                                       arbiter_client=FakeArbiter("CONSISTENT"))
    r = await task
    assert r["decision"] == "unknown" and r["needs_review"] is True
    assert sm.state != "ACCEPTED"


@pytest.mark.asyncio
async def test_stage2_timeout_stays_unknown():
    sm = _sm()
    _, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent", delay=1.0),
                                       stage2_deadline_ms=100)
    r = await task
    assert r["decision"] == "unknown" and "stage2_timeout" in r["reason_codes"]
    assert sm.state == "UNKNOWN"


@pytest.mark.asyncio
async def test_stage2_error_and_insufficient_stay_unknown():
    for aud in (FakeAuditor("consistent", raise_exc=RuntimeError("boom")), FakeAuditor("insufficient_evidence")):
        sm = _sm()
        _, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=aud)
        r = await task
        assert r["decision"] == "unknown"


@pytest.mark.asyncio
async def test_late_result_after_reset_is_discarded():
    sm = _sm()
    _, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent", delay=0.2))
    sm.retire()  # /api/session/reset
    r = await task
    assert r["status"] == "discarded_stale" and r["final"] is False and r["decision"] == "unknown"
    assert sm.state != "ACCEPTED"


@pytest.mark.asyncio
async def test_late_result_after_camera_move_is_discarded():
    sm = _sm()
    _, task = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent", delay=0.2))
    sm.evaluate_stationarity(motion_score=50.0)
    r = await task
    assert r["status"] == "discarded_stale" and sm.state != "ACCEPTED"


@pytest.mark.asyncio
async def test_superseded_observation_is_discarded():
    sm = _sm()
    _, t1 = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent", delay=0.3))
    _, t2 = await run_two_stage_gate(PASS, state_machine=sm, audit_client=FakeAuditor("consistent"))
    r2, r1 = await t2, await t1
    assert r1["status"] == "discarded_stale" and r2["status"] == "committed"


@pytest.mark.asyncio
async def test_fixture_mode_without_model_never_finalizes_accept():
    sm = _sm()
    _, task = await run_two_stage_gate(PASS, state_machine=sm, mode="fixture")
    r = await task
    assert r["decision"] == "unknown"


def test_final_rules_never_auto_accept():
    for prov in ("provisional_accepted", "provisional_rejected", "unknown"):
        for f in (None, "insufficient_evidence", "mismatch", "consistent"):
            d, *_ = final_from_stage2(prov, f)
            if d == "accepted":
                assert prov == "provisional_accepted" and f == "consistent"


def test_geometry_reports_measured_cosine_no_canned_constant():
    from backend.app.geometry import evaluate_elevated_checkpoint
    import inspect, backend.app.geometry as g
    assert "-0.994" not in inspect.getsource(g)
    m = evaluate_elevated_checkpoint(FLIP)
    assert m.witness_check_status in ("measured", "insufficient_evidence")
    if m.witness_check_status == "insufficient_evidence":
        assert m.witness_polarity_cosine is None and m.witness_aligned is False


def test_api_two_stage_ws_push():
    from fastapi.testclient import TestClient
    from backend.app import main
    client = TestClient(main.app)
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        r = client.post("/api/checkpoint/verify", json={"scenario": "pass_step2", "mode": "fixture", "two_stage": True})
        body = r.json()
        assert body["stage"] == 1 and body["final"] is False
        events = [ws.receive_json()["event"] for _ in range(2)]
        assert "checkpoint_provisional" in events
        obs = body["observation_id"]
        res = client.get(f"/api/checkpoint/stage2/{obs}", params={"wait_ms": 5000}).json()
        assert res["stage"] == 2 and res["decision"] == "unknown"
