"""Regression tests for live-run honesty fixes (2026-10-29 live Baseten vision probe)."""
import pytest
from backend.app.telemetry import build_six_model_proof_ledger
from backend.app.providers import baseten_spark as bs
from backend.app.providers.baseten_spark import BasetenSparkClient, SessionSummaryResult


def test_ledger_reports_measured_geometry_verbatim_not_canned():
    led = build_six_model_proof_ledger(scenario="flipped", include_repair_image=True,
                                       mask_iou=0.81, witness_polarity_cosine=0.42).to_dict()
    c = led["dual_open_weight_consensus"]
    assert c["silhouette_iou"] == 0.81 and c["witness_polarity_cosine"] == 0.42
    assert led["all_live_verified"] is False
    assert led["invoked_lanes_live_verified"] is False  # nothing live was invoked


@pytest.mark.asyncio
async def test_arbiter_not_triggered_reports_zero_latency_and_not_called():
    r = await BasetenSparkClient(baseten_api_key="").arbitrate_ambiguous_checkpoint(
        session_id="s", step_id="x", silhouette_iou=0.5, witness_polarity_cosine=0.9, witness_error_px=1.0,
        glimmer_finding="consistent", mode="fixture")
    assert r["triggered"] is False and r["latency_ms"] == 0.0 and r["measured"] is False


@pytest.mark.asyncio
async def test_arbiter_live_unparseable_fails_closed(monkeypatch):
    async def fake(self, **kw):
        return SessionSummaryResult(provider="baseten", model="m", summary="garbage",
                                    digest={"measured": True, "usage": {}, "request_id": "r"}, latency_ms=5.0)
    monkeypatch.setattr(BasetenSparkClient, "summarize_session", fake)
    r = await BasetenSparkClient(baseten_api_key="k").arbitrate_ambiguous_checkpoint(
        session_id="s", step_id="x", silhouette_iou=0.97, witness_polarity_cosine=-0.9, witness_error_px=40.0,
        glimmer_finding="mismatch", mode="live")
    assert r["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert r["consensus"] != "DUAL_OPEN_WEIGHT_REJECT_180_FLIP"


@pytest.mark.asyncio
async def test_arbiter_live_does_not_hardcode_mismatch(monkeypatch):
    async def fake(self, **kw):
        assert '"MISMATCH", "failure_signature": "ASYMMETRIC' not in kw["question"]  # not leading
        return SessionSummaryResult(provider="baseten", model="m", summary="",
                                    digest={"measured": True, "parsed_json": {"verdict": "CONSISTENT"}}, latency_ms=5.0)
    monkeypatch.setattr(BasetenSparkClient, "summarize_session", fake)
    r = await BasetenSparkClient(baseten_api_key="k").arbitrate_ambiguous_checkpoint(
        session_id="s", step_id="x", silhouette_iou=0.97, witness_polarity_cosine=-0.9, witness_error_px=40.0,
        glimmer_finding="mismatch", mode="live")
    assert r["verdict"] == "CONSISTENT" and r["consensus"] == "REJECT_PARTIAL_AGREEMENT"
