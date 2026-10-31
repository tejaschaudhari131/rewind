"""Regression tests for the 2026-10-31 red-team honesty pass."""
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.two_stage import diagnose_stage1_unknown


def test_fixture_pass_is_provisional_fixture_only():
    with TestClient(app) as c:
        c.post("/api/session/reset", json={})
        d = c.post("/api/checkpoint/verify", json={"scenario": "pass_step2", "mode": "fixture"}).json()
        assert d["fixture_only"] is True and d["stage2"] == "not_run"
        if d["decision"] == "accepted":
            assert d["provisional_status"] == "provisional_local_pass"
            assert "fixture only" in d["verdict_label"]


def test_timeout_elapsed_is_labelled_simulated():
    with TestClient(app) as c:
        c.post("/api/session/reset", json={})
        d = c.post("/api/checkpoint/verify", json={"scenario": "timeout", "mode": "fixture"}).json()
        assert d["decision"] == "unknown" and d["elapsed_is_simulated"] is True


def test_fixture_repair_card_is_deterministic_overlay():
    with TestClient(app) as c:
        d = c.post("/api/bonus/repair-card", json={"mode": "fixture"}).json()
        assert d["measured"] is False and d["status"] == "deterministic_overlay"
        assert "muse-image" not in d["model"] and d["latency_ms"] != 14.2


def test_flat_scan_uncertainty_is_declared_unvalidated():
    with TestClient(app) as c:
        d = c.post("/api/metrology/flat-scan", json={}).json()["flat_part_MeasurementResult"]
        assert d["uncertainty_status"] == "declared_unvalidated"


def test_stage1_diagnostics_classifies_causes():
    assert diagnose_stage1_unknown([], [{"status": 404}, {"status": 404}], 0)["cause"].startswith("sam_404")
    assert "transport" in diagnose_stage1_unknown([], [{"status": "transport:ConnectError"}], 0)["cause"]
    assert diagnose_stage1_unknown(["UNKNOWN_TIMEOUT"], [], 0)["cause"] == "stage1_deadline_exceeded"
