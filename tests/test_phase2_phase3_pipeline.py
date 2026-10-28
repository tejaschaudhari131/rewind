"""Comprehensive Phase 2 & Phase 3 Unit, Integration, and Live E2E Tests (`tests/test_phase2_phase3_pipeline.py`).

Covers:
1. Phase 2 Printable AprilTag Mat & Cutouts (`fixtures/printable/`) and Golden Synthetic Slider-Crank Dataset (`fixtures/golden_rig/`)
2. Plane-Aware Calibration & Metrology (`backend/app/calibration.py` & `backend/app/geometry.py`)
3. Parametric CAD Worker & Independent STEP/STL Verifier (`backend/app/cad_worker.py`)
4. Sub-2.8s Deterministic Gate & Atomic CAS State Machine (`backend/app/gate.py` & `backend/app/state_machine.py`)
5. FastAPI REST + WebSocket Server (`backend/app/main.py`)
6. Live End-to-End Checkpoint Verification with real `sam-3.1` on `checkpoint_pass_step2.png`
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.cad_worker import (
    generate_flat_two_hole_link_v1,
    validate_link_parameters_mm,
)
from backend.app.calibration import (
    build_calibration_envelope,
    compute_planar_calibration,
    rectify_to_mat_z0,
)
from backend.app.gate import (
    run_checkpoint_gate,
    sanitize_blinded_glimmer_context,
)
from backend.app.geometry import (
    build_geometry_envelope,
    build_measurement_record_envelope,
    evaluate_elevated_checkpoint,
    measure_flat_link_on_mat_z0,
)
from backend.app.config import settings
from backend.app.main import app
import socket
from urllib.parse import urlparse

def _skip_if_proxy_unreachable() -> None:
    if not settings.proxy_url:
        return
    parsed = urlparse(settings.proxy_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8791
    try:
        with socket.create_connection((host, port), timeout=0.6):
            pass
    except OSError:
        pytest.skip(f"Live Meta proxy tunnel ({host}:{port}) is currently offline")
    import httpx
    try:
        with httpx.Client(proxy=settings.proxy_url, timeout=1.5) as client:
            r = client.get(f"{settings.meta_api_base}/models", headers=settings.auth_headers())
            if r.status_code != 200 or not r.json().get("data"):
                pytest.skip("Live Meta proxy tunnel is not routing api.meta.ai models endpoint")
    except Exception:
        pytest.skip("Live Meta proxy tunnel is not routing api.meta.ai models endpoint")

from backend.app.schemas import make_envelope, validate_contract_envelope
from backend.app.semantic_validator import (
    SemanticValidationError,
    validate_envelope_semantics,
)
from backend.app.state_machine import SessionStateMachine, compute_frame_motion_score

ROOT = Path("/workspace/rewind")
PRINTABLE_DIR = ROOT / "fixtures" / "printable"
GOLDEN_DIR = ROOT / "fixtures" / "golden_rig"


def test_phase2_printable_and_golden_rig_files_exist() -> None:
    """Verify all Phase 2 printable PDFs/SVG/PNG and golden rig keyframes + MP4 exist and are non-empty."""
    expected_printable = [
        "rewind_apriltag_mat_400x250mm.pdf",
        "rewind_apriltag_mat_400x250mm.svg",
        "rewind_apriltag_mat_400x250mm.png",
        "rewind_apriltag_mat_letter_tiles.pdf",
        "rewind_slider_crank_cutouts_1to1.pdf",
    ]
    for name in expected_printable:
        p = PRINTABLE_DIR / name
        assert p.exists() and p.stat().st_size > 1000, f"Missing or empty printable file: {p}"

    expected_golden = [
        "step_0_complete.png",
        "step_1_no_paddle.png",
        "step_2_no_rod.png",
        "step_3_no_slider_or_crank.png",
        "step_4_base.png",
        "checkpoint_pass_step2.png",
        "checkpoint_fail_flipped_rod.png",
        "checkpoint_fail_camera_bump.png",
        "flat_scan_mat_z0.png",
        "ref_mask_step2_connecting_rod.png",
        "teardown_clip.mp4",
        "golden_metadata.json",
    ]
    for name in expected_golden:
        p = GOLDEN_DIR / name
        assert p.exists() and p.stat().st_size > 500, f"Missing or empty golden rig file: {p}"


def test_calibration_apriltag_detection_and_camera_bump() -> None:
    """Verify APRILTAG_36h11 homography estimation passes on normal frames and flags registration_lost on camera bump."""
    cal_pass = compute_planar_calibration(GOLDEN_DIR / "checkpoint_pass_step2.png")
    assert cal_pass.detected_tag_ids == [0, 1, 2, 3]
    assert cal_pass.calibration_valid is True
    assert cal_pass.registration_rmse_px <= 1.5
    assert cal_pass.reprojection_rmse_px <= 1.0

    env = build_calibration_envelope(cal_pass)
    validate_contract_envelope(env)
    validate_envelope_semantics(env)

    cal_bump = compute_planar_calibration(GOLDEN_DIR / "checkpoint_fail_camera_bump.png")
    assert cal_bump.detected_tag_ids == [0, 1, 2, 3]
    assert cal_bump.calibration_valid is False
    assert cal_bump.registration_rmse_px > 1.5
    assert "registration_lost" in cal_bump.reason_codes

    rectified, _ = rectify_to_mat_z0(GOLDEN_DIR / "flat_scan_mat_z0.png", px_per_mm=4.0)
    assert rectified.shape == (1000, 1600, 3)


def test_geometry_elevated_px_and_flipped_rod_witness_detection() -> None:
    """Verify elevated checkpoint strictly uses image pixels (`px`, `plane_id: None`) and catches 180-deg flipped rod."""
    m_pass = evaluate_elevated_checkpoint(GOLDEN_DIR / "checkpoint_pass_step2.png")
    assert m_pass.unit == "px"
    assert m_pass.coordinate_frame == "camera_undistorted_px"
    assert m_pass.plane_id is None
    assert m_pass.status == "pass"
    assert m_pass.mask_iou >= 0.82
    assert m_pass.witness_error_px <= 6.0
    assert m_pass.pivot_error_px <= 5.0

    env_pass = build_geometry_envelope(m_pass)
    validate_envelope_semantics(env_pass)

    m_flip = evaluate_elevated_checkpoint(GOLDEN_DIR / "checkpoint_fail_flipped_rod.png")
    assert m_flip.mask_iou >= 0.82  # Bbox and silhouette overlap!
    assert m_flip.witness_error_px > 6.0  # Witness stripe is at the opposite end!
    assert m_flip.seating_status == "misaligned"
    assert m_flip.status == "fail"
    assert "witness_mark_misaligned" in m_flip.reason_codes


def test_flat_link_metrology_on_mat_z0() -> None:
    """Verify homography rectification (`px -> mm`) on `flat_scan_mat_z0.png` recovers 150mm, 25mm, and 8mm dimensions."""
    res = measure_flat_link_on_mat_z0(GOLDEN_DIR / "flat_scan_mat_z0.png")
    assert res.unit == "mm"
    assert res.coordinate_frame == "calibrated_plane"
    assert res.plane_id == "mat_z0"
    assert res.status == "eligible_for_cad"
    assert abs(res.hole_center_distance - 150.0) <= 0.5
    assert abs(res.outer_width - 25.0) <= 0.5
    assert abs(res.hole_diameter_a - 8.0) <= 0.4
    assert abs(res.hole_diameter_b - 8.0) <= 0.4
    assert abs(res.end_radius_a - 12.5) <= 0.3
    assert abs(res.end_radius_b - 12.5) <= 0.3

    env = build_measurement_record_envelope(res)
    validate_envelope_semantics(env)


def test_cad_worker_step_stl_generation_and_verification(tmp_path: Path) -> None:
    """Verify `flat_two_hole_link_v1` generates watertight STL and ISO-10303-21 STEP verified to <= 0.10mm."""
    meas = measure_flat_link_on_mat_z0(GOLDEN_DIR / "flat_scan_mat_z0.png")
    cad_res = generate_flat_two_hole_link_v1(meas, output_dir=tmp_path)

    assert cad_res.status == "geometry_valid"
    assert cad_res.verification.is_watertight is True
    assert cad_res.verification.is_winding_consistent is True
    assert cad_res.verification.max_abs_error_mm <= 0.10

    assert Path(cad_res.step_path).exists()
    assert Path(cad_res.stl_path).exists()
    assert Path(cad_res.drawing_svg_path).exists()
    assert Path(cad_res.drawing_png_path).exists()

    validate_envelope_semantics(cad_res.cad_request_envelope)
    validate_envelope_semantics(cad_res.cad_artifact_envelope)

    # Verify out-of-bounds parameter rejection
    with pytest.raises(ValueError):
        validate_link_parameters_mm(
            hole_center_distance=240.0,  # > 180.0mm max bound
            outer_width=25.0,
            thickness=4.0,
            pin_diameter=8.0,
            diametral_clearance=0.4,
        )


@pytest.mark.asyncio
async def test_gate_and_state_machine_scenarios_and_atomic_cas() -> None:
    """Test two-tier gate across Pass, Flipped Rod, Camera Bump, Timeout, and Atomic CAS rejection."""
    # 1. Glimmer context blinding strips `expected_pose_id`, `expected_step`, `geometry`, `sam`, `iou`
    cleaned = sanitize_blinded_glimmer_context(
        {
            "observation_id": "obs-1",
            "expected_pose_id": "pose-secret",
            "expected_step": "step-secret",
            "geometry": {"status": "pass"},
            "iou": 0.97,
            "part_id": "connecting-rod",
        }
    )
    assert "expected_pose_id" not in cleaned
    assert "expected_step" not in cleaned
    assert "geometry" not in cleaned
    assert "iou" not in cleaned
    assert cleaned["part_id"] == "connecting-rod"

    # 2. Pass Step 2 -> accepted
    sm_pass = SessionStateMachine()
    out_pass = await run_checkpoint_gate(
        GOLDEN_DIR / "checkpoint_pass_step2.png",
        state_machine=sm_pass,
        mode="fixture",
    )
    assert out_pass.decision == "accepted"
    assert sm_pass.state == "ACCEPTED"
    assert "attach-connecting-rod" in sm_pass.completed_steps

    # 3. Flipped Rod -> rejected with semantic_visual_conflict
    sm_flip = SessionStateMachine()
    out_flip = await run_checkpoint_gate(
        GOLDEN_DIR / "checkpoint_fail_flipped_rod.png",
        state_machine=sm_flip,
        mode="fixture",
    )
    assert out_flip.decision == "rejected"
    assert "witness_mark_misaligned" in out_flip.reason_codes
    assert "semantic_visual_conflict" in out_flip.reason_codes
    assert sm_flip.state == "REJECTED"

    # 4. Camera Bump -> rejected with registration_lost + scene_version incremented
    sm_bump = SessionStateMachine()
    sv0 = sm_bump.scene_version
    out_bump = await run_checkpoint_gate(
        GOLDEN_DIR / "checkpoint_fail_camera_bump.png",
        state_machine=sm_bump,
        mode="fixture",
    )
    assert out_bump.decision == "rejected"
    assert "registration_lost" in out_bump.reason_codes
    assert sm_bump.scene_version == sv0 + 1

    # 5. Timeout (>2800ms) -> unknown with UNKNOWN_TIMEOUT
    sm_to = SessionStateMachine()
    out_to = await run_checkpoint_gate(
        GOLDEN_DIR / "checkpoint_pass_step2.png",
        state_machine=sm_to,
        mode="fixture",
        simulated_elapsed_ms=3050.0,
    )
    assert out_to.decision == "unknown"
    assert "UNKNOWN_TIMEOUT" in out_to.reason_codes
    assert sm_to.state == "UNKNOWN"

    # 6. Atomic CAS rejects stale scene_version commit
    stale_gate_env = dict(out_pass.gate_decision_envelope)
    stale_gate_env["scene_version"] = 999
    with pytest.raises(SemanticValidationError) as exc_info:
        sm_pass.commit_gate_decision(stale_gate_env)
    assert exc_info.value.code == "CAS_MISMATCH"


def test_fastapi_endpoints_and_websocket() -> None:
    """Test all FastAPI REST and WebSocket endpoints end-to-end."""
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_print = client.get("/api/printable/manifest")
    assert r_print.status_code == 200
    assert len(r_print.json()["files"]) >= 5

    r_gold = client.get("/api/golden/manifest")
    assert r_gold.status_code == 200
    assert "pass_step2" in r_gold.json()["scenarios"]

    # Reset session
    client.post("/api/session/reset")

    # Calibration endpoint
    r_cal = client.post("/api/calibration/check", json={"scenario": "pass_step2"})
    assert r_cal.status_code == 200
    assert r_cal.json()["calibration_valid"] is True

    # Checkpoint verify: flipped rod failure
    r_flip = client.post(
        "/api/checkpoint/verify",
        json={"scenario": "fail_flipped_rod", "mode": "fixture"},
    )
    assert r_flip.status_code == 200
    assert r_flip.json()["decision"] == "rejected"
    assert "semantic_visual_conflict" in r_flip.json()["reason_codes"]

    # Checkpoint verify: pass step 2
    r_pass = client.post(
        "/api/checkpoint/verify",
        json={"scenario": "pass_step2", "mode": "fixture"},
    )
    assert r_pass.status_code == 200
    assert r_pass.json()["decision"] == "accepted"

    # Flat scan on mat_z0
    r_scan = client.post("/api/metrology/flat-scan", json={})
    assert r_scan.status_code == 200
    meas_data = r_scan.json()["flat_part_MeasurementResult"]
    assert meas_data["plane_id"] == "mat_z0"
    assert abs(meas_data["hole_center_distance"] - 150.0) <= 0.5

    # CAD STEP/STL generation
    r_cad = client.post("/api/cad/generate", json={})
    assert r_cad.status_code == 200
    assert r_cad.json()["cad_VerificationResult"]["status"] == "geometry_valid"
    assert r_cad.json()["cad_VerificationResult"]["max_abs_error_mm"] <= 0.10

    # Voice command execution
    r_voice = client.post("/api/voice/command", json={"transcript": "why did that fail", "mode": "fixture"})
    assert r_voice.status_code == 200
    assert r_voice.json()["intent"] == "explain_failure"

    # Bonus lanes
    r_wonder = client.post("/api/bonus/wonder-plan", json={"mode": "fixture"})
    assert r_wonder.status_code == 200
    assert r_wonder.json()["planner_proposal_envelope"]["payload"]["model_output"]["action"] == "generate_supported_part"

    r_card = client.post("/api/bonus/repair-card", json={"mode": "fixture"})
    assert r_card.status_code == 200
    assert r_card.json()["status"] == "generated"

    r_spark = client.post("/api/bonus/spark-review", json={"mode": "fixture"})
    assert r_spark.status_code == 200
    assert "summary" in r_spark.json()

    # WebSocket session stream
    with client.websocket_connect("/ws/session") as ws:
        init_msg = ws.receive_json()
        assert init_msg["event"] == "connected"
        ws.send_text("ping")
        pong_msg = ws.receive_json()
        assert pong_msg["event"] == "pong"


@pytest.mark.asyncio
async def test_live_sam31_checkpoint_gate_on_golden_frame() -> None:
    _skip_if_proxy_unreachable()
    """Run 1 live end-to-end checkpoint gate test calling real Meta `sam-3.1` on `checkpoint_pass_step2.png`."""
    sm = SessionStateMachine()
    outcome = await run_checkpoint_gate(
        GOLDEN_DIR / "checkpoint_pass_step2.png",
        state_machine=sm,
        mode="hybrid",  # Calls live `sam-3.1` via 127.0.0.1:8791 proxy
        prompt_text="yellow connecting rod",
    )
    assert outcome.decision == "accepted"
    assert outcome.geometry_metrics.mask_iou >= 0.82
    assert outcome.geometry_metrics.witness_error_px <= 6.0
    assert outcome.segmentation_envelope["payload"]["provider_response_id"] != "resp_sam_fixture_01"
    assert len(outcome.segmentation_envelope["payload"]["objects"]) >= 1
    assert sm.state == "ACCEPTED"
