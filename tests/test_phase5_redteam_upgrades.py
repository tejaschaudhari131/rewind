from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.calibration import (
    calibrate_frame_to_mat_z0,
    compute_planar_calibration,
    detect_apriltag_36h11_corners,
)
from backend.app.geometry import (
    evaluate_elevated_checkpoint,
    evaluate_step_geometry_and_witness,
    extract_yellow_rod_mask_in_rig_zone,
)
from backend.app.main import app
from backend.app.providers.baseten_spark import (
    BasetenSparkClient,
    extract_spark_open_content_and_json,
)
from backend.app.telemetry import build_six_model_proof_ledger
from scripts.generate_printable_rig import (
    CANONICAL_TAG36H11_HEX,
    GOLDEN_DIR,
    render_canonical_apriltag_36h11,
)


def test_canonical_apriltag_36h11_bit_matrices_and_clahe_detector() -> None:
    assert set(CANONICAL_TAG36H11_HEX.keys()) == {0, 1, 2, 3}
    for tid in range(4):
        tile = render_canonical_apriltag_36h11(tid, side_px=120)
        assert tile.shape == (120, 120)
        assert tile.dtype == np.uint8

    pass_frame = cv2.imread(str(GOLDEN_DIR / "checkpoint_pass_step2.png"))
    assert pass_frame is not None
    corners = detect_apriltag_36h11_corners(pass_frame)
    assert sorted(corners.keys()) == [0, 1, 2, 3]

    cal = compute_planar_calibration(pass_frame)
    assert cal.calibration_valid is True
    assert cal.detected_tag_ids == [0, 1, 2, 3]
    assert cal.reprojection_rmse_px <= 1.5

    mat_cal = calibrate_frame_to_mat_z0(pass_frame)
    assert mat_cal.calibrated is True
    assert mat_cal.detected_tag_ids == [0, 1, 2, 3]
    assert mat_cal.reprojection_error_mm <= 0.5
    assert mat_cal.warped_bgr.shape == (1000, 1600, 3)


def test_cie_lab_witness_polarity_cosine_and_paired_crops() -> None:
    pass_img = cv2.imread(str(GOLDEN_DIR / "checkpoint_pass_step2.png"))
    flip_img = cv2.imread(str(GOLDEN_DIR / "checkpoint_fail_flipped_rod.png"))
    ref_mask = cv2.imread(str(GOLDEN_DIR / "ref_mask_step2_connecting_rod.png"), cv2.IMREAD_GRAYSCALE)
    pass_mask = extract_yellow_rod_mask_in_rig_zone(pass_img)
    flip_mask = extract_yellow_rod_mask_in_rig_zone(flip_img)

    pass_metrics = evaluate_elevated_checkpoint(
        image=pass_img,
        candidate_mask=pass_mask,
        reference_mask=ref_mask,
    )
    assert pass_metrics.status == "pass"
    assert pass_metrics.witness_aligned is True
    assert pass_metrics.witness_polarity_cosine > 0.90
    assert pass_metrics.paired_crop_ref_b64.startswith("data:image/png;base64,")
    assert pass_metrics.paired_crop_live_b64.startswith("data:image/png;base64,")

    flip_metrics = evaluate_elevated_checkpoint(
        image=flip_img,
        candidate_mask=flip_mask,
        reference_mask=ref_mask,
    )
    assert flip_metrics.status == "fail"
    assert flip_metrics.mask_iou >= 0.90
    assert flip_metrics.witness_aligned is False
    assert flip_metrics.witness_polarity_cosine < -0.80
    assert flip_metrics.witness_error_px > 6.0

    audit = evaluate_step_geometry_and_witness(
        ref_warped_bgr=pass_img,
        ref_sam_mask_u8=ref_mask,
        live_warped_bgr=flip_img,
        live_sam_mask_u8=flip_mask,
    )
    assert audit["witness_aligned"] is False
    assert audit["witness_polarity_cosine"] < -0.80
    assert audit["deterministic_verdict"] == "REJECT_WITNESS_FLIPPED_180"


def test_baseten_spark_open_1026_reasoning_json_extractor() -> None:
    msg = {
        "role": "assistant",
        "content": (
            "<think>Analyzing silhouette IoU=0.9628 vs witness_polarity_cosine=-0.9412...</think>\n"
            "```json\n"
            '{"verdict": "MISMATCH", "failure_signature": "ASYMMETRIC_ROD_INVERTED_180_DEG"}\n'
            "```"
        ),
    }
    cleaned, parsed = extract_spark_open_content_and_json(msg)
    assert "<think>" not in cleaned
    assert parsed is not None
    assert parsed["verdict"] == "MISMATCH"
    assert parsed["failure_signature"] == "ASYMMETRIC_ROD_INVERTED_180_DEG"


@pytest.mark.asyncio
async def test_dual_open_weight_consensus_arbiter_fixture_mode() -> None:
    spark = BasetenSparkClient(baseten_api_key="")
    assert spark.cfg.baseten_spark_model == "meta-models/muse-spark-open-1026"

    arb = await spark.arbitrate_ambiguous_checkpoint(
        session_id="test-dual-consensus",
        step_id="attach-connecting-rod",
        silhouette_iou=0.9628,
        witness_polarity_cosine=-0.9412,
        witness_error_px=41.2,
        glimmer_finding="mismatch",
        mode="fixture",
    )
    assert arb["triggered"] is True
    assert arb["model"] == "meta-models/muse-spark-open-1026"
    assert arb["consensus"] == "DUAL_OPEN_WEIGHT_REJECT_180_FLIP"


def test_six_model_proof_ledger_and_phase5_api_endpoints() -> None:
    ledger = build_six_model_proof_ledger(
        session_id="sess-p5",
        step_id="attach-connecting-rod",
        mode="fixture",
        sam_latency_ms=12.4,
        glimmer_latency_ms=18.2,
        baseten_triggered=True,
        total_wall_ms=45.6,
    )
    d = ledger.to_dict()
    assert len(d["receipts"]) >= 6
    model_ids = [r["model_id"] for r in d["receipts"]]
    assert "meta-models/muse-spark-open-1026" in model_ids
    assert "opencv-lab-witness-precheck" in model_ids
    assert d["all_live_verified"] is False
    assert "sam-3.1" in model_ids

    client = TestClient(app)

    # /api/verify-step on flipped rod
    res_flip = client.post(
        "/api/verify-step",
        json={"scenario": "fail_flipped_rod", "mode": "fixture"},
    )
    assert res_flip.status_code == 200
    payload = res_flip.json()
    assert payload["decision"] == "rejected"
    assert payload["witness_polarity_cosine"] < -0.80
    assert payload["paired_crop_ref_b64"].startswith("data:image/png;base64,")
    assert payload["paired_crop_live_b64"].startswith("data:image/png;base64,")
    assert payload["dual_open_weight_consensus"]["triggered"] is True
    assert payload["dual_open_weight_consensus"]["model"] == "meta-models/muse-spark-open-1026"
    assert len(payload["telemetry_trace"]["receipts"]) >= 6

    # /api/telemetry
    res_tel = client.get("/api/telemetry")
    assert res_tel.status_code == 200
    assert len(res_tel.json()["receipts"]) >= 6

    # /api/repair-card/{session_id}/{step_id}
    res_card = client.get("/api/repair-card/rewind-demo-session-001/attach-connecting-rod")
    assert res_card.status_code == 200
    assert res_card.json()["status"] == "ready"
    img_res = client.get(res_card.json()["repair_card_url"])
    assert img_res.status_code == 200
    assert img_res.headers["content-type"].startswith("image/png")
