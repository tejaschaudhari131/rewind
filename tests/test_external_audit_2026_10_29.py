"""Regression tests for the external adversarial audit of 2026-10-29 (docs/audits/2026-10-29-external).

Every reproduction probe from reproduce_audit.py is turned into an assertion that the system now
REJECTS / fails closed. Also covers the integrated height-plane projection and deadline helpers.
"""
from __future__ import annotations

import asyncio
import copy
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "fixtures" / "golden_rig"
PASS_IMG = GOLDEN / "checkpoint_pass_step2.png"

from backend.app.semantic_validator import SemanticValidationError  # noqa: E402
from backend.app.state_machine import SessionStateMachine  # noqa: E402


def _valid_sam_token() -> str:
    text = (ROOT / "tests" / "test_providers_unit.py").read_text(encoding="utf-8")
    return re.search(r'raw_token = "(<0f>[^"]+)"', text).group(1)


# ---------------------------------------------------------------- C1: Glimmer heuristic fails closed
def _glimmer_inspect():
    from modal_app.glimmer_server import _inspect_image_for_witness_and_seating

    return _inspect_image_for_witness_and_seating


CTX = {"observation_id": "11111111-1111-4111-8111-111111111111", "frame_id": "22222222-2222-4222-8222-222222222222"}


def _finding(result):
    out = result[0]
    return out.get("finding"), out.get("requires_new_view")


def test_glimmer_no_image_is_insufficient_evidence():
    finding, new_view = _finding(_glimmer_inspect()([], CTX))
    assert finding == "insufficient_evidence" and new_view is True


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((100, 100, 3), np.uint8),  # too small
        np.full((800, 1280, 3), 255, np.uint8),  # blank
        np.full((800, 1280, 3), 90, np.uint8),  # uniform, no rod
    ],
)
def test_glimmer_small_or_blank_image_is_insufficient(img):
    ok, enc = cv2.imencode(".png", img)
    finding, new_view = _finding(_glimmer_inspect()([enc.tobytes()], CTX))
    assert finding == "insufficient_evidence" and new_view is True


def test_glimmer_undecodable_bytes_insufficient():
    finding, _ = _finding(_glimmer_inspect()([b"not an image"], CTX))
    assert finding == "insufficient_evidence"


# ---------------------------------------------------------------- C4: nested / filename blinding
@pytest.mark.parametrize(
    "ctx",
    [
        {"observation_id": "x", "debug": {"geometry": {"verdict": "pass"}}},
        {"observation_id": "x", "notes": [{"sam_iou": 0.99}]},
        {"observation_id": "x", "filename": "checkpoint_fail_flipped_rod.png"},
        {"observation_id": "x", "inspection_focus": "golden step2_pass.png"},
        {"observation_id": {"nested": "id"}},
    ],
)
def test_nested_blinding_rejected(ctx):
    from backend.app.semantic_validator import assert_glimmer_input_blinded

    with pytest.raises(SemanticValidationError):
        assert_glimmer_input_blinded(ctx)


def test_sanitizer_is_allowlist():
    from backend.app.gate import sanitize_blinded_glimmer_context

    out = sanitize_blinded_glimmer_context(
        {"observation_id": "a", "frame_id": "b", "part_id": "connecting-rod", "debug": {"geometry": "ok"}, "filename": "x"}
    )
    assert set(out) == {"observation_id", "frame_id", "part_id"}


# ---------------------------------------------------------------- C3: SAM decode fails closed
def test_sam_empty_output_yields_no_instances():
    from backend.app.providers.meta_sam import decode_sam_token_string

    assert decode_sam_token_string("", image_width_px=640, image_height_px=400) == []


@pytest.mark.parametrize("bad", ["garbage without tokens", "<0f>0<|box;x1=1;y1=1|>", "<0f>0<|mask;x=0;y=0;data=5,5,~!!|>"])
def test_sam_malformed_output_raises(bad):
    from backend.app.providers.meta_sam import SamDecodeError, decode_sam_token_string

    with pytest.raises(SamDecodeError):
        decode_sam_token_string(bad, image_width_px=640, image_height_px=400)


def test_sam_mask_outside_image_raises():
    from backend.app.providers.meta_sam import SamDecodeError, decode_sam_token_string

    with pytest.raises(SamDecodeError):
        decode_sam_token_string(_valid_sam_token(), image_width_px=50, image_height_px=50)


def test_sam_valid_decodes_and_confidence_not_defaulted():
    from backend.app.providers.meta_sam import decode_sam_token_string

    inst = decode_sam_token_string(_valid_sam_token(), image_width_px=640, image_height_px=400)
    assert len(inst) == 1 and inst[0].area_px > 0
    tok = _valid_sam_token().replace(";c=0.9532163739204407", "")
    no_conf = decode_sam_token_string(tok, image_width_px=640, image_height_px=400)
    assert no_conf and no_conf[0].confidence is None


def test_sam_multi_box_prompt_uses_half_open_and_validates_size():
    from backend.app.providers.meta_sam import format_multi_box_prompt

    p = format_multi_box_prompt([(10, 10, 19, 19), (30, 30, 39, 39)], image_size=(64, 64))
    assert "x1=10;y1=10;x2=19;y2=19" in p and "x2=39" in p  # inclusive wire format round-trips
    with pytest.raises(Exception):
        format_multi_box_prompt([(10, 10, 100, 100)], image_size=(64, 64))


# ---------------------------------------------------------------- B2: flat measurement / projection
def _camera(h_cam=500.0):
    K = np.array([[900.0, 0, 640], [0, 900.0, 400], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0])  # looking straight down
    t = -R @ np.array([200.0, 125.0, h_cam])
    return K, R, t


def test_height_plane_projection_removes_parallax():
    from backend.app.projection import pixels_to_height_plane_mm

    K, R, t = _camera()
    pts_world = np.array([[125.0, 125.0, 4.0], [275.0, 125.0, 4.0]])
    cam = (R @ pts_world.T).T + t
    px = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
    on_top = pixels_to_height_plane_mm(px, K, np.zeros(5), R, t, 4.0)
    on_mat = pixels_to_height_plane_mm(px, K, np.zeros(5), R, t, 0.0)
    assert abs(np.linalg.norm(on_top[1] - on_top[0]) - 150.0) < 1e-6
    # Naive z=0 mapping overstates by Z/(Z-h) = 500/496 -> ~1.21 mm: far beyond a 0.25 mm claim.
    assert abs(np.linalg.norm(on_mat[1] - on_mat[0]) - 150.0 * 500 / 496) < 1e-3


def test_height_plane_projection_rejects_behind_camera():
    from backend.app.projection import pixels_to_height_plane_mm

    K, R, t = _camera()
    with pytest.raises(Exception):
        pixels_to_height_plane_mm(np.array([[640.0, 400.0]]), K, np.zeros(5), R, t, 600.0)


def test_top_plane_homography_matches_projection():
    from backend.app.projection import top_plane_homography

    K, R, t = _camera()
    H = top_plane_homography(K, R, t, 4.0)
    X = np.array([150.0, 100.0, 1.0])
    p = H @ X
    cam = R @ np.array([150.0, 100.0, 4.0]) + t
    q = K @ (cam / cam[2])
    assert np.allclose(p[:2] / p[2], q[:2], atol=1e-6)


def test_user_supplied_dims_are_not_labelled_calibrated(tmp_path):
    from backend.app.cad_worker import generate_flat_two_hole_link_v1

    res = generate_flat_two_hole_link_v1(
        {"hole_center_distance": 150, "outer_width": 25, "thickness": 4, "pin_diameter": 8, "diametral_clearance": 0.4,
         "plane_id": "mat_z0", "uncertainty_95_mm": None},
        output_dir=tmp_path,
    )
    params = res.cad_request_envelope["payload"]["parameters"]
    assert params["hole_center_distance"]["source"] == "manual_caliper"
    assert params["hole_center_distance"]["uncertainty_95_mm"] is None


def test_calibration_never_claims_validated():
    from backend.app.calibration import build_calibration_envelope, compute_planar_calibration

    cal = compute_planar_calibration(cv2.imread(str(PASS_IMG)))
    env = build_calibration_envelope(cal)
    assert env["payload"]["status"] in ("unvalidated", "invalid")


# ---------------------------------------------------------------- B5: STEP tampering
@pytest.fixture(scope="module")
def cad_out():
    from backend.app.cad_worker import generate_flat_two_hole_link_v1

    d = Path(tempfile.mkdtemp())
    res = generate_flat_two_hole_link_v1(
        {"hole_center_distance": 150, "outer_width": 25, "thickness": 4, "pin_diameter": 8, "diametral_clearance": 0.4,
         "plane_id": "mat_z0", "uncertainty_95_mm": 0.25},
        output_dir=d,
    )
    yield d, res
    shutil.rmtree(d, ignore_errors=True)


def _verify(step, stl):
    from backend.app.cad_worker import verify_generated_cad_artifacts

    return verify_generated_cad_artifacts(step, stl, 150.0, 25.0, 4.0, 8.0)


def _files(d):
    return next(d.glob("*.step")), next(d.glob("*.stl"))


def test_step_untampered_is_faceted_and_valid(cad_out):
    d, res = cad_out
    assert res.verification.status == "geometry_valid"
    assert res.verification.step_kind == "faceted_mesh_derived_step"
    text = _files(d)[0].read_text()
    assert "CYLINDRICAL_SURFACE" not in text and "FACETED_BREP(" in text


@pytest.mark.parametrize("mutation", ["drop_faces", "drop_one_face", "dangling", "move_vertex"])
def test_step_tampering_rejected(cad_out, tmp_path, mutation):
    d, _ = cad_out
    step, stl = _files(d)
    text = step.read_text()
    if mutation == "drop_faces":
        text = "\n".join(l for l in text.splitlines() if "FACE_SURFACE(" not in l)
    elif mutation == "drop_one_face":
        lines = text.splitlines()
        idx = next(i for i, l in enumerate(lines) if "FACE_SURFACE(" in l)
        del lines[idx]
        text = "\n".join(lines)
    elif mutation == "dangling":
        text = re.sub(r"FACE_OUTER_BOUND\('',#(\d+)", "FACE_OUTER_BOUND('',#99999999", text, count=1)
    elif mutation == "move_vertex":
        text = re.sub(r"(#100 = CARTESIAN_POINT\('',\()([-0-9.]+)", lambda m: m.group(1) + str(float(m.group(2)) + 3.0), text, count=1)
    bad = tmp_path / step.name
    bad.write_text(text)
    assert _verify(bad, stl).status == "rejected"


# ---------------------------------------------------------------- B3: Letter tiles are 1:1
def test_letter_tile_layout_is_one_to_one():
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_printable_rig as g

    layout = g.letter_tile_layout()
    assert len(layout) == 4
    xs = sorted({(t["x0_mm"], t["x1_mm"]) for t in layout})
    ys = sorted({(t["y0_mm"], t["y1_mm"]) for t in layout})
    assert xs[0][0] == 0 and xs[-1][1] == 400 and ys[0][0] == 0 and ys[-1][1] == 250
    for t in layout:
        assert t["scale"] == 1.0
        assert t["draw_w_mm"] == t["x1_mm"] - t["x0_mm"] and t["draw_h_mm"] == t["y1_mm"] - t["y0_mm"]


@pytest.mark.skipif(shutil.which("pdfimages") is None, reason="poppler pdfimages not installed")
def test_letter_tiles_pdf_print_scale():
    pdf = ROOT / "fixtures" / "printable" / "rewind_apriltag_mat_letter_tiles.pdf"
    out = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines()[2:] if l.strip()]
    assert len(rows) == 4
    for r in rows:
        # 8 px/mm mat raster drawn 1:1 => 203.2 ppi
        assert abs(int(r[12]) - 203) <= 1 and abs(int(r[13]) - 203) <= 1


# ---------------------------------------------------------------- C2: deadline helper + late provider
async def test_deadline_helper_timeout_and_error():
    from backend.app.deadline import collect_before_deadline

    loop = asyncio.get_running_loop()

    async def slow():
        await asyncio.sleep(1.0)
        return 1

    async def fast():
        return 2

    async def boom():
        raise RuntimeError("x")

    r = await collect_before_deadline(slow(), fast(), started_at=loop.time(), deadline_seconds=0.05)
    assert r.status == "unknown_timeout"
    r = await collect_before_deadline(boom(), fast(), started_at=loop.time(), deadline_seconds=1.0)
    assert r.status == "unknown_provider_error"
    r = await collect_before_deadline(fast(), fast(), started_at=loop.time(), deadline_seconds=1.0)
    assert r.status == "complete" and r.values == (2, 2)
    # Earlier local work counts against the budget.
    r = await collect_before_deadline(fast(), started_at=loop.time() - 5.0, deadline_seconds=1.0)
    assert r.status == "unknown_timeout"


class _SlowSam:
    async def segment_image(self, *a, **k):
        await asyncio.sleep(2.0)
        raise AssertionError("should have been abandoned")


async def test_late_provider_yields_unknown_and_no_accept():
    from backend.app.gate import run_checkpoint_gate

    sm = SessionStateMachine()
    sm.current_step_index = 1
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="hybrid", sam_client=_SlowSam(), provider_deadline_ms=100)
    assert out.decision == "unknown"
    assert "UNKNOWN_TIMEOUT" in out.reason_codes


async def test_simulated_elapsed_is_never_clamped_down():
    from backend.app.gate import run_checkpoint_gate

    sm = SessionStateMachine()
    sm.current_step_index = 1
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="fixture", simulated_elapsed_ms=3100.0)
    assert out.elapsed_ms >= 3100.0
    assert out.decision != "accepted"


class _EmptySam:
    def __init__(self, n=0):
        self.n = n

    async def segment_image(self, raw, *, prompt_text, part_id):
        from backend.app.gate import _build_fixture_sam_result

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        res = _build_fixture_sam_result(img, part_id=part_id, prompt_text=prompt_text)
        if self.n == 0:
            res.instances = []
        else:
            res.instances = [res.instances[0]] * self.n
        return res


@pytest.mark.parametrize("n,code", [(0, "sam_no_instances"), (2, "sam_ambiguous_instances")])
async def test_empty_or_ambiguous_sam_gate_is_unknown(n, code):
    from backend.app.gate import run_checkpoint_gate

    sm = SessionStateMachine()
    sm.current_step_index = 1
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="hybrid", sam_client=_EmptySam(n), provider_deadline_ms=30000)
    assert out.decision == "unknown"
    assert code in out.reason_codes


def test_fixture_sam_empty_mask_returns_zero_instances():
    from backend.app.gate import _build_fixture_sam_result

    res = _build_fixture_sam_result(np.zeros((400, 640, 3), np.uint8), part_id="connecting-rod", prompt_text="rod")
    assert res.instances == []


# ---------------------------------------------------------------- A1/A2: movement, concurrency, reset
class _GatedSam:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def segment_image(self, raw, *, prompt_text, part_id):
        self.started.set()
        await self.release.wait()
        return await _EmptySam(1).segment_image(raw, prompt_text=prompt_text, part_id=part_id)


async def _gate_with_interrupt(interrupt):
    from backend.app.gate import run_checkpoint_gate

    sm = SessionStateMachine()
    sm.current_step_index = 1
    sam = _GatedSam()
    task = asyncio.create_task(
        run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="hybrid", sam_client=sam, provider_deadline_ms=30000)
    )
    await asyncio.wait_for(sam.started.wait(), 20)
    await interrupt(sm)
    sam.release.set()
    return sm, await task


async def test_movement_during_gate_prevents_commit():
    async def move(sm):
        sm.evaluate_stationarity(motion_score=50.0)

    sm, out = await _gate_with_interrupt(move)
    assert out.decision == "unknown" and "STALE_OBSERVATION" in out.reason_codes
    assert sm.last_gate_decision is None


async def test_reset_during_gate_prevents_commit():
    async def reset(sm):
        sm.retire()

    sm, out = await _gate_with_interrupt(reset)
    assert out.decision == "unknown"
    assert sm.last_gate_decision is None
    with pytest.raises(SemanticValidationError):
        sm.begin_checkpoint_observation()


async def test_concurrent_gate_only_latest_can_commit():
    from backend.app.gate import run_checkpoint_gate

    async def superseding_gate(sm):
        sm.begin_checkpoint_observation()  # a second check supersedes the first

    sm, out = await _gate_with_interrupt(superseding_gate)
    assert out.decision == "unknown" and "STALE_OBSERVATION" in out.reason_codes


def test_movement_while_checking_invalidates_observation():
    sm = SessionStateMachine()
    sm.current_step_index = 1
    sm.begin_checkpoint_observation()
    tok = sm.capture_token()
    sm.evaluate_stationarity(motion_score=50.0)
    assert not sm.token_is_current(tok)
    assert sm.active_observation_id is None


def test_session_ids_are_unique():
    assert SessionStateMachine().session_id != SessionStateMachine().session_id


# ---------------------------------------------------------------- A4: semantic bypasses
def test_commit_without_active_observation_rejected():
    sm = SessionStateMachine()
    sm.current_step_index = 1
    with pytest.raises(SemanticValidationError):
        sm.commit_gate_decision({"payload": {"decision": "accepted"}, "session_id": sm.session_id})


async def test_accept_without_evidence_chain_rejected():
    from backend.app.gate import run_checkpoint_gate
    from backend.app.semantic_validator import validate_cycle_semantics

    sm = SessionStateMachine()
    sm.current_step_index = 1
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="fixture")
    gate_env = copy.deepcopy(out.gate_decision_envelope)
    gate_env["payload"]["decision"] = "accepted"
    # Drop geometry from the cycle: an accepted gate must not validate without it.
    cycle = [out.calibration_envelope, out.segmentation_envelope, out.visual_audit_envelope, gate_env]
    with pytest.raises(SemanticValidationError):
        validate_cycle_semantics(cycle)


async def test_accept_with_foreign_session_rejected():
    from backend.app.gate import run_checkpoint_gate
    from backend.app.semantic_validator import validate_cycle_semantics

    sm = SessionStateMachine()
    sm.current_step_index = 1
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="fixture")
    envs = [copy.deepcopy(e) for e in (out.calibration_envelope, out.segmentation_envelope, out.geometry_envelope,
                                       out.visual_audit_envelope, out.gate_decision_envelope)]
    envs[2]["session_id"] = "00000000-0000-4000-8000-00000000dead"
    with pytest.raises(SemanticValidationError):
        validate_cycle_semantics(envs)


# ---------------------------------------------------------------- A3/C5: voice and step specificity
@pytest.mark.parametrize("phrase", ["don't check step", "do not check step", "never check step", "cancel check step"])
def test_voice_negation_not_mapped(phrase):
    from backend.app.providers.meta_voice import map_transcript_to_command

    assert map_transcript_to_command(phrase).intent == "unrecognized"


def test_voice_positive_still_maps():
    from backend.app.providers.meta_voice import map_transcript_to_command

    assert map_transcript_to_command("check step").intent == "check_step"


def test_voice_check_step_does_not_fabricate_pass():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    client.post("/api/session/reset")
    r = client.post("/api/voice/command", json={"transcript": "check step", "mode": "fixture"}).json()
    assert r["action_result"]["checkpoint_decision"] == "unknown"


async def test_unsupported_step_is_unknown():
    from backend.app.gate import run_checkpoint_gate

    sm = SessionStateMachine()
    sm.current_step_index = 2  # attach-output-paddle: no verifier
    out = await run_checkpoint_gate(PASS_IMG, state_machine=sm, mode="fixture")
    assert out.decision == "unknown" and "UNSUPPORTED_STEP_VERIFIER" in out.reason_codes


def test_api_rejects_path_outside_fixtures_and_bad_mode():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    r = client.post("/api/checkpoint/verify", json={"scenario": "pass_step2", "image_path": "/etc/passwd"})
    assert r.status_code == 400
    r = client.post("/api/checkpoint/verify", json={"scenario": "pass_step2", "mode": "yolo"})
    assert r.status_code == 422


def test_voice_live_uses_existing_method():
    from backend.app.providers.meta_voice import MetaVoiceClient

    assert hasattr(MetaVoiceClient, "transcribe_wav_24khz")
    src = (ROOT / "backend" / "app" / "main.py").read_text()
    assert "transcribe_wav_file" not in src
