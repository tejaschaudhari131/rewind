"""Sub-2.8s Two-Tier Deterministic Verification Gate (`backend/app/gate.py`).

Implements the REWIND two-tier checkpoint verification pipeline:
- Tier 1 (0.00-0.65s): Stationary check + fast local AprilTag registration (`registration_rmse_px <= 1.5`)
  + local witness stripe check -> emits provisional status.
- Tier 2 (0.40-2.50s): Concurrent `asyncio.gather` running `sam-3.1` segmentation and blinded
  `Muse Glimmer` visual audit (stripping `expected_pose_id`, `expected_step`, and all forbidden
  upstream geometry/SAM keys before calling Glimmer).
- Gate Commit (2.50-2.80s): Deterministic `gate_service` evaluating `CONTRACT.md` rules and
  committing via atomic compare-and-set on `(scene_version, plan_version, step_id, observation_id)`.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.app.calibration import (
    CalibrationComputation,
    build_calibration_envelope,
    compute_planar_calibration,
)
from backend.app.config import settings
from backend.app.deadline import collect_before_deadline
from backend.app.geometry import (
    ElevatedCheckpointMetrics,
    build_geometry_envelope,
    evaluate_elevated_checkpoint,
    extract_yellow_rod_mask_in_rig_zone,
)
from backend.app.providers.meta_sam import (
    MetaSamClient,
    SamDetectedInstance,
    SamSegmentationResult,
    compute_mask_moments,
)
from backend.app.providers.baseten_spark import BasetenSparkClient
from backend.app.providers.baseten_vision import BasetenVisionAuditor
from backend.app.providers.modal_glimmer import GlimmerAuditResult, ModalGlimmerClient
from backend.app.telemetry import build_six_model_proof_ledger
from backend.app.schemas import make_envelope, sha256_hex
from backend.app.semantic_validator import (
    FORBIDDEN_GLIMMER_CONTEXT_KEYS,
    GLIMMER_CONTEXT_ALLOWED_KEYS,
    SemanticValidationError,
    assert_glimmer_input_blinded,
    validate_cycle_semantics,
    validate_envelope_semantics,
)
from backend.app.state_machine import SessionStateMachine

EXTRA_BLINDED_KEYS = frozenset({"expected_pose_id", "expected_step", "expected_witness_px", "expected_pivots_px"})


def sanitize_blinded_glimmer_context(raw_context: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip `expected_pose_id`, `expected_step`, and all forbidden SAM/Geometry/Gate keys before Glimmer."""
    # Allowlist only (external audit 2026-10-29, C4): nested objects and unknown keys are dropped.
    cleaned = {
        k: v
        for k, v in raw_context.items()
        if k in GLIMMER_CONTEXT_ALLOWED_KEYS and isinstance(v, str)
    }
    assert_glimmer_input_blinded(cleaned)
    return cleaned


@dataclass
class GateRunOutcome:
    """Complete result of running the two-tier verification gate on a stationary checkpoint frame."""

    decision: str
    reason_codes: List[str]
    provisional_status: str
    elapsed_ms: float
    calibration: CalibrationComputation
    geometry_metrics: ElevatedCheckpointMetrics
    glimmer_verdict: str
    glimmer_confidence: float
    witness_mark_aligned: bool
    calibration_envelope: Dict[str, Any]
    segmentation_envelope: Dict[str, Any]
    geometry_envelope: Dict[str, Any]
    visual_audit_envelope: Dict[str, Any]
    gate_decision_envelope: Dict[str, Any]
    session_snapshot: Dict[str, Any]
    dual_open_weight_consensus: Dict[str, Any] = field(default_factory=dict)
    telemetry_trace: Dict[str, Any] = field(default_factory=dict)
    repair_card_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "provisional_status": self.provisional_status,
            "elapsed_ms": self.elapsed_ms,
            "registration_rmse_px": self.calibration.registration_rmse_px,
            "calibration_valid": self.calibration.calibration_valid,
            "detected_tag_ids": self.calibration.detected_tag_ids,
            "detected_corners_px": self.calibration.detected_corners_px,
            "geometry_metrics": self.geometry_metrics.to_dict(),
            "glimmer_verdict": self.glimmer_verdict,
            "glimmer_confidence": self.glimmer_confidence,
            "witness_mark_aligned": self.witness_mark_aligned,
            "blinded_audit_check": {
                "expected_pose_id_stripped": True,
                "expected_step_stripped": True,
                "forbidden_keys_blocked": sorted(FORBIDDEN_GLIMMER_CONTEXT_KEYS | EXTRA_BLINDED_KEYS),
            },
            "envelopes": {
                "calibration": self.calibration_envelope,
                "segmentation": self.segmentation_envelope,
                "geometry": self.geometry_envelope,
                "visual_audit": self.visual_audit_envelope,
                "glimmer_verification": self.visual_audit_envelope,
                "gate_decision": self.gate_decision_envelope,
                "gate_verdict": self.gate_decision_envelope,
            },
            "session_snapshot": self.session_snapshot,
            "witness_polarity_cosine": self.geometry_metrics.witness_polarity_cosine,
            "paired_crop_ref_b64": self.geometry_metrics.paired_crop_ref_b64,
            "paired_crop_live_b64": self.geometry_metrics.paired_crop_live_b64,
            "dual_open_weight_consensus": self.dual_open_weight_consensus,
            "telemetry_trace": self.telemetry_trace,
            "repair_card_url": self.repair_card_url,
        }


def _build_fixture_sam_result(
    img_bgr: np.ndarray,
    part_id: str = "connecting-rod",
    prompt_text: str = "yellow connecting rod",
) -> SamSegmentationResult:
    """Fast local segmentation fallback producing a schema-valid `SamSegmentationResult`."""
    h, w = img_bgr.shape[:2]
    mask = extract_yellow_rod_mask_in_rig_zone(img_bgr)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # Empty local mask -> zero instances (fails closed at the gate), never a fake box.
        return SamSegmentationResult(
            provider_response_id="resp_sam_fixture_01",
            provider_request_id="req_sam_fixture_01",
            image_width_px=w,
            image_height_px=h,
            raw_output_text="",
            latency_ms=18.5,
            instances=[],
        )
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = mask[y1 : y2 + 1, x1 : x2 + 1]
    area, centroid, angle_deg = compute_mask_moments(mask)
    inst = SamDetectedInstance(
        frame_index=0,
        sam_track_id="0",
        part_id=part_id,
        prompt_text=prompt_text,
        box_xyxy_px=(x1, y1, x2, y2),
        mask_token="fixture_mask",
        crop_mask=crop,
        full_frame_mask=mask,
        confidence=0.96,
        area_px=area,
        centroid_xy_px=centroid,
        principal_angle_deg=angle_deg,
    )
    return SamSegmentationResult(
        provider_response_id="resp_sam_fixture_01",
        provider_request_id="req_sam_fixture_01",
        image_width_px=w,
        image_height_px=h,
        raw_output_text="<0f>0<|box;fixture|>",
        latency_ms=18.5,
        instances=[inst],
    )


def _build_blinded_visual_audit_fixture(
    img_bgr: np.ndarray,
    *,
    observation_id: str,
    frame_id: str,
    blinded_context: Mapping[str, Any],
) -> Tuple[GlimmerAuditResult, bool, float]:
    """Inspect the image directly (without any upstream SAM/geometry hints) for witness stripe alignment."""
    assert_glimmer_input_blinded(blinded_context)
    # Inspect the left crank-end region of the connecting rod in the raw image to see if the black/white
    # witness stripe is visibly adjacent to the "WITNESS REF" marker printed on the mat at ~(475, 308) px
    roi = img_bgr[288:328, 455:495]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    has_dark_bar = bool(np.any(gray < 65))
    has_white_bar = bool(np.any(gray > 215))
    witness_aligned = bool(has_dark_bar and has_white_bar)

    if witness_aligned:
        finding = "consistent"
        contradicted: List[str] = []
        statement = "Yellow connecting rod is seated on crank and wrist pins with black/white witness stripe aligned to WITNESS REF."
        conf = 0.94
    else:
        finding = "mismatch"
        contradicted = ["witness-stripe-aligned"]
        statement = "Yellow connecting rod body is present on pins, but the black/white witness stripe is flipped 180 degrees to the opposite end away from WITNESS REF."
        conf = 0.92

    audit_dict = {
        "observation_id": observation_id,
        "finding": finding,
        "observations": [
            {
                "frame_id": frame_id,
                "region_xyxy_px": [385, 280, 840, 336],
                "statement": statement,
            }
        ],
        "contradicted_predicate_ids": contradicted,
        "requires_new_view": False,
    }
    return (
        GlimmerAuditResult(
            model_used="muse-glimmer-blinded",
            audit=audit_dict,
            latency_ms=42.0,
        ),
        witness_aligned,
        conf,
    )


def evaluate_deterministic_gate_rules(
    *,
    geometry_metrics: ElevatedCheckpointMetrics,
    glimmer_finding: str,
    glimmer_confidence: float,
    witness_mark_aligned: bool,
    elapsed_ms: float,
    min_sam_iou: float = 0.82,
    max_witness_error_px: float = 6.0,
    max_pivot_error_px: float = 5.0,
    min_glimmer_confidence: float = 0.78,
    max_registration_rmse_px: float = 1.5,
    deadline_ms: float = 2800.0,
) -> Tuple[str, List[str]]:
    """Evaluate all deterministic gate rules from `CONTRACT.md`.

    Returns `(decision, reason_codes)` where `decision` is one of `("accepted", "rejected", "unknown")`.
    """
    reasons: List[str] = []

    # 1. Hard registration check
    if not geometry_metrics.calibration_valid or geometry_metrics.registration_rmse_px > max_registration_rmse_px:
        return "rejected", ["registration_lost"]

    # 2. Hard latency budget check (2.80s)
    if elapsed_ms > deadline_ms:
        return "unknown", ["UNKNOWN_TIMEOUT"]

    # 3. Check geometry thresholds & Glimmer visual audit
    iou_ok = geometry_metrics.mask_iou >= min_sam_iou
    witness_px_ok = geometry_metrics.witness_error_px <= max_witness_error_px
    pivot_px_ok = geometry_metrics.pivot_error_px <= max_pivot_error_px
    geom_pass = geometry_metrics.status == "pass" and iou_ok and witness_px_ok and pivot_px_ok

    glimmer_pass = (
        glimmer_finding == "consistent"
        and witness_mark_aligned
        and glimmer_confidence >= min_glimmer_confidence
    )

    if not iou_ok:
        reasons.append("mask_iou_below_threshold")
    if not pivot_px_ok:
        reasons.append("pivot_error_exceeded")
    if not witness_px_ok or not witness_mark_aligned:
        reasons.append("witness_mark_misaligned")

    # Detect semantic/visual conflict when silhouette IoU passes (>= 0.82) but witness orientation fails!
    if iou_ok and (not witness_px_ok or glimmer_finding == "mismatch"):
        if "semantic_visual_conflict" not in reasons:
            reasons.append("semantic_visual_conflict")

    # A model that says "consistent" while local witness evidence disagrees is a conflict, not a
    # silent rewrite to "mismatch" (external audit 2026-10-29, C4).
    if glimmer_finding == "consistent" and not witness_mark_aligned:
        return "unknown", sorted(set(reasons) | {"model_local_disagreement"})

    # Enforce semantic_validator invariant: if geometry_envelope status=='pass' and Glimmer finding=='mismatch',
    # decision MUST be 'unknown' (conflict between passing geometry and failing visual audit).
    if geometry_metrics.status == "pass" and glimmer_finding == "mismatch":
        return "unknown", reasons or ["semantic_visual_conflict"]

    if geom_pass and glimmer_pass:
        return "accepted", []

    if glimmer_finding == "insufficient_evidence" or glimmer_confidence < min_glimmer_confidence:
        reasons.append("low_visual_audit_confidence")
        return "unknown", reasons

    return "rejected", reasons or ["verification_failed"]


SUPPORTED_GATE_STEPS: Dict[str, List[str]] = {
    # Only the connecting-rod checkpoint has a real verifier; every other step is UNKNOWN
    # (external audit 2026-10-29, A3) instead of silently reusing rod predicates.
    "attach-connecting-rod": ["rod-visible-alignment", "witness-stripe-aligned"],
}
VALID_GATE_MODES = ("fixture", "hybrid", "live")


def _file_sha256(path: Path) -> str:
    try:
        return sha256_hex(Path(path).read_bytes())
    except OSError:
        return sha256_hex(f"missing:{Path(path).name}")


def _insufficient_audit(observation_id: str, frame_id: str, statement: str) -> GlimmerAuditResult:
    return GlimmerAuditResult(
        model_used="none",
        audit={
            "observation_id": observation_id,
            "finding": "insufficient_evidence",
            "observations": [{"frame_id": frame_id, "region_xyxy_px": [0, 0, 1, 1], "statement": statement}],
            "contradicted_predicate_ids": [],
            "requires_new_view": True,
        },
        latency_ms=0.0,
    )


async def run_checkpoint_gate(
    image: Union[str, Path, np.ndarray],
    *,
    state_machine: Optional[SessionStateMachine] = None,
    mode: str = "fixture",
    raw_glimmer_context: Optional[Mapping[str, Any]] = None,
    simulated_elapsed_ms: Optional[float] = None,
    prompt_text: str = "yellow connecting rod",
    motion_score: Optional[float] = None,
    provider_deadline_ms: Optional[float] = None,
    sam_client: Optional[Any] = None,
    audit_client: Optional[Any] = None,
) -> GateRunOutcome:
    """Execute the checkpoint verification pipeline and (only if still current) commit to `state_machine`.

    Fail-closed guarantees (external audit 2026-10-29):
    - Providers are collected with a hard deadline; a late or failed provider yields UNKNOWN.
    - Zero or multiple SAM instances yield UNKNOWN (no instances[0] guessing).
    - The commit is skipped (decision UNKNOWN / STALE_OBSERVATION) if the session moved, was reset,
      or another observation superseded this one while providers were running.
    - `simulated_elapsed_ms` can only *increase* the measured elapsed time; nothing clamps it down.
    """
    if mode not in VALID_GATE_MODES:
        raise ValueError(f"Unsupported gate mode {mode!r}; expected one of {VALID_GATE_MODES}.")
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    t0 = time.perf_counter()
    sm = state_machine or SessionStateMachine()

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        raw_bytes = image_path.read_bytes()
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        source_uri = f"fixtures/golden_rig/{image_path.name}"
    else:
        img_bgr = image
        ok, enc = cv2.imencode(".png", img_bgr)
        raw_bytes = enc.tobytes() if ok else b""
        source_uri = "evidence://observations/live_frame.png"

    if img_bgr is None:
        raise ValueError("Invalid checkpoint image provided to run_checkpoint_gate")

    obs_sha256 = sha256_hex(raw_bytes)
    obs_id = str(uuid.uuid4())
    frame_id = str(uuid.uuid4())

    override_reasons: List[str] = []
    if motion_score is not None:
        if sm.evaluate_stationarity(motion_score=motion_score) == "MOVING":
            override_reasons.append("scene_not_stationary")
        stationarity_source = "measured_frame_difference"
    else:
        # A single still frame carries no motion evidence; record that honestly.
        stationarity_source = "not_measured_single_frame"

    sv, pv, step_id, locked_obs_id = sm.begin_checkpoint_observation(
        observation_id=obs_id,
        observation_sha256=obs_sha256,
    )
    commit_token = sm.capture_token()
    step_predicates = SUPPORTED_GATE_STEPS.get(step_id)
    if step_predicates is None:
        override_reasons.append("UNSUPPORTED_STEP_VERIFIER")

    cal = compute_planar_calibration(img_bgr)
    local_metrics = evaluate_elevated_checkpoint(img_bgr, cal=cal)
    provisional = sm.set_provisional_local_status(
        registration_rmse_px=cal.registration_rmse_px,
        witness_error_px=local_metrics.witness_error_px,
    )

    base_context: Dict[str, Any] = {
        "observation_id": locked_obs_id,
        "frame_id": frame_id,
        "part_id": sm.part_id,
        "inspection_focus": "Verify yellow connecting rod is seated on crank and wrist pins and black/white witness stripe aligns with WITNESS REF on the left crank side.",
    }
    if raw_glimmer_context:
        base_context.update(dict(raw_glimmer_context))
    # Callers cannot override the binding IDs.
    base_context["observation_id"] = locked_obs_id
    base_context["frame_id"] = frame_id
    blinded_context = sanitize_blinded_glimmer_context(base_context)
    reference_image_bytes: Optional[bytes] = None
    try:
        from backend.app.geometry import GOLDEN_REF_PASS_IMAGE_PATH

        reference_image_bytes = GOLDEN_REF_PASS_IMAGE_PATH.read_bytes()
    except OSError:
        reference_image_bytes = None

    async def _run_sam_branch() -> SamSegmentationResult:
        if mode in ("live", "hybrid"):
            client = sam_client or MetaSamClient()
            return await client.segment_image(raw_bytes, prompt_text=prompt_text, part_id=sm.part_id)
        return _build_fixture_sam_result(img_bgr, part_id=sm.part_id, prompt_text=prompt_text)

    async def _run_glimmer_branch() -> Tuple[GlimmerAuditResult, bool, float]:
        local_audit, wit_ok, local_conf = _build_blinded_visual_audit_fixture(
            img_bgr,
            observation_id=locked_obs_id,
            frame_id=frame_id,
            blinded_context=blinded_context,
        )
        if mode != "live":
            return local_audit, wit_ok, local_conf
        if audit_client is not None:
            live_audit = await audit_client.audit_checkpoint(
                current_image=raw_bytes, reference_image=reference_image_bytes, blinded_context=blinded_context
            )
        else:
            try:
                live_audit = await BasetenVisionAuditor().audit_checkpoint(
                    current_image=raw_bytes, reference_image=reference_image_bytes, blinded_context=blinded_context
                )
            except Exception:
                live_audit = await ModalGlimmerClient().audit_checkpoint(
                    current_image=raw_bytes, reference_image=reference_image_bytes, blinded_context=blinded_context
                )
        raw_verdict = (live_audit.usage or {}).get("raw_verdict") or {}
        conf = raw_verdict.get("confidence")
        live_conf = float(conf) if isinstance(conf, (int, float)) else local_conf
        # The model's finding is never rewritten; disagreement is resolved by the gate rules.
        return live_audit, wit_ok, live_conf

    deadline_ms = float(provider_deadline_ms if provider_deadline_ms is not None else settings.gate_deadline_ms)
    completion = await collect_before_deadline(
        _run_sam_branch(), _run_glimmer_branch(), started_at=started_at, deadline_seconds=deadline_ms / 1000.0
    )
    if completion.status == "complete":
        sam_result, (glimmer_result, witness_aligned, glimmer_conf) = completion.values
    else:
        override_reasons.append("UNKNOWN_TIMEOUT" if completion.status == "unknown_timeout" else "PROVIDER_ERROR")
        h, w = img_bgr.shape[:2]
        sam_result = SamSegmentationResult(
            provider_response_id="resp_sam_unavailable",
            provider_request_id="req_sam_unavailable",
            image_width_px=w,
            image_height_px=h,
            raw_output_text="",
            latency_ms=0.0,
            instances=[],
        )
        glimmer_result = _insufficient_audit(locked_obs_id, frame_id, f"Provider result unavailable ({completion.status}).")
        witness_aligned, glimmer_conf = False, 0.0

    sam_mask: Optional[np.ndarray] = None
    if completion.status == "complete":
        if len(sam_result.instances) == 0:
            override_reasons.append("sam_no_instances")
        elif len(sam_result.instances) > 1:
            override_reasons.append("sam_ambiguous_instances")
        else:
            inst = sam_result.instances[0]
            if int(np.count_nonzero(inst.full_frame_mask)) == 0:
                override_reasons.append("sam_empty_mask")
            else:
                sam_mask = inst.full_frame_mask

    geom_metrics = evaluate_elevated_checkpoint(img_bgr, candidate_mask=sam_mask, cal=cal)
    glimmer_finding = str(glimmer_result.audit.get("finding", "insufficient_evidence"))

    actual_elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    effective_elapsed_ms = max(actual_elapsed_ms, float(simulated_elapsed_ms or 0.0))

    decision, reason_codes = evaluate_deterministic_gate_rules(
        geometry_metrics=geom_metrics,
        glimmer_finding=glimmer_finding,
        glimmer_confidence=glimmer_conf,
        witness_mark_aligned=witness_aligned,
        elapsed_ms=effective_elapsed_ms,
        deadline_ms=float(settings.gate_deadline_ms),
    )
    if override_reasons and "registration_lost" not in reason_codes:
        decision = "unknown"
        reason_codes = list(dict.fromkeys(override_reasons + list(reason_codes)))

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    from backend.app.geometry import GOLDEN_REF_MASK_PATH

    common_env_kwargs: Dict[str, Any] = {
        "session_id": sm.session_id,
        "scene_version": sv,
        "plan_version": pv,
        "step_id": step_id,
        "observation_id": locked_obs_id,
        "observation_sha256": obs_sha256,
        "reference_sha256": _file_sha256(GOLDEN_REF_MASK_PATH),
        "calibration_id": sm.calibration_id,
        "calibration_sha256": sha256_hex(json.dumps(cal.H_img_to_mat_mm, sort_keys=True)),
        "policy_id": sm.policy_id,
        "policy_sha256": sha256_hex(json.dumps(sm.current_step, sort_keys=True)),
        "captured_at": now_iso,
        "produced_at": now_iso,
    }

    cal_env = build_calibration_envelope(
        cal,
        session_id=sm.session_id,
        scene_version=sv,
        plan_version=pv,
        step_id=step_id,
        observation_id=locked_obs_id,
        calibration_frame_id=frame_id,
    )

    seg_payload = sam_result.to_segmentation_payload(
        frame_id=frame_id,
        source_uri=source_uri,
        source_sha256=obs_sha256,
    )
    seg_env = make_envelope(payload_type="segmentation", payload=seg_payload, parent_message_ids=[], **common_env_kwargs)

    geom_env = build_geometry_envelope(
        geom_metrics,
        part_id=sm.part_id,
        segmentation_message_id=seg_env["message_id"],
        evidence_frame_id=frame_id,
        parent_message_ids=[seg_env["message_id"]],
        **common_env_kwargs,
    )
    geom_env["observation_sha256"] = obs_sha256
    geom_env["captured_at"] = now_iso
    geom_env["produced_at"] = now_iso

    audit_payload = glimmer_result.to_visual_audit_payload(
        model_id="muse-glimmer",
        model_revision="glimmer-30b-a3b-1026",
        deployment_revision="modal-rewind-glimmer-v1",
    )
    audit_env = make_envelope(payload_type="visual_audit", payload=audit_payload, parent_message_ids=[], **common_env_kwargs)

    def _gate_envelope(dec: str, reasons: List[str]) -> Dict[str, Any]:
        payload = {
            "decision": dec,
            "reason_codes": reasons,
            "geometry_message_id": geom_env["message_id"],
            "audit_message_id": audit_env["message_id"],
            "segmentation_message_id": seg_env["message_id"],
            "checked_scene_version": sv,
            "committed_scene_version": sv if dec == "accepted" else None,
            "completed_predicate_ids": list(step_predicates or []) if dec == "accepted" else [],
            "decision_at": now_iso,
            "elapsed_ms": effective_elapsed_ms,
        }
        return make_envelope(
            payload_type="gate_decision",
            payload=payload,
            parent_message_ids=[seg_env["message_id"], geom_env["message_id"], audit_env["message_id"]],
            **common_env_kwargs,
        )

    gate_env = _gate_envelope(decision, reason_codes)
    validate_cycle_semantics([seg_env, geom_env, audit_env, gate_env])

    # Commit only if this observation is still the session's current one.
    committed = False
    if sm.token_is_current(commit_token):
        try:
            for env in (cal_env, seg_env, geom_env, audit_env):
                sm.record_envelope(env)
            snapshot = sm.commit_gate_decision(gate_env)
            committed = True
        except SemanticValidationError as exc:
            if exc.code not in ("CAS_MISMATCH", "SESSION_RETIRED"):
                raise
    if not committed:
        decision = "unknown"
        reason_codes = list(dict.fromkeys(["STALE_OBSERVATION"] + [r for r in reason_codes]))
        gate_env = _gate_envelope(decision, reason_codes)
        snapshot = sm.snapshot()

    glimmer_verdict = "pass" if glimmer_finding == "consistent" and witness_aligned else "fail"

    baseten_client = BasetenSparkClient()
    baseten_consensus = await baseten_client.arbitrate_ambiguous_checkpoint(
        session_id=sm.session_id,
        step_id=step_id,
        silhouette_iou=geom_metrics.mask_iou,
        witness_polarity_cosine=geom_metrics.witness_polarity_cosine,
        witness_error_px=geom_metrics.witness_error_px,
        glimmer_finding=glimmer_finding,
        mode=mode,
    )

    first_inst = sam_result.instances[0] if sam_result.instances else None
    rle_b85_len = max(len(str(getattr(first_inst, "mask_token", "")).encode("ascii")), 512) if first_inst else 512
    mask_area_px = int(first_inst.area_px) if first_inst else 0

    scenario_tag = "flipped" if not witness_aligned else ("bump" if not cal.calibration_valid else "pass")
    repair_card_url = f"/api/repair-card/{sm.session_id}/{step_id}" if decision != "accepted" else ""
    telemetry_env = build_six_model_proof_ledger(
        session_id=sm.session_id,
        step_id=step_id,
        scenario=scenario_tag,
        total_wall_ms=effective_elapsed_ms,
        sam_latency_ms=round(float(sam_result.latency_ms), 1),
        rle_base85_bytes=rle_b85_len,
        decoded_mask_area_px=mask_area_px,
        glimmer_latency_ms=round(float(glimmer_result.latency_ms), 1),
        glimmer_verdict=glimmer_finding,
        baseten_triggered=bool(baseten_consensus.get("triggered", False)),
        baseten_latency_ms=round(float(baseten_consensus.get("latency_ms", 0.0)), 1),
        baseten_verdict=str(baseten_consensus.get("baseten_verdict", "NOT_CALLED")),
        baseten_measured=bool(baseten_consensus.get("measured", False)),
        baseten_usage=baseten_consensus.get("usage") or None,
        baseten_request_id=str(baseten_consensus.get("request_id", "")),
        mask_iou=float(geom_metrics.mask_iou),
        witness_polarity_cosine=float(geom_metrics.witness_polarity_cosine),
        witness_aligned=bool(witness_aligned),
        include_repair_image=(decision != "accepted"),
        repair_image_status="ASYNC_PENDING" if decision != "accepted" else "LIVE_API",
        voice_transcript="check step",
        sam_measured=(mode in ("live", "hybrid") and completion.status == "complete"),
        glimmer_model=glimmer_result.model_used,
        glimmer_usage=glimmer_result.usage,
        glimmer_request_id=str((glimmer_result.usage or {}).get("request_id", "")),
        glimmer_measured=(mode == "live" and bool(glimmer_result.usage)),
    )
    trace = telemetry_env.to_dict()
    trace["stationarity_source"] = stationarity_source
    trace["committed"] = committed
    trace["provider_completion"] = completion.status

    return GateRunOutcome(
        decision=decision,
        reason_codes=reason_codes,
        provisional_status=provisional,
        elapsed_ms=effective_elapsed_ms,
        calibration=cal,
        geometry_metrics=geom_metrics,
        glimmer_verdict=glimmer_verdict,
        glimmer_confidence=glimmer_conf,
        witness_mark_aligned=witness_aligned,
        calibration_envelope=cal_env,
        segmentation_envelope=seg_env,
        geometry_envelope=geom_env,
        visual_audit_envelope=audit_env,
        gate_decision_envelope=gate_env,
        session_snapshot=snapshot,
        dual_open_weight_consensus=baseten_consensus,
        telemetry_trace=trace,
        repair_card_url=repair_card_url,
    )
