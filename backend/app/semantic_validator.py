from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from .config import settings
from .schemas import (
    PAYLOAD_TYPE_TO_PRODUCER,
    SUPPORTED_LINK_BOUNDS_MM,
    validate_contract_envelope,
    validate_glimmer_output,
    validate_wonder_output,
)

FORBIDDEN_GLIMMER_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "geometry",
        "metrics",
        "predicates",
        "hard_checks_passed",
        "sam",
        "segmentation",
        "masks",
        "mask_png_uri",
        "iou",
        "planner_proposal",
        "gate_decision",
        "decision",
    }
)

CHECKPOINT_CYCLE_STAGES: frozenset[str] = frozenset(
    {
        "segmentation",
        "geometry",
        "visual_audit",
        "planner_proposal",
        "gate_decision",
    }
)


class SemanticValidationError(ValueError):
    """Raised when a contract message or cycle violates a runtime semantic invariant."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.detail = message


def _parse_iso_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def assert_single_writer(envelope: Mapping[str, Any]) -> None:
    payload_type = envelope.get("payload_type")
    producer = envelope.get("producer")
    expected = PAYLOAD_TYPE_TO_PRODUCER.get(str(payload_type))
    if expected is None:
        raise SemanticValidationError(
            "UNKNOWN_STAGE",
            f"Unknown payload_type {payload_type!r}.",
        )
    if producer != expected:
        raise SemanticValidationError(
            "SINGLE_WRITER_VIOLATION",
            f"Stage {payload_type!r} must be produced by {expected!r}, got {producer!r}.",
        )


def assert_envelope_timestamps(envelope: Mapping[str, Any]) -> None:
    captured = _parse_iso_ts(str(envelope["captured_at"]))
    produced = _parse_iso_ts(str(envelope["produced_at"]))
    if produced < captured:
        raise SemanticValidationError(
            "NON_MONOTONIC_TIMESTAMPS",
            f"produced_at ({envelope['produced_at']}) < captured_at ({envelope['captured_at']}).",
        )


GLIMMER_CONTEXT_ALLOWED_KEYS: frozenset[str] = frozenset(
    {"observation_id", "frame_id", "part_id", "inspection_focus"}
)
_FORBIDDEN_SUBSTRINGS = ("mask", "iou", "geometry", "metric", "predicate", "decision", "verdict", "sam_", "expected_", "golden", "pass", "flipped", "fail")


def _scan_forbidden(value: Any, path: str, found: list[str]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            key = str(k)
            if key in FORBIDDEN_GLIMMER_CONTEXT_KEYS or any(t in key.lower() for t in _FORBIDDEN_SUBSTRINGS):
                found.append(f"{path}{key}")
            _scan_forbidden(v, f"{path}{key}.", found)
    elif isinstance(value, (list, tuple, set)):
        for i, v in enumerate(value):
            _scan_forbidden(v, f"{path}[{i}].", found)


def assert_glimmer_input_blinded(context: Mapping[str, Any]) -> None:
    """Ensure Glimmer visual audit input context is strictly blinded from SAM/Geometry/Gate outputs.

    Checks recursively (nested dicts/lists) and rejects non-string values for allowed keys and
    fixture-filename leaks such as ``step2_flipped.png`` (external audit 2026-10-29, C4).
    """
    found: list[str] = []
    _scan_forbidden(context, "", found)
    for k, v in context.items():
        if k in GLIMMER_CONTEXT_ALLOWED_KEYS and not isinstance(v, str):
            found.append(f"{k}(non-string)")
        if isinstance(v, str) and any(t in v.lower() for t in (".png", ".jpg", "flipped", "step2_pass", "golden")):
            found.append(f"{k}(filename/label leak)")
    leaked = sorted(set(found) | set(FORBIDDEN_GLIMMER_CONTEXT_KEYS.intersection(context.keys())))
    if leaked:
        raise SemanticValidationError(
            "GLIMMER_BLINDING_VIOLATION",
            f"Glimmer input context contains forbidden upstream keys: {sorted(leaked)}.",
        )


def validate_geometry_semantics(envelope: Mapping[str, Any]) -> None:
    payload = envelope["payload"]
    metrics = payload.get("metrics", [])
    if not metrics:
        raise SemanticValidationError("EMPTY_METRICS", "Geometry payload has no metrics.")

    for m in metrics:
        unit = m.get("unit")
        space = m.get("space")
        plane_id = m.get("plane_id")
        val = m.get("value")
        metric_name = m.get("metric")

        # Elevated assembly metrics cannot claim millimeter units without a calibrated plane
        if unit == "mm":
            if space != "calibrated_plane" or not plane_id or plane_id != "mat_z0":
                raise SemanticValidationError(
                    "PLANE_UNIT_VIOLATION",
                    f"Metric {metric_name!r} uses unit='mm' without space='calibrated_plane' and plane_id='mat_z0'.",
                )
        if metric_name in ("mask_iou", "visible_coverage") and val is not None:
            if not (0.0 <= float(val) <= 1.0):
                raise SemanticValidationError(
                    "RATIO_OUT_OF_BOUNDS",
                    f"Metric {metric_name!r} value={val} must be in [0, 1].",
                )

    if payload.get("status") == "pass":
        if not payload.get("calibration_valid") or not payload.get("hands_clear"):
            raise SemanticValidationError(
                "GEOMETRY_PASS_INVALID_PREREQ",
                "Geometry status='pass' requires calibration_valid=True and hands_clear=True.",
            )
        if any(m.get("result") != "pass" for m in metrics):
            raise SemanticValidationError(
                "GEOMETRY_PASS_WITH_FAILED_METRIC",
                "Geometry status='pass' requires all metrics to have result='pass'.",
            )


def validate_visual_audit_semantics(envelope: Mapping[str, Any]) -> None:
    payload = envelope["payload"]
    model_output = payload["model_output"]
    validate_glimmer_output(model_output)

    if model_output.get("observation_id") != envelope["observation_id"]:
        raise SemanticValidationError(
            "OBSERVATION_ID_MISMATCH",
            f"VisualAudit model_output.observation_id {model_output.get('observation_id')!r} != {envelope['observation_id']!r}.",
        )
    for obs in model_output.get("observations", []):
        region = obs.get("region_xyxy_px")
        if region is not None:
            if len(region) != 4 or region[2] < region[0] or region[3] < region[1]:
                raise SemanticValidationError(
                    "INVALID_BBOX",
                    f"VisualAudit region_xyxy_px {region} is invalid.",
                )
    if model_output.get("finding") == "consistent" and model_output.get("contradicted_predicate_ids"):
        raise SemanticValidationError(
            "VISUAL_AUDIT_CONTRADICTION",
            "VisualAudit finding='consistent' cannot list contradicted_predicate_ids.",
        )


def validate_planner_proposal_semantics(envelope: Mapping[str, Any]) -> None:
    model_output = envelope["payload"]["model_output"]
    validate_wonder_output(model_output)
    if model_output.get("observation_id") != envelope["observation_id"]:
        raise SemanticValidationError(
            "OBSERVATION_ID_MISMATCH",
            "PlannerProposal model_output.observation_id does not match envelope observation_id.",
        )
    if model_output.get("action") == "generate_supported_part" and not model_output.get(
        "measurement_record_id"
    ):
        raise SemanticValidationError(
            "MISSING_MEASUREMENT_RECORD_ID",
            "action='generate_supported_part' requires a non-null measurement_record_id.",
        )


def validate_cad_request_semantics(envelope: Mapping[str, Any]) -> None:
    params = envelope["payload"]["parameters"]
    for key, (lo, hi) in SUPPORTED_LINK_BOUNDS_MM.items():
        dim = params[key]
        val = float(dim["value_mm"])
        if not (lo <= val <= hi):
            raise SemanticValidationError(
                "CAD_PARAMETER_OUT_OF_BOUNDS",
                f"CadRequest parameter {key}={val}mm outside supported range [{lo}, {hi}]mm.",
            )
        if dim["source"] == "calibrated_flat_scan" and dim.get("plane_id") != "mat_z0":
            raise SemanticValidationError(
                "PLANE_UNIT_VIOLATION",
                f"CadRequest parameter {key} from calibrated_flat_scan must use plane_id='mat_z0'.",
            )


def validate_gate_decision_semantics(
    gate_envelope: Mapping[str, Any],
    *,
    geometry_envelope: Mapping[str, Any] | None = None,
    visual_audit_envelope: Mapping[str, Any] | None = None,
) -> None:
    payload = gate_envelope["payload"]
    decision = payload["decision"]
    elapsed_ms = float(payload["elapsed_ms"])
    scene_version = int(gate_envelope["scene_version"])
    checked_sv = int(payload["checked_scene_version"])
    committed_sv = payload["committed_scene_version"]

    captured = _parse_iso_ts(str(gate_envelope["captured_at"]))
    decision_at = _parse_iso_ts(str(payload["decision_at"]))
    if decision_at < captured:
        raise SemanticValidationError(
            "NON_MONOTONIC_TIMESTAMPS",
            f"Gate decision_at ({payload['decision_at']}) < captured_at ({gate_envelope['captured_at']}).",
        )

    if checked_sv != scene_version:
        raise SemanticValidationError(
            "SCENE_VERSION_MISMATCH",
            f"checked_scene_version={checked_sv} != envelope scene_version={scene_version}.",
        )

    if elapsed_ms > settings.gate_deadline_ms and decision == "accepted":
        raise SemanticValidationError(
            "DEADLINE_EXCEEDED_NOT_TIMEOUT",
            f"Gate elapsed_ms={elapsed_ms} > {settings.gate_deadline_ms}ms cannot be 'accepted' (must be 'unknown' with UNKNOWN_TIMEOUT).",
        )

    if decision == "accepted":
        if committed_sv != scene_version:
            raise SemanticValidationError(
                "INVALID_ACCEPT_MUTATION",
                f"decision='accepted' requires committed_scene_version={scene_version}, got {committed_sv}.",
            )
        if not payload.get("completed_predicate_ids"):
            raise SemanticValidationError(
                "ACCEPT_WITHOUT_COMPLETED_PREDICATES",
                "decision='accepted' requires non-empty completed_predicate_ids.",
            )
    else:
        if committed_sv is not None:
            raise SemanticValidationError(
                "ILLEGAL_STATE_ADVANCE",
                f"Non-accepted decision={decision!r} must have committed_scene_version=None, got {committed_sv}.",
            )

    if geometry_envelope is not None and visual_audit_envelope is not None:
        geom_status = geometry_envelope["payload"]["status"]
        audit_out = visual_audit_envelope["payload"]["model_output"]
        finding = audit_out["finding"]

        if decision == "accepted":
            if geom_status != "pass":
                raise SemanticValidationError(
                    "ACCEPT_WITH_FAILED_GEOMETRY",
                    f"Cannot accept when geometry status={geom_status!r}.",
                )
            if finding != "consistent" or audit_out.get("contradicted_predicate_ids"):
                raise SemanticValidationError(
                    "ACCEPT_WITH_NON_CONSISTENT_AUDIT",
                    f"Cannot accept when visual_audit finding={finding!r}.",
                )

        if geom_status == "pass" and finding == "mismatch" and decision not in ("unknown",):
            raise SemanticValidationError(
                "MISSING_CONFLICT_OUTCOME",
                f"Geometry pass + Glimmer mismatch requires decision='unknown', got {decision!r}.",
            )


def validate_envelope_semantics(envelope: Mapping[str, Any]) -> None:
    """Validate both single-envelope semantic invariants and JSON Schema."""
    assert_single_writer(envelope)
    assert_envelope_timestamps(envelope)

    payload_type = envelope["payload_type"]
    if payload_type == "geometry":
        validate_geometry_semantics(envelope)
    elif payload_type == "visual_audit":
        validate_visual_audit_semantics(envelope)
    elif payload_type == "planner_proposal":
        validate_planner_proposal_semantics(envelope)
    elif payload_type == "cad_request":
        validate_cad_request_semantics(envelope)
    elif payload_type == "gate_decision":
        validate_gate_decision_semantics(envelope)

    validate_contract_envelope(envelope)


def validate_cycle_semantics(envelopes: Sequence[Mapping[str, Any]]) -> None:
    """Validate an entire sequence of contract messages (e.g. contracts/example-cycle.json)."""
    if not envelopes:
        raise SemanticValidationError("EMPTY_CYCLE", "Envelope sequence cannot be empty.")

    by_stage: dict[str, Mapping[str, Any]] = {}
    by_message_id: dict[str, Mapping[str, Any]] = {}
    known_frame_ids: set[str] = set()

    for env in envelopes:
        validate_envelope_semantics(env)
        by_stage[env["payload_type"]] = env
        by_message_id[env["message_id"]] = env
        if env["payload_type"] == "segmentation":
            for fr in env["payload"].get("frames", []):
                known_frame_ids.add(fr["frame_id"])

    # 1. Verify parent_message_ids resolve inside the cycle when present
    for env in envelopes:
        for pid in env.get("parent_message_ids", []):
            if pid not in by_message_id:
                raise SemanticValidationError(
                    "UNRESOLVED_PARENT_MESSAGE_ID",
                    f"Message {env['message_id']} references unknown parent_message_id {pid}.",
                )

    # 2. Verify Atomic CAS tuple across all checkpoint stages in this cycle
    checkpoint_envs = [
        env for env in envelopes if env["payload_type"] in CHECKPOINT_CYCLE_STAGES
    ]
    if len(checkpoint_envs) > 1:
        ref = checkpoint_envs[0]
        ref_cas = (
            ref["session_id"],
            ref["reference_sha256"],
            ref["calibration_sha256"],
            ref["policy_sha256"],
            ref["scene_version"],
            ref["plan_version"],
            ref["step_id"],
            ref["observation_id"],
            ref["observation_sha256"],
            ref["calibration_id"],
            ref["policy_id"],
        )
        for env in checkpoint_envs[1:]:
            cas = (
                env["session_id"],
                env["reference_sha256"],
                env["calibration_sha256"],
                env["policy_sha256"],
                env["scene_version"],
                env["plan_version"],
                env["step_id"],
                env["observation_id"],
                env["observation_sha256"],
                env["calibration_id"],
                env["policy_id"],
            )
            if cas != ref_cas:
                raise SemanticValidationError(
                    "CAS_MISMATCH",
                    f"Stage {env['payload_type']!r} CAS tuple {cas} != reference {ref_cas}.",
                )

    # 3. Cross-stage checks
    geom_env = by_stage.get("geometry")
    audit_env = by_stage.get("visual_audit")
    gate_env = by_stage.get("gate_decision")
    plan_env = by_stage.get("planner_proposal")
    cad_req_env = by_stage.get("cad_request")

    if audit_env is not None and known_frame_ids:
        for obs in audit_env["payload"]["model_output"].get("observations", []):
            if obs["frame_id"] not in known_frame_ids:
                raise SemanticValidationError(
                    "UNKNOWN_EVIDENCE_FRAME_ID",
                    f"VisualAudit references unknown frame_id {obs['frame_id']!r}.",
                )

    if plan_env is not None and cad_req_env is not None:
        plan_out = plan_env["payload"]["model_output"]
        if plan_out.get("action") == "generate_supported_part":
            if (
                plan_out.get("measurement_record_id")
                != cad_req_env["payload"].get("measurement_record_id")
            ):
                raise SemanticValidationError(
                    "MEASUREMENT_ID_MISMATCH",
                    "PlannerProposal measurement_record_id does not match CadRequest measurement_record_id.",
                )

    if gate_env is not None:
        gate_payload = gate_env["payload"]
        seg_env = by_stage.get("segmentation")
        if gate_payload.get("decision") == "accepted":
            # External audit 2026-10-29, A4: an accepted gate needs the full evidence chain.
            missing = [n for n, e in (("segmentation", seg_env), ("geometry", geom_env), ("visual_audit", audit_env)) if e is None]
            if missing:
                raise SemanticValidationError(
                    "ACCEPT_WITHOUT_EVIDENCE_CHAIN",
                    f"Accepted gate_decision is missing upstream stages {missing}.",
                )
            links = (
                ("segmentation_message_id", seg_env),
                ("geometry_message_id", geom_env),
                ("audit_message_id", audit_env),
            )
            parents = set(gate_env.get("parent_message_ids", []))
            for key, env in links:
                if gate_payload.get(key) != env["message_id"] or env["message_id"] not in parents:
                    raise SemanticValidationError(
                        "EVIDENCE_LINK_MISMATCH",
                        f"Accepted gate_decision {key} does not reference the cycle's {env['payload_type']} message.",
                    )
            if geom_env["payload"].get("segmentation_message_id") not in (None, seg_env["message_id"]):
                raise SemanticValidationError(
                    "EVIDENCE_LINK_MISMATCH",
                    "Geometry segmentation_message_id does not match the cycle's segmentation message.",
                )
            if audit_env["payload"]["model_output"].get("requires_new_view"):
                raise SemanticValidationError(
                    "ACCEPT_REQUIRES_NEW_VIEW",
                    "Cannot accept when the visual audit requires a new view.",
                )
        validate_gate_decision_semantics(
            gate_env,
            geometry_envelope=geom_env,
            visual_audit_envelope=audit_env,
        )
