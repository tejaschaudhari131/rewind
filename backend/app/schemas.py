from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"

PAYLOAD_TYPE_TO_PRODUCER: dict[str, str] = {
    "calibration": "calibration_service",
    "measurement_record": "measurement_service",
    "segmentation": "sam_adapter",
    "geometry": "geometry_service",
    "visual_audit": "glimmer_adapter",
    "planner_proposal": "wonder_adapter",
    "gate_decision": "gate_service",
    "cad_request": "coordinator",
    "cad_artifact": "cad_worker",
}

CANONICAL_PARTS: tuple[str, ...] = (
    "crank_support",
    "crank_disk",
    "connecting_rod",
    "slider_block",
    "linear_guide",
    "output_paddle",
    "crank-support",
    "crank-disk",
    "connecting-rod",
    "slider-block",
    "linear-guide",
    "output-paddle",
)

GATE_DECISIONS: tuple[str, ...] = (
    "accepted",
    "rejected",
    "unknown",
    "pending",
)

SUPPORTED_LINK_BOUNDS_MM: dict[str, tuple[float, float]] = {
    "hole_center_distance": (40.0, 180.0),
    "outer_width": (8.0, 40.0),
    "thickness": (2.0, 12.0),
    "pin_diameter": (3.0, 10.0),
    "diametral_clearance": (0.1, 1.5),
}


@lru_cache(maxsize=8)
def load_schema(schema_filename: str) -> dict[str, Any]:
    path = CONTRACTS_DIR / schema_filename
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def get_validator(schema_filename: str) -> Draft202012Validator:
    schema = load_schema(schema_filename)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_contract_envelope(envelope: Mapping[str, Any]) -> None:
    """Validate an inter-stage message envelope against contracts/contract.schema.json."""
    get_validator("contract.schema.json").validate(dict(envelope))


def validate_glimmer_output(payload: Mapping[str, Any]) -> None:
    """Validate compact Glimmer visual audit output against contracts/glimmer-output.schema.json."""
    get_validator("glimmer-output.schema.json").validate(dict(payload))


def validate_wonder_output(payload: Mapping[str, Any]) -> None:
    """Validate compact Wonder/Spark planner proposal against contracts/wonder-output.schema.json."""
    get_validator("wonder-output.schema.json").validate(dict(payload))


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return hashlib.sha256(raw).hexdigest()


def make_artifact(
    *,
    uri: str,
    mime_type: str,
    data: bytes | str = b"",
    sha256: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id or str(uuid.uuid4()),
        "uri": uri,
        "sha256": sha256 or sha256_hex(data or uri),
        "mime_type": mime_type,
    }


def make_envelope(
    *,
    payload_type: str,
    payload: Mapping[str, Any],
    session_id: str = "00000000-0000-4000-8000-000000000001",
    scene_version: int = 1,
    plan_version: int = 1,
    step_id: str = "attach-connecting-rod",
    observation_id: str = "00000000-0000-4000-8000-000000000002",
    observation_sha256: str = "b" * 64,
    reference_id: str = "00000000-0000-4000-8000-000000000003",
    reference_sha256: str = "c" * 64,
    calibration_id: str = "fixed-camera-v1",
    calibration_sha256: str = "d" * 64,
    policy_id: str = "slider-crank-v1",
    policy_sha256: str = "e" * 64,
    captured_at: str | None = None,
    produced_at: str | None = None,
    parent_message_ids: Sequence[str] = (),
    message_id: str | None = None,
    producer: str | None = None,
) -> dict[str, Any]:
    """Construct a contract-compliant inter-stage envelope and validate its schema."""
    resolved_producer = producer or PAYLOAD_TYPE_TO_PRODUCER.get(payload_type)
    if not resolved_producer:
        raise ValueError(f"Unknown contract payload_type: {payload_type!r}")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "message_id": message_id or str(uuid.uuid4()),
        "session_id": session_id,
        "scene_version": int(scene_version),
        "plan_version": int(plan_version),
        "step_id": step_id,
        "observation_id": observation_id,
        "observation_sha256": observation_sha256,
        "reference_id": reference_id,
        "reference_sha256": reference_sha256,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_sha256,
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "captured_at": captured_at or now_iso,
        "produced_at": produced_at or now_iso,
        "parent_message_ids": list(parent_message_ids),
        "producer": resolved_producer,
        "payload_type": payload_type,
        "payload": dict(payload),
    }
    validate_contract_envelope(envelope)
    return envelope
