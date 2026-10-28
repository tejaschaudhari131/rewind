from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.app.schemas import (
    validate_contract_envelope,
    validate_glimmer_output,
    validate_wonder_output,
)
from backend.app.semantic_validator import (
    SemanticValidationError,
    assert_glimmer_input_blinded,
    validate_cycle_semantics,
    validate_envelope_semantics,
)

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


@pytest.fixture
def example_cycle() -> list[dict]:
    data = json.loads((CONTRACTS_DIR / "example-cycle.json").read_text(encoding="utf-8"))
    return data["messages"]


def test_example_cycle_passes_schema_and_semantics(example_cycle: list[dict]) -> None:
    for env in example_cycle:
        validate_contract_envelope(env)
        validate_envelope_semantics(env)
    validate_cycle_semantics(example_cycle)


def test_single_writer_violation_rejected(example_cycle: list[dict]) -> None:
    bad = copy.deepcopy(example_cycle[0])
    bad["producer"] = "wonder_adapter"  # segmentation must be produced by sam_adapter
    with pytest.raises(SemanticValidationError) as exc:
        validate_envelope_semantics(bad)
    assert exc.value.code == "SINGLE_WRITER_VIOLATION"


def test_non_monotonic_timestamps_rejected(example_cycle: list[dict]) -> None:
    bad = copy.deepcopy(example_cycle[0])
    bad["captured_at"] = "2026-10-28T16:10:00Z"
    bad["produced_at"] = "2026-10-28T16:05:00Z"
    with pytest.raises(SemanticValidationError) as exc:
        validate_envelope_semantics(bad)
    assert exc.value.code == "NON_MONOTONIC_TIMESTAMPS"


def test_plane_unit_violation_elevated_mm_rejected(example_cycle: list[dict]) -> None:
    geom = copy.deepcopy(next(e for e in example_cycle if e["payload_type"] == "geometry"))
    geom["payload"]["metrics"][0]["space"] = "undistorted_image"
    geom["payload"]["metrics"][0]["plane_id"] = None
    geom["payload"]["metrics"][0]["unit"] = "mm"  # illegal on undistorted_image without mat_z0!
    with pytest.raises(SemanticValidationError) as exc:
        validate_envelope_semantics(geom)
    assert exc.value.code == "PLANE_UNIT_VIOLATION"


def test_glimmer_blinding_rejects_geometry_and_sam_keys() -> None:
    clean_ctx = {
        "observation_id": "22222222-2222-4222-8222-222222222222",
        "step_id": "attach-connecting-rod",
        "part_id": "connecting-rod",
        "frame_id": "frame-000120",
    }
    assert_glimmer_input_blinded(clean_ctx)

    for forbidden_key in ("geometry", "sam", "iou", "hard_checks_passed", "gate_decision"):
        leaked_ctx = dict(clean_ctx, **{forbidden_key: {"status": "pass"}})
        with pytest.raises(SemanticValidationError) as exc:
            assert_glimmer_input_blinded(leaked_ctx)
        assert exc.value.code == "GLIMMER_BLINDING_VIOLATION"


def test_atomic_cas_mismatch_rejected(example_cycle: list[dict]) -> None:
    mutated = copy.deepcopy(example_cycle)
    for env in mutated:
        if env["payload_type"] == "gate_decision":
            env["scene_version"] = 99
            env["payload"]["checked_scene_version"] = 99
            env["payload"]["committed_scene_version"] = 99
    with pytest.raises(SemanticValidationError) as exc:
        validate_cycle_semantics(mutated)
    assert exc.value.code == "CAS_MISMATCH"


def test_gate_cannot_accept_when_geometry_fails(example_cycle: list[dict]) -> None:
    mutated = copy.deepcopy(example_cycle)
    for env in mutated:
        if env["payload_type"] == "geometry":
            env["payload"]["status"] = "fail"
            env["payload"]["seating_status"] = "misaligned"
            env["payload"]["metrics"][0]["result"] = "fail"
            env["payload"]["reason_codes"] = ["PART_POSITION_OUT_OF_TOLERANCE"]
    with pytest.raises(SemanticValidationError) as exc:
        validate_cycle_semantics(mutated)
    assert exc.value.code == "ACCEPT_WITH_FAILED_GEOMETRY"


def test_gate_conflict_requires_unknown_decision(example_cycle: list[dict]) -> None:
    mutated = copy.deepcopy(example_cycle)
    for env in mutated:
        if env["payload_type"] == "visual_audit":
            env["payload"]["model_output"]["finding"] = "mismatch"
            env["payload"]["model_output"]["contradicted_predicate_ids"] = ["rod_orientation_valid"]
        if env["payload_type"] == "gate_decision":
            env["payload"]["decision"] = "rejected"
            env["payload"]["committed_scene_version"] = None
    with pytest.raises(SemanticValidationError) as exc:
        validate_cycle_semantics(mutated)
    assert exc.value.code == "MISSING_CONFLICT_OUTCOME"


def test_gate_deadline_exceeded_cannot_accept(example_cycle: list[dict]) -> None:
    gate = copy.deepcopy(next(e for e in example_cycle if e["payload_type"] == "gate_decision"))
    gate["payload"]["elapsed_ms"] = 3150.0  # > 2800ms
    with pytest.raises(SemanticValidationError) as exc:
        validate_envelope_semantics(gate)
    assert exc.value.code == "DEADLINE_EXCEEDED_NOT_TIMEOUT"


def test_compact_glimmer_and_wonder_schema_validation() -> None:
    glimmer_payload = {
        "observation_id": "22222222-2222-4222-8222-222222222222",
        "finding": "consistent",
        "observations": [
            {
                "frame_id": "frame-000120",
                "region_xyxy_px": [214, 151, 423, 223],
                "statement": "Connecting rod bridges crank disk and slider block cleanly.",
            }
        ],
        "contradicted_predicate_ids": [],
        "requires_new_view": False,
    }
    validate_glimmer_output(glimmer_payload)

    wonder_payload = {
        "observation_id": "22222222-2222-4222-8222-222222222222",
        "action": "generate_supported_part",
        "part_id": "connecting-rod",
        "instruction": "Wait for replacement connecting rod CAD generation.",
        "evidence_ids": ["frame-000120"],
        "required_predicate_ids": ["rod_crank_joint_seated", "rod_slider_joint_seated"],
        "measurement_record_id": "meas-flat-link-001",
    }
    validate_wonder_output(wonder_payload)
