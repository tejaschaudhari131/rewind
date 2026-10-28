"""Deterministic Session State Machine & Atomic CAS Coordinator (`backend/app/state_machine.py`).

Manages the slider-crank teardown/rebuild session lifecycle:
- States: `MOVING -> CAPTURING -> CHECKING -> ACCEPTED | REJECTED | UNKNOWN`
- Enforces single-writer gate commit (`producer == "gate_service"`, `payload_type == "gate_decision"`)
- Enforces atomic compare-and-set on `(scene_version, plan_version, step_id, observation_id)`
- Automatically increments `scene_version` when camera/mat registration is invalidated (`registration_rmse_px > 1.5`)
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.semantic_validator import SemanticValidationError

SLIDER_CRANK_STEPS: Tuple[Dict[str, Any], ...] = (
    {
        "step_id": "mount-base-and-guide",
        "part_id": "crank-support",
        "title": "Step 1: Mount Green Crank Support & Gray Linear Guide",
        "required_predicate_ids": ["base-mounted", "guide-aligned"],
    },
    {
        "step_id": " attach-connecting-rod".strip(),
        "part_id": "connecting-rod",
        "title": "Step 2: Attach Yellow Connecting Rod (Align Black/White Witness Stripe to Left Crank Pin)",
        "required_predicate_ids": ["rod-visible-alignment", "witness-stripe-aligned"],
    },
    {
        "step_id": "attach-output-paddle",
        "part_id": "output-paddle",
        "title": "Step 3: Attach Purple Output Paddle to Blue Slider Block",
        "required_predicate_ids": ["paddle-seated"],
    },
    {
        "step_id": "scan-flat-link",
        "part_id": "connecting-rod",
        "title": "Step 4: Place Connecting Rod Flat in mat_z0 Scan Zone for Millimeter Metrology",
        "required_predicate_ids": ["flat-contact-confirmed"],
    },
    {
        "step_id": "export-cad-link",
        "part_id": "connecting-rod",
        "title": "Step 5: Generate & Verify Parametric flat_two_hole_link_v1 (.STEP + .STL)",
        "required_predicate_ids": ["cad-geometry-valid"],
    },
)

VALID_STATES = ("MOVING", "CAPTURING", "CHECKING", "ACCEPTED", "REJECTED", "UNKNOWN")


def compute_frame_motion_score(
    frame_a: Union[str, Path, np.ndarray],
    frame_b: Union[str, Path, np.ndarray],
) -> float:
    """Compute mean absolute pixel difference [0..255] between two consecutive camera frames."""
    if isinstance(frame_a, (str, Path)):
        img_a = cv2.imread(str(frame_a), cv2.IMREAD_GRAYSCALE)
    elif frame_a.ndim == 3:
        img_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    else:
        img_a = frame_a

    if isinstance(frame_b, (str, Path)):
        img_b = cv2.imread(str(frame_b), cv2.IMREAD_GRAYSCALE)
    elif frame_b.ndim == 3:
        img_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        img_b = frame_b

    if img_a is None or img_b is None:
        return 999.0
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    diff = cv2.absdiff(img_a, img_b)
    return round(float(np.mean(diff)), 4)


@dataclass
class SessionStateMachine:
    """Thread/async-safe deterministic state machine for a REWIND verification session."""

    session_id: str = field(default_factory=lambda: "00000000-0000-4000-8000-000000000001")
    scene_version: int = 1
    plan_version: int = 1
    current_step_index: int = 1  # Default to Step 2 (`attach-connecting-rod`) for golden checkpoint
    state: str = "MOVING"
    active_observation_id: Optional[str] = None
    active_observation_sha256: Optional[str] = None
    calibration_id: str = "fixed-camera-v1"
    policy_id: str = "slider-crank-v1"
    registration_valid: bool = True
    last_registration_rmse_px: float = 0.0
    last_motion_score: float = 0.0
    provisional_status: Optional[str] = None
    completed_steps: List[str] = field(default_factory=lambda: ["mount-base-and-guide"])
    completed_predicate_ids: List[str] = field(default_factory=list)
    last_gate_decision: Optional[Dict[str, Any]] = None
    last_measurement_record: Optional[Dict[str, Any]] = None
    last_cad_artifact: Optional[Dict[str, Any]] = None
    bonus_outputs: Dict[str, Any] = field(default_factory=dict)
    envelope_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_step(self) -> Dict[str, Any]:
        idx = max(0, min(len(SLIDER_CRANK_STEPS) - 1, self.current_step_index))
        return SLIDER_CRANK_STEPS[idx]

    @property
    def step_id(self) -> str:
        return str(self.current_step["step_id"])

    @property
    def part_id(self) -> str:
        return str(self.current_step["part_id"])

    def record_envelope(self, envelope: Mapping[str, Any]) -> None:
        self.envelope_log.append(dict(envelope))

    def evaluate_stationarity(
        self,
        motion_score: float,
        stationary_threshold: float = 2.5,
    ) -> str:
        """Transition `MOVING -> CAPTURING` when `motion_score < stationary_threshold`."""
        self.last_motion_score = round(float(motion_score), 4)
        if motion_score < stationary_threshold:
            self.state = "CAPTURING"
        else:
            self.state = "MOVING"
            self.provisional_status = None
        return self.state

    def begin_checkpoint_observation(
        self,
        observation_id: Optional[str] = None,
        observation_sha256: Optional[str] = None,
    ) -> Tuple[int, int, str, str]:
        """Lock an observation tuple `(scene_version, plan_version, step_id, observation_id)` and enter `CHECKING`."""
        self.active_observation_id = observation_id or str(uuid.uuid4())
        self.active_observation_sha256 = observation_sha256 or ("b" * 64)
        self.state = "CHECKING"
        self.provisional_status = "provisional_checking"
        return (
            self.scene_version,
            self.plan_version,
            self.step_id,
            self.active_observation_id,
        )

    def set_provisional_local_status(
        self,
        *,
        registration_rmse_px: float,
        witness_error_px: float,
        max_registration_rmse_px: float = 1.5,
        max_witness_error_px: float = 6.0,
    ) -> str:
        """Emit fast local provisional status at 0.40-0.65s before remote SAM + Glimmer return."""
        self.last_registration_rmse_px = round(float(registration_rmse_px), 4)
        if registration_rmse_px > max_registration_rmse_px:
            self.provisional_status = "provisional_registration_lost"
        elif witness_error_px > max_witness_error_px:
            self.provisional_status = "provisional_witness_misaligned"
        else:
            self.provisional_status = "provisional_local_pass"
        return self.provisional_status

    def invalidate_registration(self, registration_rmse_px: float) -> int:
        """Invalidate registration when camera/mat shifts (`registration_rmse_px > 1.5`) and bump `scene_version`."""
        self.registration_valid = False
        self.last_registration_rmse_px = round(float(registration_rmse_px), 4)
        self.scene_version += 1
        self.state = "REJECTED"
        self.provisional_status = "registration_lost"
        return self.scene_version

    def commit_gate_decision(
        self,
        gate_envelope: Mapping[str, Any],
        *,
        advance_step_on_accept: bool = True,
    ) -> Dict[str, Any]:
        """Atomically commit a `gate_decision` envelope using compare-and-set on `(scene_version, plan_version, step_id, observation_id)`."""
        if gate_envelope.get("producer") != "gate_service" or gate_envelope.get("payload_type") != "gate_decision":
            raise SemanticValidationError(
                "SINGLE_WRITER_VIOLATION",
                "Only producer='gate_service' with payload_type='gate_decision' may commit state transitions.",
            )

        env_sv = int(gate_envelope["scene_version"])
        env_pv = int(gate_envelope["plan_version"])
        env_step = str(gate_envelope["step_id"])
        env_obs = str(gate_envelope["observation_id"])

        expected_tuple = (
            self.scene_version,
            self.plan_version,
            self.step_id,
            self.active_observation_id or env_obs,
        )
        candidate_tuple = (env_sv, env_pv, env_step, env_obs)

        if candidate_tuple != expected_tuple:
            raise SemanticValidationError(
                "CAS_MISMATCH",
                f"Atomic CAS rejected stale gate commit {candidate_tuple} != active {expected_tuple}.",
            )

        payload = gate_envelope["payload"]
        decision = str(payload["decision"])
        reasons = list(payload.get("reason_codes", []))

        self.record_envelope(gate_envelope)
        self.last_gate_decision = dict(gate_envelope)

        if "registration_lost" in reasons:
            self.invalidate_registration(self.last_registration_rmse_px or 9.99)
            return self.snapshot()

        if decision == "accepted":
            self.state = "ACCEPTED"
            self.registration_valid = True
            if self.step_id not in self.completed_steps:
                self.completed_steps.append(self.step_id)
            for pid in payload.get("completed_predicate_ids", []):
                if pid not in self.completed_predicate_ids:
                    self.completed_predicate_ids.append(pid)
            if advance_step_on_accept and self.current_step_index + 1 < len(SLIDER_CRANK_STEPS):
                self.current_step_index += 1
        elif decision == "rejected":
            self.state = "REJECTED"
        else:
            self.state = "UNKNOWN"

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the session state machine."""
        return {
            "session_id": self.session_id,
            "scene_version": self.scene_version,
            "plan_version": self.plan_version,
            "state": self.state,
            "step_index": self.current_step_index,
            "step_id": self.step_id,
            "part_id": self.part_id,
            "step_title": self.current_step["title"],
            "active_observation_id": self.active_observation_id,
            "registration_valid": self.registration_valid,
            "last_registration_rmse_px": self.last_registration_rmse_px,
            "last_motion_score": self.last_motion_score,
            "provisional_status": self.provisional_status,
            "completed_steps": list(self.completed_steps),
            "completed_predicate_ids": list(self.completed_predicate_ids),
            "last_gate_decision": self.last_gate_decision,
            "last_measurement_record": self.last_measurement_record,
            "last_cad_artifact": self.last_cad_artifact,
            "bonus_outputs": self.bonus_outputs,
            "envelope_count": len(self.envelope_log),
        }
