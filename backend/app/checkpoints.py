"""Per-step checkpoint configuration loader (furniture reframe).

Step ids, SAM concept prompts, reference images and witness parameters live in JSON
(`fixtures/ikea/checkpoints.json` by default, override with REWIND_CHECKPOINTS_JSON) instead of
hard-coded part names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_JSON = REPO_ROOT / "fixtures" / "ikea" / "checkpoints.json"


@dataclass(frozen=True)
class CheckpointConfig:
    step_id: str
    step_number: int
    total_steps: int
    title: str
    part_label: str
    sam_concept_prompt: str
    reference_image: Path
    audit_profile: str
    witness: Dict[str, Any] = field(default_factory=dict)
    panel_iou_min: float = 0.80
    sam_secondary_concept: Optional[str] = None

    @property
    def display_label(self) -> str:
        return f"Step {self.step_number} of {self.total_steps} · {self.part_label}"


@dataclass(frozen=True)
class CheckpointSet:
    profile: str
    product: str
    synthetic: bool
    steps: List[CheckpointConfig]
    scenarios: Dict[str, Path]

    @property
    def total_steps(self) -> int:
        return self.steps[0].total_steps

    def step(self, step_id: Optional[str] = None) -> CheckpointConfig:
        if step_id is None:
            return self.steps[0]
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(f"unknown checkpoint step_id {step_id!r}")


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_checkpoint_set(path: Optional[Path | str] = None) -> CheckpointSet:
    src = Path(path or os.environ.get("REWIND_CHECKPOINTS_JSON") or DEFAULT_CHECKPOINTS_JSON)
    data = json.loads(src.read_text())
    total = int(data.get("total_steps", len(data["steps"])))
    steps = []
    for s in data["steps"]:
        if not s.get("sam_concept_prompt"):
            raise ValueError(f"step {s.get('step_id')} has no sam_concept_prompt")
        steps.append(CheckpointConfig(
            step_id=s["step_id"], step_number=int(s["step_number"]), total_steps=total,
            title=s["title"], part_label=s["part_label"], sam_concept_prompt=s["sam_concept_prompt"],
            reference_image=_resolve(s["reference_image"]), audit_profile=s.get("audit_profile", "furniture_panel"),
            witness=dict(s.get("witness", {})), panel_iou_min=float(s.get("panel_iou_min", 0.80)),
            sam_secondary_concept=s.get("sam_secondary_concept"),
        ))
    if not steps:
        raise ValueError("checkpoint config has no steps")
    return CheckpointSet(profile=data.get("profile", "furniture_panel"), product=data.get("product", ""),
                         synthetic=bool(data.get("synthetic", False)), steps=steps,
                         scenarios={k: _resolve(v) for k, v in data.get("scenarios", {}).items()})


@lru_cache(maxsize=1)
def default_checkpoint_set() -> CheckpointSet:
    return load_checkpoint_set()
