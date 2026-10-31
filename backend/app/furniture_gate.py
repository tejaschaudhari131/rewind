"""Two-stage checkpoint gate for the flat-pack furniture reframe (panel face / orientation).

Stage 1 (provisional, local, fast): panel silhouette IoU + hole-pattern face witness + tape-edge polarity
witness (all DETERMINISTIC_LOCAL). In live mode, sam-3.1 is also called with the configured concept prompt
(e.g. "wooden panel"); a SAM failure or zero instances makes stage 1 `unknown` (fail closed).
Stage 2 (live only): blinded Baseten muse-spark-open audit on panel crops with the neutral furniture prompt.
Final rules are shared with the slider-crank gate (`two_stage.final_from_stage2`): never auto-accept,
accept only when both stages agree; fixture mode never runs stage 2 and is labelled fixture-only.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from .checkpoints import CheckpointConfig
from .geometry import (check_hole_pattern_witness, check_tape_edge_witness, crop_panel_png, panel_mask_iou)
from .two_stage import final_from_stage2

logger = logging.getLogger("rewind.furniture_gate")
STAGE2_DEADLINE_S = 20.0


def provisional_from_witnesses(iou: Optional[float], holes: Dict[str, Any], tape: Dict[str, Any],
                               iou_min: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if iou is None:
        return "unknown", ["panel_not_found"]
    statuses = (holes["status"], tape["status"])
    if "fail" in statuses:
        if holes["status"] == "fail":
            reasons.append("panel_face_flipped_holes_not_visible")
        if tape["status"] == "fail":
            reasons.append("tape_edge_reversed")
        return "provisional_rejected", reasons
    if iou < iou_min:
        return "provisional_rejected", ["panel_silhouette_mismatch"]
    if statuses == ("pass", "pass"):
        return "provisional_accepted", []
    reasons = [f"{w['check']}:{w.get('reason')}" for w in (holes, tape) if w["status"] != "pass"]
    return "unknown", reasons + ["witness_insufficient_evidence"]


async def run_stage1(frame_path: Path, cfg: CheckpointConfig, *, mode: str = "fixture",
                     sam_client: Any = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ref = cv2.imread(str(cfg.reference_image), cv2.IMREAD_COLOR)
    live = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if ref is None or live is None:
        return {"stage": 1, "provisional_verdict": "unknown", "provisional_reason_codes": ["frame_unreadable"],
                "stage1_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
    w = cfg.witness
    holes = check_hole_pattern_witness(ref, live, w.get("hole_pattern"))
    tape = check_tape_edge_witness(ref, live, w.get("tape_edge"))
    iou = panel_mask_iou(ref, live)
    provisional, reasons = provisional_from_witnesses(iou, holes, tape, cfg.panel_iou_min)
    sam_info: Dict[str, Any] = {"called": False, "measured": False, "concept_prompt": cfg.sam_concept_prompt}
    if mode == "live":
        if sam_client is None:
            from .providers.meta_sam import MetaSamClient
            sam_client = MetaSamClient()
        sam_info["called"] = True
        try:
            res = await sam_client.segment_image(frame_path, prompt_text=cfg.sam_concept_prompt, part_id=cfg.step_id)
            sam_info.update(measured=True, instances=len(res.instances or []), latency_ms=round(res.latency_ms, 1),
                            attempts=list(getattr(sam_client, "last_attempts", []) or []))
            if not res.instances:
                reasons = reasons + ["sam_no_instances"]
                if provisional == "provisional_accepted":  # SAM can only withhold a pass, never mask a STOP
                    provisional = "unknown"
        except Exception as exc:  # fail closed
            sam_info.update(error=f"{type(exc).__name__}: {exc}"[:300],
                            attempts=list(getattr(sam_client, "last_attempts", []) or []))
            reasons = reasons + ["sam_provider_error"]
            if provisional == "provisional_accepted":
                provisional = "unknown"
    if provisional == "unknown":
        logger.warning("furniture stage1 unknown step=%s reasons=%s sam=%s", cfg.step_id, reasons,
                       {k: sam_info.get(k) for k in ("called", "instances", "error", "attempts")})
    return {
        "stage": 1, "gate": "two_stage_furniture", "decision": "pending", "final": False,
        "observation_id": str(uuid.uuid4()), "step_id": cfg.step_id, "step_label": cfg.display_label,
        "step_title": cfg.title, "provisional_verdict": provisional, "provisional_reason_codes": reasons,
        "panel_silhouette_iou": iou, "hole_pattern_witness": holes, "tape_edge_witness": tape, "sam": sam_info,
        "stage1_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1), "mode": mode,
    }


async def run_stage2(frame_path: Path, cfg: CheckpointConfig, stage1: Dict[str, Any], *, mode: str,
                     auditor: Any = None) -> Dict[str, Any]:
    provisional = stage1["provisional_verdict"]
    if mode != "live":
        return {"stage": 2, "final": True, "status": "not_run", "fixture_only": True, "decision": "unknown"
                if provisional != "provisional_rejected" else "rejected",
                "resolution": "stage2_not_run_fixture", "model_finding": None, "measured": False,
                "note": "Fixture mode: no model call; a provisional pass is never confirmed."}
    ref_png = crop_panel_png(cv2.imread(str(cfg.reference_image)))
    live_png = crop_panel_png(cv2.imread(str(frame_path)))
    t0 = time.perf_counter()
    finding, error, usage, latency = None, None, {}, None
    if ref_png is None or live_png is None:
        error = "panel_crop_unavailable"
    else:
        if auditor is None:
            from .providers.baseten_vision import BasetenVisionAuditor
            auditor = BasetenVisionAuditor(max_tokens=1536)
        ctx = {"observation_id": stage1["observation_id"], "frame_id": "furniture-live"}
        try:
            res = await asyncio.wait_for(auditor.audit_checkpoint(current_image=live_png, reference_image=ref_png,
                                         blinded_context=ctx, audit_profile=cfg.audit_profile), STAGE2_DEADLINE_S)
            finding, usage, latency = res.audit["finding"], dict(res.usage or {}), res.latency_ms
        except asyncio.TimeoutError:
            error = "stage2_timeout"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
    decision, resolution, needs_review, reasons = final_from_stage2(provisional, finding)
    if provisional == "provisional_rejected" and finding != "consistent":
        # Deterministic local witnesses saw the wrong face; an unsure/failed model must not lift the STOP.
        decision, resolution, needs_review = "rejected", "local_witness_reject_upheld", False
    return {"stage": 2, "final": True, "status": "committed" if error is None else "error",
            "decision": decision, "resolution": resolution, "needs_review": needs_review, "reason_codes": reasons,
            "model_finding": finding, "model": "meta-models/muse-spark-open-1026", "measured": error is None,
            "latency_ms": latency, "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "error": error,
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "request_id")
                      if k in usage}, "raw_verdict": usage.get("raw_verdict")}


async def verify_furniture_checkpoint(frame_path: Path, cfg: CheckpointConfig, *, mode: str = "fixture",
                                      sam_client: Any = None, auditor: Any = None) -> Dict[str, Any]:
    s1 = await run_stage1(frame_path, cfg, mode=mode, sam_client=sam_client)
    if "observation_id" not in s1:
        return {"stage1": s1, "stage2": None, "decision": "unknown", "final": True}
    s2 = await run_stage2(frame_path, cfg, s1, mode=mode, auditor=auditor)
    out = {"stage1": s1, "stage2": s2, "decision": s2["decision"], "final": True, "mode": mode,
           "step_label": cfg.display_label}
    if mode != "live":
        out["fixture_only"] = True
        if s1["provisional_verdict"] == "provisional_accepted":
            out["verdict_label"] = "PROVISIONAL PASS (fixture only: stage 2 model confirmation not run)"
    return out
