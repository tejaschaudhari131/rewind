"""Two-stage checkpoint gate (decision 2026-10-29).

Stage 1 (fast, < 2.8 s): live SAM 3.1 (or fixture SAM) + deterministic local checks (silhouette IoU,
CIE LAB witness-polarity cosine, registration). Produces a PROVISIONAL verdict only:
``provisional_accepted`` / ``provisional_rejected`` / ``unknown``. Never final, never committed.

Stage 2 (async, deadline ~15 s): live Baseten ``meta-models/muse-spark-open-1026`` blinded vision audit
(+ consensus arbiter when silhouette and witness disagree). Final decision rules (fail closed):

* model ``consistent`` AND stage 1 ``provisional_accepted``            -> ``accepted`` (confirmed)
* model ``mismatch`` AND stage 1 ``provisional_rejected`` / ``unknown`` -> ``rejected`` (confirmed)
* model ``mismatch`` AND stage 1 ``provisional_accepted``               -> ``unknown`` + needs_review (overridden)
* model ``consistent`` AND stage 1 not accepted                         -> ``unknown`` + needs_review (never auto-accept)
* timeout / error / unreadable / ``insufficient_evidence``              -> ``unknown`` (fail closed)

Stage 2 commits through the same CAS token captured in stage 1; if the session was reset, the camera
moved, or a newer observation superseded it, the late result is DISCARDED (status ``discarded_stale``).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import numpy as np

from backend.app.gate import run_checkpoint_gate
from backend.app.providers.baseten_spark import BasetenSparkClient
from backend.app.providers.baseten_vision import BasetenVisionAuditor
from backend.app.state_machine import SessionStateMachine

STAGE1_DEADLINE_MS = 2800.0
STAGE2_DEADLINE_MS = float(os.getenv("REWIND_STAGE2_DEADLINE_MS", "20000"))
# Paired blinded crops (~475 prompt tokens) instead of full frames (~2,805). Thinking used 737-936
# completion tokens on full frames; 1536 leaves headroom, truncation -> insufficient_evidence (fail closed).
STAGE2_MAX_TOKENS = int(os.getenv("REWIND_STAGE2_MAX_TOKENS", "1536"))
MIN_IOU = 0.82
MIN_WITNESS_COS = 0.65
MAX_WITNESS_ERR_PX = 6.0


def provisional_from_stage1(outcome) -> tuple[str, list[str]]:
    """Map deterministic stage-1 evidence to a provisional verdict (never final)."""
    g = outcome.geometry_metrics
    reasons = [r for r in outcome.reason_codes if r != "UNKNOWN_TIMEOUT"]
    if any(r in outcome.reason_codes for r in ("UNKNOWN_TIMEOUT", "PROVIDER_ERROR", "sam_no_instances",
                                                "sam_ambiguous_instances", "sam_empty_mask",
                                                "UNSUPPORTED_STEP_VERIFIER", "scene_not_stationary")):
        return "unknown", list(outcome.reason_codes)
    if not g.calibration_valid or "registration_lost" in outcome.reason_codes:
        return "provisional_rejected", ["registration_lost"]
    cos = g.witness_polarity_cosine
    if cos is None:
        return "unknown", reasons + ["witness_insufficient_evidence"]
    if g.mask_iou < MIN_IOU:
        return "provisional_rejected", reasons + ["mask_iou_below_threshold"]
    if cos < 0.0 or g.witness_error_px > MAX_WITNESS_ERR_PX:
        return "provisional_rejected", sorted(set(reasons + ["witness_mark_misaligned"]))
    if cos >= MIN_WITNESS_COS and g.status == "pass":
        return "provisional_accepted", []
    return "unknown", reasons + ["witness_ambiguous"]


def final_from_stage2(provisional: str, finding: Optional[str]) -> tuple[str, str, bool, list[str]]:
    """Return (decision, resolution, needs_review, reason_codes). Never auto-accepts."""
    if finding == "consistent":
        if provisional == "provisional_accepted":
            return "accepted", "confirmed", False, []
        return "unknown", "overridden", True, ["model_local_disagreement_needs_review"]
    if finding == "mismatch":
        if provisional == "provisional_accepted":
            return "unknown", "overridden", True, ["model_override_needs_review", "witness_mark_misaligned"]
        return "rejected", "confirmed", False, ["witness_mark_misaligned", "model_confirmed_mismatch"]
    return "unknown", "unconfirmed", False, ["stage2_insufficient_evidence"]


async def run_two_stage_gate(
    image: Union[str, Path, np.ndarray],
    *,
    state_machine: SessionStateMachine,
    mode: str = "fixture",
    sam_client: Optional[Any] = None,
    audit_client: Optional[Any] = None,
    arbiter_client: Optional[Any] = None,
    stage1_deadline_ms: float = STAGE1_DEADLINE_MS,
    stage2_deadline_ms: float = STAGE2_DEADLINE_MS,
    on_stage2: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    motion_score: Optional[float] = None,
) -> tuple[Dict[str, Any], "asyncio.Task[Dict[str, Any]]"]:
    """Run stage 1 now and schedule stage 2. Returns (stage1_payload, stage2_task)."""
    if sam_client is None and mode in ("live", "hybrid"):
        from backend.app.providers.meta_sam import MetaSamClient
        sam_client = MetaSamClient()
    t0 = time.perf_counter()
    outcome = await run_checkpoint_gate(
        image, state_machine=state_machine, mode=mode, sam_client=sam_client,
        provider_deadline_ms=stage1_deadline_ms, visual_auditor="local", defer_commit=True,
        motion_score=motion_score,
    )
    stage1_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    provisional, p_reasons = provisional_from_stage1(outcome)
    d = outcome.deferred or {}
    g = outcome.geometry_metrics
    stage1 = {
        "stage": 1,
        "gate": "two_stage",
        "decision": "pending",
        "final": False,
        "provisional_verdict": provisional,
        "provisional_reason_codes": p_reasons,
        "stage1_elapsed_ms": stage1_ms,
        "stage1_within_deadline": stage1_ms <= stage1_deadline_ms,
        "session_id": d.get("session_id"),
        "step_id": d.get("step_id"),
        "observation_id": d.get("observation_id"),
        "silhouette_iou": round(float(g.mask_iou), 4),
        "witness_polarity_cosine": g.witness_polarity_cosine,
        "witness_check_status": g.witness_check_status,
        "witness_error_px": round(float(g.witness_error_px), 2),
        "registration_rmse_px": outcome.calibration.registration_rmse_px,
        "calibration_valid": outcome.calibration.calibration_valid,
        "sam": {
            "measured": bool(d.get("sam_measured")),
            "latency_ms": round(float(getattr(d.get("sam_result"), "latency_ms", 0.0) or 0.0), 1),
            "instances": len(getattr(d.get("sam_result"), "instances", []) or []),
            "attempts": list(getattr(sam_client, "last_attempts", []) or []),
        },
        "paired_crop_ref_b64": g.paired_crop_ref_b64,
        "paired_crop_live_b64": g.paired_crop_live_b64,
        "mode": mode,
    }

    async def _stage2() -> Dict[str, Any]:
        s2_t0 = time.perf_counter()
        audit_res = None
        audit_input = "none"
        error = None
        finding: Optional[str] = None
        live = mode == "live"
        cos = g.witness_polarity_cosine
        arbiter_task = None
        # Consensus check runs CONCURRENTLY with the vision audit when the local lanes disagree / are unmeasurable.
        local_disagree = cos is None or (g.mask_iou >= MIN_IOU and cos < MIN_WITNESS_COS)
        if p_reasons != ["registration_lost"] and local_disagree and (live or arbiter_client is not None):
            arb = arbiter_client or BasetenSparkClient()
            arbiter_task = asyncio.create_task(arb.arbitrate_ambiguous_checkpoint(
                session_id=str(d.get("session_id")), step_id=str(d.get("step_id")),
                silhouette_iou=float(g.mask_iou), witness_polarity_cosine=cos,
                witness_error_px=float(g.witness_error_px), glimmer_finding="not_yet_available", mode=mode,
            ))
        try:
            if p_reasons == ["registration_lost"]:
                # Measurement frame is invalid: deterministic hard reject, no model needed (fail closed).
                error = "registration_lost_no_model_call"
            elif live or audit_client is not None:
                auditor = audit_client or BasetenVisionAuditor(max_tokens=STAGE2_MAX_TOKENS)
                cur_img, ref_img = d["raw_bytes"], d["reference_image_bytes"]
                if g.paired_crop_ref_b64 and g.paired_crop_live_b64:
                    import base64 as _b64
                    def _dec(x: str) -> bytes:
                        return _b64.b64decode(x.split(",", 1)[1] if x.startswith("data:") else x)
                    cur_img, ref_img = _dec(g.paired_crop_live_b64), _dec(g.paired_crop_ref_b64)
                    audit_input = "paired_blinded_crops"
                else:
                    audit_input = "full_frames"
                audit_res = await asyncio.wait_for(
                    auditor.audit_checkpoint(
                        current_image=cur_img, reference_image=ref_img,
                        blinded_context=d["blinded_context"],
                    ),
                    timeout=stage2_deadline_ms / 1000.0,
                )
                finding = str(audit_res.audit.get("finding", "insufficient_evidence"))
            else:
                # Fixture mode: no model is called. Stage 2 cannot confirm anything -> stays unknown.
                error = "fixture_mode_no_model_call"
        except asyncio.TimeoutError:
            error = "stage2_timeout"
        except Exception as exc:  # network / HTTP / parse
            error = f"stage2_error:{type(exc).__name__}"
        if finding not in ("consistent", "mismatch"):
            finding_for_rules = None
        else:
            finding_for_rules = finding

        arbiter: Dict[str, Any] = {"triggered": False}
        if arbiter_task is not None:
            remaining = max(0.05, stage2_deadline_ms / 1000.0 - (time.perf_counter() - s2_t0))
            try:
                arbiter = await asyncio.wait_for(arbiter_task, timeout=remaining)
            except asyncio.TimeoutError:
                arbiter = {"triggered": True, "verdict": "TIMEOUT", "measured": False}
            except Exception as exc:
                arbiter = {"triggered": True, "verdict": f"ERROR:{type(exc).__name__}", "measured": False}

        if p_reasons == ["registration_lost"]:
            decision, resolution, needs_review, reasons = "rejected", "deterministic_registration_lost", False, ["registration_lost"]
            error = None
        else:
            decision, resolution, needs_review, reasons = final_from_stage2(provisional, finding_for_rules)
        av = str(arbiter.get("verdict", "")).upper()
        # A consensus check that explicitly says CONSISTENT contradicts a rejection -> not confirmed.
        # (A timed-out/errored consensus check cannot rescue a part; rejection stands, it is the safe side.)
        if arbiter.get("triggered") and decision == "rejected" and av == "CONSISTENT":
            decision, resolution, needs_review = "unknown", "unconfirmed", True
            reasons = reasons + ["arbiter_not_confirming"]
        if error:
            reasons = reasons + [error]
        stage2_ms = round((time.perf_counter() - s2_t0) * 1000.0, 1)

        committed, genv, snap = d["commit_final"](decision, reasons, audit_res, stage2_ms)
        status = "committed" if committed else "discarded_stale"
        if not committed:
            decision = "unknown"
        usage = dict((audit_res.usage if audit_res else {}) or {})
        raw = usage.pop("raw_verdict", {}) or {}
        payload = {
            "stage": 2,
            "gate": "two_stage",
            "final": committed,
            "status": status,
            "session_id": d.get("session_id"),
            "step_id": d.get("step_id"),
            "observation_id": d.get("observation_id"),
            "provisional_verdict": provisional,
            "decision": decision,
            "resolution": resolution if committed else "discarded",
            "needs_review": bool(needs_review) and committed,
            "reason_codes": reasons,
            "model_finding": finding,
            "model_reason": str(raw.get("statement", ""))[:500],
            "model_confidence": raw.get("confidence"),
            "auditor": {
                "model": getattr(audit_res, "model_used", None),
                "measured": audit_res is not None and bool(usage.get("request_id")),
                "latency_ms": getattr(audit_res, "latency_ms", 0.0) if audit_res else 0.0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "request_id": usage.get("request_id", ""),
                "input": audit_input,
                "finish_reason": usage.get("finish_reason"),
                "max_tokens": usage.get("max_tokens"),
            },
            "arbiter": {k: arbiter.get(k) for k in ("triggered", "verdict", "consensus", "latency_ms", "measured",
                                                    "request_id", "explanation") if k in arbiter}
                       | {"usage": arbiter.get("usage") or {}},
            "stage1_elapsed_ms": stage1_ms,
            "stage2_elapsed_ms": stage2_ms,
            "stage2_deadline_ms": stage2_deadline_ms,
            "gate_envelope_message_id": genv.get("message_id"),
            "session_snapshot": snap,
        }
        if on_stage2 is not None:
            try:
                await on_stage2(payload)
            except Exception:
                pass
        return payload

    task = asyncio.create_task(_stage2())
    return stage1, task
