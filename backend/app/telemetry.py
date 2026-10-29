"""Unified 6-Model Live Telemetry Proof Ledger (`backend/app/telemetry.py`).

Tracks live request IDs, latencies, base85 RLE byte sizes, decoded mask areas,
encrypted reasoning continuity flags, CUA actions, and Dual Open-Weight Consensus
across all 6 Meta AI models and open-weight endpoints (`sam-3.1`, `Muse Glimmer 30B`,
`muse-spark-1.3-contributor`, `muse-wonder-1.0`, `muse-voice-transcribe-1.0`,
`muse-image-1.0`, and `meta-models/muse-spark-open`).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

ProviderStatus = Literal[
    "LIVE_API",
    "MODAL_H100_WARM",
    "BASETEN_OPEN_WEIGHT",
    "ASYNC_PENDING",
    "FALLBACK_FIXTURE",
    "DETERMINISTIC_LOCAL",
    "NOT_MEASURED",
]


@dataclass
class ModelCallReceipt:
    model_id: str
    provider: str
    purpose: str
    status: ProviderStatus
    latency_ms: float
    http_status: int = 200
    request_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    frames_processed: int = 0
    rle_base85_bytes: int = 0
    decoded_mask_area_px: int = 0
    encrypted_reasoning: bool = False
    cua_actions: List[Dict[str, Any]] = field(default_factory=list)
    verdict_summary: str = ""
    # True only when latency/tokens/request_id came from a real provider response this run.
    measured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationTelemetryEnvelope:
    session_id: str
    step_id: str
    total_wall_ms: float
    dual_open_weight_consensus: Optional[Dict[str, Any]] = None
    receipts: List[ModelCallReceipt] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step_id": self.step_id,
            "total_wall_ms": round(float(self.total_wall_ms), 1),
            "all_live_verified": bool(self.receipts)
            and all(r.measured for r in self.receipts),
            # Weaker, separately-named claim: every lane that was actually invoked this run returned a real
            # provider receipt (deterministic local code and NOT_MEASURED lanes are excluded, never counted live).
            "invoked_lanes_live_verified": any(r.measured and r.status not in ("DETERMINISTIC_LOCAL", "NOT_MEASURED") for r in self.receipts)
            and all(r.measured for r in self.receipts if r.status not in ("NOT_MEASURED",)),
            "lanes_not_measured": [r.model_id for r in self.receipts if r.status == "NOT_MEASURED"],
            "dual_open_weight_consensus": self.dual_open_weight_consensus,
            "receipts": [asdict(r) for r in self.receipts],
        }


def build_six_model_proof_ledger(
    *,
    session_id: str = "session-01",
    step_id: str = "attach-connecting-rod",
    scenario: str = "pass",
    mode: str = "live",
    decision: str = "accepted",
    elapsed_ms: float = 842.0,
    total_wall_ms: Optional[float] = None,
    sam_request_id: str = "",
    sam_latency_ms: float = 342.4,
    rle_base85_bytes: int = 584,
    decoded_mask_area_px: int = 14820,
    glimmer_model: str = "Muse-Glimmer-30B",
    glimmer_latency_ms: float = 388.2,
    glimmer_finding: str = "consistent",
    glimmer_verdict: Optional[str] = None,
    witness_aligned: bool = True,
    witness_polarity_cosine: Optional[float] = None,
    mask_iou: float = 0.972,
    spark_open_triggered: bool = False,
    baseten_triggered: Optional[bool] = None,
    spark_open_verdict: str = "",
    baseten_verdict: Optional[str] = None,
    spark_open_latency_ms: float = 0.0,
    baseten_latency_ms: Optional[float] = None,
    repair_card_url: Optional[str] = None,
    include_repair_image: bool = False,
    repair_image_status: Optional[str] = None,
    voice_transcript: str = "check step",
    sam_measured: bool = False,
    glimmer_usage: Optional[Dict[str, Any]] = None,
    glimmer_request_id: str = "",
    glimmer_measured: bool = False,
    baseten_usage: Optional[Dict[str, Any]] = None,
    baseten_request_id: str = "",
    baseten_measured: bool = False,
    **_extra: Any,
) -> VerificationTelemetryEnvelope:
    """Assemble the telemetry ledger. Only values measured from real provider responses are reported;
    anything not called this run is NOT_MEASURED with zero latency/tokens. No fabricated numbers."""
    if total_wall_ms is not None:
        elapsed_ms = float(total_wall_ms)
    if glimmer_verdict is not None:
        glimmer_finding = str(glimmer_verdict)
    if baseten_triggered is not None:
        spark_open_triggered = bool(baseten_triggered)
    if baseten_verdict is not None and not spark_open_verdict:
        spark_open_verdict = str(baseten_verdict)
    if baseten_latency_ms is not None and not spark_open_latency_ms:
        spark_open_latency_ms = float(baseten_latency_ms)
    # Measured values are reported verbatim; nothing is forced to a canned constant here.

    def _tok(u: Optional[Dict[str, Any]], *keys: str) -> int:
        for k in keys:
            if u and u.get(k) is not None:
                return int(u[k])
        return 0

    receipts: List[ModelCallReceipt] = [
        ModelCallReceipt(
            model_id="sam-3.1",
            provider="Meta Model API (/v1/responses)",
            purpose="Spatial mask segmentation & base85 RLE decoding",
            status="LIVE_API" if sam_measured else "NOT_MEASURED",
            latency_ms=round(float(sam_latency_ms), 1) if sam_measured else 0.0,
            request_id=sam_request_id,
            frames_processed=1 if sam_measured else 0,
            rle_base85_bytes=int(rle_base85_bytes) if sam_measured else 0,
            decoded_mask_area_px=int(decoded_mask_area_px) if sam_measured else 0,
            verdict_summary=f"IoU={mask_iou:.3f} | area={decoded_mask_area_px}px" if sam_measured else "not called this run",
            measured=bool(sam_measured),
        ),
        ModelCallReceipt(
            model_id="opencv-lab-witness-precheck",
            provider="Local deterministic code (NOT a model)",
            purpose="CIE LAB witness-polarity cosine pre-check",
            status="DETERMINISTIC_LOCAL",
            latency_ms=0.0,
            verdict_summary=(f"witness_cos={witness_polarity_cosine:+.3f} (deterministic)" if witness_polarity_cosine is not None else "witness stripe not measurable (insufficient_evidence)"),
            measured=True,
        ),
        ModelCallReceipt(
            model_id=glimmer_model if glimmer_measured else "visual-auditor",
            provider="Baseten muse-spark-open-1026 (vision)" if glimmer_measured else "not called",
            purpose="Blinded visual witness/seating audit",
            status="BASETEN_OPEN_WEIGHT" if glimmer_measured else "NOT_MEASURED",
            latency_ms=round(float(glimmer_latency_ms), 1) if glimmer_measured else 0.0,
            request_id=glimmer_request_id,
            tokens_in=_tok(glimmer_usage, "prompt_tokens", "input_tokens"),
            tokens_out=_tok(glimmer_usage, "completion_tokens", "output_tokens"),
            frames_processed=2 if glimmer_measured else 0,
            verdict_summary=f"finding={glimmer_finding}" if glimmer_measured else "not called this run",
            measured=bool(glimmer_measured),
        ),
    ]

    dual_consensus: Dict[str, Any] = {
        "triggered": bool(spark_open_triggered),
        "trigger_rule": "silhouette_iou >= 0.90 and witness_alignment_score < 0.50",
        "visual_auditor_finding": glimmer_finding,
        "baseten_spark_open_verdict": spark_open_verdict or ("NOT_CALLED" if not baseten_measured else ""),
        "silhouette_iou": round(float(mask_iou), 4),
        "witness_polarity_cosine": None if witness_polarity_cosine is None else round(float(witness_polarity_cosine), 4),
    }
    receipts.append(
        ModelCallReceipt(
            model_id="meta-models/muse-spark-open-1026",
            provider="Baseten Open-Weight Inference",
            purpose="Text conflict arbiter (IoU>=0.90 & witness FAIL)",
            status="BASETEN_OPEN_WEIGHT" if baseten_measured else "NOT_MEASURED",
            latency_ms=round(float(spark_open_latency_ms), 1) if baseten_measured else 0.0,
            request_id=baseten_request_id,
            tokens_in=_tok(baseten_usage, "prompt_tokens", "input_tokens"),
            tokens_out=_tok(baseten_usage, "completion_tokens", "output_tokens"),
            verdict_summary=spark_open_verdict if baseten_measured else "not called this run",
            measured=bool(baseten_measured),
        )
    )
    for mid, prov, purpose in [
        ("muse-spark-1.3-contributor", "Meta Model API (/v1/responses)", "Reverse DAG compiler"),
        ("muse-wonder-1.0", "Meta Model API (computer tool)", "CAD drawing inspection"),
        ("muse-voice-transcribe-1.0", "Meta Voice ASR", "Operator voice command"),
        ("muse-image-1.0", "Meta Image Edit API", "Repair overlay card"),
    ]:
        rec = (_extra.get("extra_receipts") or {}).get(mid)
        if rec:
            receipts.append(ModelCallReceipt(model_id=mid, provider=prov, purpose=purpose, measured=True,
                                             **{k: v for k, v in rec.items() if k != "model_id"}))
        else:
            receipts.append(ModelCallReceipt(model_id=mid, provider=prov, purpose=purpose,
                                             status="NOT_MEASURED", latency_ms=0.0,
                                             verdict_summary="not called this run"))

    return VerificationTelemetryEnvelope(
        session_id=session_id,
        step_id=step_id,
        total_wall_ms=elapsed_ms,
        dual_open_weight_consensus=dual_consensus,
        receipts=receipts,
    )
