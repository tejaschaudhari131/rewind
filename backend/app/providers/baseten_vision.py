"""Live blinded visual auditor on Baseten `meta-models/muse-spark-open-1026` (OpenAI-compatible).

Sends blinded image(s) as `image_url` base64 parts and parses a strict JSON verdict.
Latency, usage tokens and request id are taken verbatim from the provider response;
nothing is fabricated. If the endpoint rejects image input, the caller must fall back
(see gate.py) and the telemetry ledger marks the auditor NOT_MEASURED.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Mapping

import httpx

from ..config import settings
from ..schemas import validate_glimmer_output
from ..semantic_validator import assert_glimmer_input_blinded
from .baseten_spark import extract_spark_open_content_and_json
from .modal_glimmer import GlimmerAuditResult

AUDIT_PROMPT = (
    "You are a blinded visual QA auditor for a physical slider-crank reassembly. "
    "Image 1 is the REFERENCE crop of the connecting rod in its correct orientation; image 2 is the LIVE crop. "
    "The rod is asymmetric: one end carries a black/white witness stripe. Decide whether the witness stripe in the "
    "LIVE crop is at the same end as in the REFERENCE, and whether the rod looks seated on its pins. "
    "Return ONLY JSON: {\"finding\": \"consistent\"|\"mismatch\"|\"insufficient_evidence\", "
    "\"witness_mark_aligned\": bool, \"seating_flush\": bool, \"confidence\": 0..1, \"statement\": str}"
)


def _to_data_url(src: bytes | Path | str) -> str:
    raw = Path(src).read_bytes() if isinstance(src, (Path, str)) else bytes(src)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


class BasetenVisionAuditor:
    def __init__(self, *, api_key: str | None = None, timeout_s: float = 90.0, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self.api_key = api_key if api_key is not None else settings.baseten_api_key
        self.model = settings.baseten_spark_model
        self.timeout_s = timeout_s

    async def audit_checkpoint(
        self,
        *,
        current_image: bytes | Path | str,
        reference_image: bytes | Path | str | None,
        blinded_context: Mapping[str, Any],
        client: httpx.AsyncClient | None = None,
    ) -> GlimmerAuditResult:
        assert_glimmer_input_blinded(blinded_context)
        if not self.api_key:
            raise RuntimeError("BASETEN_API_KEY not configured")
        obs_id = str(blinded_context.get("observation_id", "00000000-0000-4000-8000-000000000002"))
        frame_id = str(blinded_context.get("frame_id", "frame-000120"))
        parts: list[dict[str, Any]] = [{"type": "text", "text": AUDIT_PROMPT}]
        if reference_image is not None:
            parts.append({"type": "image_url", "image_url": {"url": _to_data_url(reference_image)}})
        parts.append({"type": "image_url", "image_url": {"url": _to_data_url(current_image)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        url = f"{settings.baseten_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        if client is not None:
            resp = await client.post(url, headers=headers, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False) as c:
                resp = await c.post(url, headers=headers, json=payload)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp.raise_for_status()
        body = resp.json()
        msg = body.get("choices", [{}])[0].get("message", {})
        finish_reason = body.get("choices", [{}])[0].get("finish_reason")
        _, parsed = extract_spark_open_content_and_json(msg)
        if finish_reason == "length" and not parsed:
            parsed = {"finding": "insufficient_evidence", "statement": "model output truncated (max_tokens)"}
        if not parsed or parsed.get("finding") not in ("consistent", "mismatch", "insufficient_evidence"):
            parsed = {"finding": "insufficient_evidence", "statement": "unparseable model output"}
        finding = parsed["finding"]
        audit = {
            "observation_id": obs_id,
            "finding": finding,
            "observations": [{"frame_id": frame_id, "region_xyxy_px": [0, 0, 1, 1],
                              "statement": str(parsed.get("statement", ""))[:500] or finding}],
            "contradicted_predicate_ids": ["witness-stripe-aligned"] if finding == "mismatch" else [],
            "requires_new_view": finding == "insufficient_evidence",
        }
        validate_glimmer_output(audit)
        usage = dict(body.get("usage") or {})
        usage["request_id"] = str(body.get("id") or resp.headers.get("x-request-id") or "")
        usage["raw_verdict"] = parsed
        usage["finish_reason"] = finish_reason
        usage["max_tokens"] = self.max_tokens
        return GlimmerAuditResult(model_used=self.model, audit=audit, latency_ms=latency_ms, usage=usage)
