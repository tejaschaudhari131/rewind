from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from ..config import Settings, settings


def extract_spark_open_content_and_json(message: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """
    `meta-models/muse-spark-open-1026` on Baseten always reasons (`reasoning_content` or `<think>...</think>`).
    This helper cleanly strips `<think>...</think>` blocks and markdown code fences and extracts any embedded JSON object.
    """
    raw_content = str(message.get("content") or "").strip()
    reasoning_content = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()

    # Strip <think>...</think> blocks if present in content
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if not cleaned and reasoning_content:
        cleaned = reasoning_content

    # Try to extract JSON from ```json ... ``` fence or outermost { ... }
    parsed_json: dict[str, Any] | None = None
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate_str = fence_match.group(1) if fence_match else None
    if not candidate_str:
        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            candidate_str = brace_match.group(1)
    if candidate_str:
        try:
            obj = json.loads(candidate_str)
            if isinstance(obj, dict):
                parsed_json = obj
        except Exception:
            parsed_json = None

    return cleaned or raw_content, parsed_json


@dataclass(slots=True)
class SessionSummaryResult:
    provider: str
    model: str
    summary: str
    digest: dict[str, Any]
    latency_ms: float


class BasetenSparkClient:
    """Async post-session failure review & QA client (`meta-models/muse-spark-open-1026` on Baseten with local/contributor fallback)."""

    def __init__(
        self,
        *,
        baseten_api_key: str | None = None,
        cfg: Settings = settings,
    ) -> None:
        self.cfg = cfg
        self.baseten_api_key = (
            baseten_api_key if baseten_api_key is not None else cfg.baseten_api_key
        )

    async def summarize_session(
        self,
        *,
        session_id: str,
        cycle_events: Sequence[Mapping[str, Any]] | None = None,
        gate_events: Sequence[Mapping[str, Any]] | None = None,
        question: str = "Summarize the assembly checkpoint outcomes and root cause of any rejected steps.",
        use_remote_fallback: bool = False,
        timeout_s: float = 30.0,
    ) -> SessionSummaryResult:
        events = list(cycle_events or gate_events or [])
        accepted_count = sum(1 for e in events if e.get("decision") == "accepted")
        rejected_count = sum(1 for e in events if e.get("decision") == "rejected")
        digest: dict[str, Any] = {
            "session_id": session_id,
            "total_cycles": len(events),
            "accepted_cycles": accepted_count,
            "rejected_cycles": rejected_count,
        }

        if not self.baseten_api_key and not use_remote_fallback:
            return SessionSummaryResult(
                provider="local-deterministic-fallback",
                model=self.cfg.baseten_spark_model,
                summary=(
                    f"Session {session_id}: {len(events)} cycles recorded "
                    f"({accepted_count} accepted, {rejected_count} rejected)."
                ),
                digest=digest,
                latency_ms=0.1,
            )

        prompt = (
            f"Session {session_id}: analyze these assembly gate events and answer concisely:\n"
            f"Events: {events}\nQuestion: {question}"
        )

        if self.baseten_api_key:
            t0 = time.perf_counter()
            headers = {
                "Authorization": f"Bearer {self.baseten_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.cfg.baseten_spark_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,  # reasoning model: 2048 was observed to truncate before JSON
                "temperature": 0.1,
            }
            async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
                resp = await client.post(
                    f"{self.cfg.baseten_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            msg = data.get("choices", [{}])[0].get("message", {})
            text, parsed_json = extract_spark_open_content_and_json(msg)
            if parsed_json:
                digest["parsed_json"] = parsed_json
            digest["usage"] = data.get("usage") or {}
            digest["request_id"] = str(data.get("id") or resp.headers.get("x-request-id") or "")
            digest["measured"] = True
            return SessionSummaryResult(
                provider="baseten",
                model=self.cfg.baseten_spark_model,
                summary=text,
                digest=digest,
                latency_ms=latency_ms,
            )

        self.cfg.require_meta_api_key()
        t0 = time.perf_counter()
        payload = {
            "model": self.cfg.glimmer_fallback_model,
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": 2048,
            "input": prompt,
        }
        async with httpx.AsyncClient(proxy=self.cfg.proxy_url, timeout=timeout_s) as client:
            resp = await client.post(
                f"{self.cfg.meta_api_base}/responses",
                headers=self.cfg.auth_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        text_out = ""
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_out = str(content.get("text", ""))
        return SessionSummaryResult(
            provider="meta_contributor_fallback",
            model=self.cfg.glimmer_fallback_model,
            summary=text_out,
            digest=digest,
            latency_ms=latency_ms,
        )

    async def arbitrate_ambiguous_checkpoint(
        self,
        *,
        session_id: str,
        step_id: str,
        silhouette_iou: float,
        witness_polarity_cosine: float,
        witness_error_px: float,
        glimmer_finding: str,
        mode: str = "fixture",
    ) -> dict[str, Any]:
        """
        Dual Open-Weight Consensus Arbiter (`meta-models/muse-spark-open-1026` on Baseten).
        Explicitly triggers whenever `silhouette_iou >= 0.90` AND witness alignment fails
        (`witness_polarity_cosine < 0.50` or `witness_error_px > 6.0`) — the signature of
        the 180-degree flipped asymmetric connecting rod.
        """
        t0 = time.perf_counter()
        triggered = bool(
            silhouette_iou >= 0.90
            and (witness_polarity_cosine < 0.50 or witness_error_px > 6.0)
        )
        if not triggered:
            return {
                "triggered": False,
                "provider": "not_called",
                "model": self.cfg.baseten_spark_model,
                "verdict": "NOT_CALLED",
                "baseten_verdict": "NOT_CALLED",
                "consensus": "PASS_NO_CONFLICT",
                "latency_ms": 0.0,
                "measured": False,
            }

        if mode in ("live", "hybrid") and self.baseten_api_key:
            # Neutral question: the model is NOT told which answer to give (no leading JSON template).
            summary_res = await self.summarize_session(
                session_id=session_id,
                gate_events=[
                    {
                        "step_id": step_id,
                        "silhouette_iou": round(float(silhouette_iou), 4),
                        "witness_polarity_cosine": round(float(witness_polarity_cosine), 4),
                        "witness_error_px": round(float(witness_error_px), 2),
                        "visual_auditor_finding": glimmer_finding,
                    }
                ],
                question=(
                    "An asymmetric connecting rod was checked. Given these measurements (silhouette IoU vs reference, "
                    "cosine of the witness-stripe polarity vector vs reference where +1 = same orientation and -1 = "
                    "reversed, witness position error in px) and the independent visual auditor finding, decide whether "
                    "the rod is installed in the reference orientation. Return ONLY JSON "
                    '{"verdict": "CONSISTENT"|"MISMATCH"|"INSUFFICIENT_EVIDENCE", "failure_signature": str, "reason": str}.'
                ),
                use_remote_fallback=(mode == "live"),
            )
            parsed = (summary_res.digest or {}).get("parsed_json") or {}
            measured = bool((summary_res.digest or {}).get("measured"))
            provider = "BASETEN_OPEN_WEIGHT" if summary_res.provider == "baseten" else summary_res.provider
        else:
            parsed = {"verdict": "MISMATCH", "failure_signature": "ASYMMETRIC_ROD_INVERTED_180_DEG"}
            summary_res = SessionSummaryResult(
                provider="local-deterministic-fallback", model=self.cfg.baseten_spark_model,
                summary="fixture mode: deterministic rule (no model call)", digest={}, latency_ms=0.0,
            )
            measured = False
            provider = "LOCAL_DETERMINISTIC_FIXTURE"
        verdict = str(parsed.get("verdict", "")).upper()
        if verdict not in ("CONSISTENT", "MISMATCH", "INSUFFICIENT_EVIDENCE"):
            verdict = "INSUFFICIENT_EVIDENCE"  # fail closed on unparseable output
        auditor_mismatch = "mismatch" in str(glimmer_finding).lower()
        if verdict == "MISMATCH" and auditor_mismatch:
            consensus = "DUAL_OPEN_WEIGHT_REJECT_180_FLIP" if measured or provider == "LOCAL_DETERMINISTIC_FIXTURE" else "REJECT"
        elif verdict == "MISMATCH" or auditor_mismatch:
            consensus = "REJECT_PARTIAL_AGREEMENT"
        else:
            consensus = "NO_CONSENSUS_FAIL_CLOSED"
        digest = summary_res.digest or {}
        return {
            "triggered": True,
            "provider": provider,
            "model": self.cfg.baseten_spark_model,
            "verdict": verdict,
            "baseten_verdict": verdict,
            "failure_signature": str(parsed.get("failure_signature", "")),
            "visual_auditor_finding": glimmer_finding,
            "consensus": consensus,
            "explanation": str(parsed.get("reason", ""))[:500] or (
                f"IoU={silhouette_iou:.3f}, witness cos={witness_polarity_cosine:+.3f}, err={witness_error_px:.1f}px."
            ),
            "latency_ms": float(summary_res.latency_ms) if measured else 0.0,
            "measured": measured,
            "usage": digest.get("usage") or {},
            "request_id": digest.get("request_id", ""),
        }
