from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from ..config import Settings, settings


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
        timeout_s: float = 25.0,
    ) -> SessionSummaryResult:
        events = list(cycle_events or gate_events or [])
        accepted_count = sum(1 for e in events if e.get("decision") == "accepted")
        rejected_count = sum(1 for e in events if e.get("decision") == "rejected")
        digest = {
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
                "max_tokens": 512,
                "temperature": 0.1,
            }
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    f"{self.cfg.baseten_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            text = data["choices"][0]["message"]["content"]
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
            "max_output_tokens": 1024,
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
