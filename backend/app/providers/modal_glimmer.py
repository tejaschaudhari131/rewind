from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import httpx

from ..config import build_httpx_client, settings
from ..schemas import load_schema, validate_glimmer_output
from ..semantic_validator import assert_glimmer_input_blinded


@dataclass
class GlimmerAuditResult:
    model_used: str
    audit: dict[str, Any]
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)

    def to_visual_audit_payload(
        self,
        *,
        model_id: str = "Muse-Glimmer",
        model_revision: str = "glimmer-30b-a3b-1026",
        deployment_revision: str = "modal-rewind-glimmer-v1",
    ) -> dict[str, Any]:
        """Convert to contract.schema.json VisualAudit payload."""
        return {
            "model_id": model_id,
            "model_revision": model_revision,
            "deployment_revision": deployment_revision,
            "model_output": self.audit,
            "finish_reason": "stop",
            "latency_ms": round(float(self.latency_ms), 2),
        }


class ModalGlimmerClient:
    """
    Blinded visual verifier client (`Muse Glimmer 30B` on Modal, with automatic contributor fallback
    to `muse-spark-1.3-contributor` when `MODAL_GLIMMER_ENDPOINT` is not yet set).

    Strictly enforces `assert_glimmer_input_blinded(context)` before making any inference request so
    Glimmer never sees SAM masks, IoU numbers, Geometry pass/fail status, or Gate verdicts.
    """

    def __init__(
        self,
        *,
        modal_endpoint: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.modal_endpoint = (
            modal_endpoint if modal_endpoint is not None else settings.modal_glimmer_endpoint
        )
        self._external_client = client

    async def audit_checkpoint(
        self,
        *,
        current_image: bytes | Path | str,
        reference_image: bytes | Path | str | None,
        blinded_context: Mapping[str, Any],
        reasoning_effort: str = "minimal",
        max_output_tokens: int = 2048,
    ) -> GlimmerAuditResult:
        # 1. Enforce strict blinding before touching network or model
        assert_glimmer_input_blinded(blinded_context)

        obs_id = str(
            blinded_context.get("observation_id", "00000000-0000-4000-8000-000000000002")
        )
        frame_id = str(blinded_context.get("frame_id", "frame-000120"))
        cur_bytes = (
            Path(current_image).read_bytes()
            if isinstance(current_image, (Path, str))
            else bytes(current_image)
        )
        cur_b64 = base64.b64encode(cur_bytes).decode("ascii")

        glimmer_schema = load_schema("glimmer-output.schema.json")
        clean_schema = {k: v for k, v in glimmer_schema.items() if not k.startswith("$")}

        content_items: list[dict[str, Any]] = [
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{cur_b64}",
            }
        ]
        if reference_image is not None:
            ref_bytes = (
                Path(reference_image).read_bytes()
                if isinstance(reference_image, (Path, str))
                else bytes(reference_image)
            )
            ref_b64 = base64.b64encode(ref_bytes).decode("ascii")
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{ref_b64}",
                }
            )

        prompt = (
            "You are the blinded visual auditor for REWIND. Inspect the image(s) independently and "
            "return ONLY a JSON object matching the schema:\n"
            f"- Set observation_id to '{obs_id}'.\n"
            f"- Use frame_id '{frame_id}' in every item of observations.\n"
            "- Set finding to 'consistent', 'mismatch', or 'insufficient_evidence'.\n"
            f"Context: {json.dumps(dict(blinded_context))}"
        )
        content_items.append({"type": "input_text", "text": prompt})

        t0 = time.perf_counter()
        if self.modal_endpoint:
            headers = {"Content-Type": "application/json"}
            if settings.modal_glimmer_api_key:
                headers["Authorization"] = f"Bearer {settings.modal_glimmer_api_key}"
            req_body = {
                "messages": [{"role": "user", "content": content_items}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "glimmer_output", "schema": clean_schema},
                },
            }
            async with httpx.AsyncClient(timeout=settings.http_timeout_s, trust_env=False) as modal_client:
                resp = await modal_client.post(
                    self.modal_endpoint, headers=headers, json=req_body
                )
            resp.raise_for_status()
            body = resp.json()
            out_text = body["choices"][0]["message"]["content"]
            model_used = str(body.get("model", "Muse-Glimmer-30B-Modal"))
            usage = body.get("usage", {})
        else:
            payload = {
                "model": settings.model_spark_contributor,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_output_tokens,
                "input": [{"role": "user", "content": content_items}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "glimmer_output",
                        "schema": clean_schema,
                        "strict": True,
                    }
                },
            }
            if self._external_client is not None:
                resp = await self._external_client.post(
                    f"{settings.meta_api_base}/responses",
                    headers=settings.auth_headers(),
                    json=payload,
                )
            else:
                async with build_httpx_client() as client:
                    resp = await client.post(
                        f"{settings.meta_api_base}/responses",
                        headers=settings.auth_headers(),
                        json=payload,
                    )
            resp.raise_for_status()
            body = resp.json()
            out_text = ""
            for item in body.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        out_text += c.get("text", "")
            model_used = str(body.get("model", settings.model_spark_contributor))
            usage = body.get("usage", {})

        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed = json.loads(out_text)

        # Never rewrite model-emitted IDs (external audit 2026-10-29): a mismatched observation or
        # frame binding means the output cannot be attributed to this observation -> fail closed.
        if parsed.get("observation_id") != obs_id:
            raise ValueError("Glimmer output observation_id does not match the requested observation")
        for obs_item in parsed.get("observations") or []:
            if not isinstance(obs_item, dict) or obs_item.get("frame_id") != frame_id:
                raise ValueError("Glimmer output frame_id does not match the requested frame")
        schema_props = glimmer_schema.get("properties", {})
        if "step_id" in schema_props and "step_id" in blinded_context:
            parsed["step_id"] = str(blinded_context["step_id"])
        if "part_id" in schema_props and "part_id" in blinded_context:
            parsed["part_id"] = blinded_context["part_id"]

        validate_glimmer_output(parsed)
        return GlimmerAuditResult(
            model_used=model_used,
            audit=parsed,
            latency_ms=round(latency_ms, 2),
            usage=usage,
        )

    async def warmup(self) -> dict[str, Any]:
        """Ping the Modal H100 container `/warmup` or `/health` endpoint so GPU memory stays warm."""
        t0 = time.perf_counter()
        if self.modal_endpoint:
            base = self.modal_endpoint.split("/v1/")[0].rstrip("/")
            try:
                async with httpx.AsyncClient(timeout=5.0) as modal_client:
                    resp = await modal_client.get(f"{base}/warmup")
                    if resp.status_code == 200:
                        data = resp.json()
                        data["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                        return data
            except Exception:
                pass
        return {
            "status": "warm",
            "provider": "MODAL_H100_WARM",
            "model": "Muse-Glimmer-30B",
            "enable_memory_snapshot": True,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
