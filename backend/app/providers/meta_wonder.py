from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

import httpx

from ..config import build_httpx_client, settings
from ..schemas import load_schema, validate_wonder_output


@dataclass
class WonderCompilerResult:
    model_used: str
    provider_response_id: str
    proposal: dict[str, Any]
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)

    def to_planner_proposal_payload(self) -> dict[str, Any]:
        """Convert to contract.schema.json PlannerProposal payload."""
        return {
            "model_id": "muse-wonder-1.0",
            "provider_response_id": self.provider_response_id,
            "model_output": self.proposal,
        }


@dataclass
class WonderCuaStepResult:
    model_used: str
    provider_response_id: str
    computer_calls: List[dict[str, Any]]
    text_summary: str
    latency_ms: float
    usage: dict = field(default_factory=dict)


class MetaWonderClient:
    """
    Reverse reassembly state compiler & CUA inspector client.

    Defaults to `muse-spark-1.3-contributor` ($0.10/1M in, $0.20/1M out, covered by $150 free credits)
    while supporting explicit switching to `muse-wonder-1.0` via `REWIND_COMPILER_MODEL=muse-wonder-1.0`
    or `model_override="muse-wonder-1.0"`.
    """

    def __init__(
        self,
        *,
        model_override: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model_override or settings.compiler_model
        self._external_client = client

    async def compile_planner_proposal(
        self,
        *,
        observation_id: str,
        step_id: str,
        part_id: str | None,
        evidence_ids: list[str],
        required_predicate_ids: list[str],
        geometry_summary: Mapping[str, Any],
        image: bytes | Path | str | None = None,
        measurement_record_id: str | None = None,
        reasoning_effort: str = "minimal",
        max_output_tokens: int = 2048,
    ) -> WonderCompilerResult:
        wonder_schema = load_schema("wonder-output.schema.json")
        clean_schema = {k: v for k, v in wonder_schema.items() if not k.startswith("$")}

        content_items: list[dict[str, Any]] = []
        if image is not None:
            raw_bytes = Path(image).read_bytes() if isinstance(image, (Path, str)) else bytes(image)
            b64_png = base64.b64encode(raw_bytes).decode("ascii")
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_png}",
                }
            )

        prompt = (
            "You are the REWIND reassembly planner. Return ONLY a JSON object matching the schema.\n"
            f"- Set observation_id to '{observation_id}'.\n"
            f"- Current step_id: '{step_id}', target part_id: {json.dumps(part_id)}.\n"
            f"- Available evidence_ids: {json.dumps(evidence_ids)}.\n"
            f"- Required predicate_ids: {json.dumps(required_predicate_ids)}.\n"
            f"- Available measurement_record_id: {json.dumps(measurement_record_id)}.\n"
            f"- Geometry check summary: {json.dumps(dict(geometry_summary))}\n"
            "If geometry indicates a missing or damaged connecting_rod and a measurement_record_id is provided, "
            "propose action='generate_supported_part' with that measurement_record_id. Otherwise propose "
            "action='instruct_move' or 'request_capture' with a clear operator instruction."
        )
        content_items.append({"type": "input_text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "user", "content": content_items}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "wonder_output",
                    "schema": clean_schema,
                    "strict": True,
                }
            },
        }

        t0 = time.perf_counter()
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
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()

        out_text = ""
        for item in body.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out_text += c.get("text", "")

        parsed = json.loads(out_text)

        # Deterministically pin caller's exact IDs so LLM UUID hyphen typos never break CAS
        parsed["observation_id"] = observation_id
        if part_id is not None:
            parsed["part_id"] = part_id
        if evidence_ids:
            parsed["evidence_ids"] = list(evidence_ids)
        if required_predicate_ids:
            parsed["required_predicate_ids"] = list(required_predicate_ids)
        schema_props = wonder_schema.get("properties", {})
        if "step_id" in schema_props:
            parsed["step_id"] = step_id
        if "frame_id" in schema_props and evidence_ids:
            parsed["frame_id"] = evidence_ids[0]
        if parsed.get("action") == "generate_supported_part" and measurement_record_id is not None:
            parsed["measurement_record_id"] = measurement_record_id

        validate_wonder_output(parsed)
        return WonderCompilerResult(
            model_used=str(body.get("model", self.model)),
            provider_response_id=str(body.get("id", "resp_wonder_unknown")),
            proposal=parsed,
            latency_ms=round(latency_ms, 2),
            usage=body.get("usage", {}),
        )

    async def run_cua_step(
        self,
        *,
        screenshot_source: bytes | Path | str,
        task_prompt: str,
        display_width: int = 1200,
        display_height: int = 680,
        raw_instruction: bool = False,
        previous_response_id: str | None = None,
        call_id: str | None = None,
    ) -> WonderCuaStepResult:
        raw_bytes = (
            Path(screenshot_source).read_bytes()
            if isinstance(screenshot_source, (Path, str))
            else bytes(screenshot_source)
        )
        payload = self.build_cua_inspection_payload(
            cad_preview_png=raw_bytes,
            display_width=display_width,
            display_height=display_height,
            expected_part_description=task_prompt,
            model=self.model,
        )
        if raw_instruction:
            # Generic computer-use task (e.g. manual check-off): send the task verbatim, not the CAD template.
            payload["input"][0]["content"][1]["text"] = task_prompt
        if previous_response_id and call_id:
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            payload["previous_response_id"] = previous_response_id
            payload["input"] = [{"type": "computer_call_output", "call_id": call_id,
                                 "output": {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}}]
        t0 = time.perf_counter()
        async with build_httpx_client() as client:
            resp = await client.post(
                f"{settings.meta_api_base}/responses",
                headers=settings.auth_headers(),
                json=payload,
            )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp.raise_for_status()
        body = resp.json()
        calls: List[dict[str, Any]] = []
        texts: List[str] = []
        for item in body.get("output", []):
            if item.get("type") == "computer_call":
                calls.append(item)
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
        return WonderCuaStepResult(
            model_used=str(body.get("model", self.model)),
            provider_response_id=str(body.get("id", "resp_cua_unknown")),
            computer_calls=calls,
            text_summary=" ".join(texts).strip(),
            latency_ms=latency_ms,
            usage=dict(body.get("usage") or {}),
        )

    @staticmethod
    def build_cua_inspection_payload(
        *,
        cad_preview_png: bytes,
        display_width: int = 1280,
        display_height: int = 800,
        expected_part_description: str = "flat two-hole connecting rod link",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Construct the Meta Responses API Computer Use (`type: computer`) inspection payload."""
        b64 = base64.b64encode(cad_preview_png).decode("ascii")
        return {
            "model": model or settings.model_wonder_full,
            "tools": [
                {
                    "type": "computer",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": "linux",
                }
            ],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                        },
                        {
                            "type": "input_text",
                            "text": (
                                f"Inspect the generated 3D CAD preview and orthographic drawing for: "
                                f"{expected_part_description}. Verify two circular pivot through-holes and rounded ends."
                            ),
                        },
                    ],
                }
            ],
        }
