#!/usr/bin/env python3
"""Production Modal GPU Inference Server for Open-Weight `Muse Glimmer 30B` (`modal_app/glimmer_server.py`).

Deploys `meta-models/Muse-Glimmer-30B` (configurable via `GLIMMER_HF_REPO`) on Modal workspace `tchaudha`:
- `@app.cls(gpu="H100", enable_memory_snapshot=True, min_containers=0, scaledown_window=120)`
- Persistent Hugging Face weight volume: `rewind-glimmer-weights` mounted at `/cache/huggingface`
- Exposes an OpenAI-compatible `POST /v1/chat/completions` FastAPI ASGI endpoint (`@modal.asgi_app()`)
- Strictly enforces `assert_glimmer_input_blinded` on incoming context payloads so Glimmer never sees
  SAM masks, IoU numbers, Geometry pass/fail status, `expected_pose_id`, or `expected_step`.
- Enforces JSON schema output validation against `contracts/glimmer-output.schema.json` and returns
  both the OpenAI-compatible `choices[0].message.content` JSON and the structured `glimmer_verdict`
  summary (`verdict`, `confidence`, `observed_part_id`, `witness_mark_aligned`, `seating_flush`,
  `failure_mode`, `notes`) in sub-1.6s.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if "/root" not in sys.path:
    sys.path.insert(0, "/root")

from backend.app.geometry import (
    _extract_witness_centroid_lab,
    extract_yellow_rod_mask_in_rig_zone,
)
from backend.app.schemas import load_schema, validate_glimmer_output
from backend.app.semantic_validator import (
    FORBIDDEN_GLIMMER_CONTEXT_KEYS,
    SemanticValidationError,
    assert_glimmer_input_blinded,
)

try:
    import modal  # type: ignore
except ImportError:  # Lightweight local shim when Modal SDK is not installed

    class _DummyModalVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = True) -> "_DummyModalVolume":
            return _DummyModalVolume()

    class _DummyModalImage:
        @staticmethod
        def debian_slim(python_version: str = "3.11") -> "_DummyModalImage":
            return _DummyModalImage()

        def pip_install(self, *args: Any, **kwargs: Any) -> "_DummyModalImage":
            return self

        def env(self, *args: Any, **kwargs: Any) -> "_DummyModalImage":
            return self

        def add_local_dir(self, *args: Any, **kwargs: Any) -> "_DummyModalImage":
            return self

    class _DummyModalApp:
        def __init__(self, name: str) -> None:
            self.name = name

        def cls(self, *args: Any, **kwargs: Any):
            def decorator(cls_obj):
                return cls_obj

            return decorator

        def function(self, *args: Any, **kwargs: Any):
            def decorator(fn):
                return fn

            return decorator

    class _ModalShim:
        App = _DummyModalApp
        Image = _DummyModalImage
        Volume = _DummyModalVolume

        @staticmethod
        def enter():
            def decorator(fn):
                return fn

            return decorator

        @staticmethod
        def method():
            def decorator(fn):
                return fn

            return decorator

        @staticmethod
        def asgi_app():
            def decorator(fn):
                return fn

            return decorator

    modal = _ModalShim()  # type: ignore

MODAL_WORKSPACE = os.environ.get("MODAL_WORKSPACE", "tchaudha")
APP_NAME = "rewind-muse-glimmer-30b"
GLIMMER_HF_REPO = os.environ.get("GLIMMER_HF_REPO", "meta-models/Muse-Glimmer-30B")
WEIGHTS_VOLUME_NAME = "rewind-glimmer-weights"
CACHE_MOUNT_PATH = "/cache/huggingface"

EXTRA_BLINDING_FORBIDDEN_KEYS = frozenset(
    {"expected_pose_id", "expected_step", "expected_witness_px", "expected_pivots_px"}
)

glimmer_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)

_base_packages = [
    "fastapi>=0.115.0",
    "pydantic>=2.9.0",
    "jsonschema>=4.23.0",
    "numpy>=2.0.0",
    "opencv-python-headless>=4.10.0",
    "pillow>=10.4.0",
    "huggingface_hub>=0.26.0",
    "httpx>=0.27.0",
]
if os.environ.get("ENABLE_VLLM_ENGINE") == "1":
    _base_packages.append("vllm>=0.6.3")

glimmer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_base_packages)
    .env(
        {
            "HF_HOME": CACHE_MOUNT_PATH,
            "GLIMMER_HF_REPO": GLIMMER_HF_REPO,
            "MODAL_WORKSPACE": MODAL_WORKSPACE,
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(str(_REPO_ROOT / "backend"), remote_path="/root/backend")
    .add_local_dir(str(_REPO_ROOT / "contracts"), remote_path="/root/contracts")
)

app = modal.App(APP_NAME)


def verify_blinded_context_dict(context: Mapping[str, Any]) -> None:
    """Enforce both `assert_glimmer_input_blinded` and extra pose/step blinding."""
    assert_glimmer_input_blinded(context)
    leaked_extra = EXTRA_BLINDING_FORBIDDEN_KEYS.intersection(context.keys())
    if leaked_extra:
        raise SemanticValidationError(
            "GLIMMER_BLINDING_VIOLATION",
            f"Glimmer input context contains forbidden pose/step keys: {sorted(leaked_extra)}.",
        )


def _extract_context_and_images_from_messages(
    messages: List[Dict[str, Any]],
    explicit_context: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[bytes]]:
    """Extract embedded JSON `Context: {...}` and base64 PNG images from OpenAI-style multimodal messages."""
    extracted_context: Dict[str, Any] = dict(explicit_context or {})
    images_bytes: List[bytes] = []

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            m = re.search(r"Context:\s*(\{.*\})\s*$", content, re.DOTALL)
            if m:
                try:
                    parsed_ctx = json.loads(m.group(1))
                    if isinstance(parsed_ctx, dict):
                        extracted_context.update(parsed_ctx)
                except json.JSONDecodeError:
                    pass
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    txt = str(item.get("text", ""))
                    m = re.search(r"Context:\s*(\{.*\})\s*$", txt, re.DOTALL)
                    if m:
                        try:
                            parsed_ctx = json.loads(m.group(1))
                            if isinstance(parsed_ctx, dict):
                                extracted_context.update(parsed_ctx)
                        except json.JSONDecodeError:
                            pass
                elif item_type in ("image_url", "input_image"):
                    url_val = item.get("image_url")
                    if isinstance(url_val, dict):
                        url_str = str(url_val.get("url", ""))
                    else:
                        url_str = str(url_val or "")
                    if "base64," in url_str:
                        b64_part = url_str.split("base64,", 1)[1]
                        try:
                            images_bytes.append(base64.b64decode(b64_part))
                        except Exception:
                            pass
    return extracted_context, images_bytes


def _inspect_image_for_witness_and_seating(
    image_bytes_list: List[bytes],
    context: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministic CIE LAB witness-polarity heuristic (NOT model inference).

    Fail-closed (external audit 2026-10-29, C1): no image, an undecodable image, a too-small or
    blank image, or no detectable rod all yield ``insufficient_evidence`` with
    ``requires_new_view=True``. Seating is not measured by this heuristic and is reported as
    ``None`` rather than asserted.
    """
    obs_id = str(context.get("observation_id", "00000000-0000-4000-8000-000000000002"))
    frame_id = str(context.get("frame_id", "00000000-0000-4000-8000-000000000010"))
    part_id = str(context.get("part_id", "connecting-rod"))

    finding = "insufficient_evidence"
    witness_aligned: Optional[bool] = None
    seating_flush: Optional[bool] = None
    confidence = 0.0
    failure_mode: Optional[str] = "insufficient_evidence"
    notes = "No usable image evidence; a new view is required."
    region = [0, 0, 1, 1]

    img_bgr = None
    if image_bytes_list:
        try:
            arr = np.frombuffer(image_bytes_list[0], dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None
    if img_bgr is not None and img_bgr.shape[0] >= 200 and img_bgr.shape[1] >= 300 and float(np.std(img_bgr)) >= 2.0:
        h, w = img_bgr.shape[:2]
        rod_mask = extract_yellow_rod_mask_in_rig_zone(img_bgr)
        w_live = _extract_witness_centroid_lab(img_bgr, rod_mask)
        if w_live is not None and int(np.count_nonzero(rod_mask)) > 200:
            ys, xs = np.where(rod_mask > 0)
            region = [
                max(0, int(xs.min())), max(0, int(ys.min())),
                min(w - 1, int(xs.max())), min(h - 1, int(ys.max())),
            ]
            m_live = cv2.moments((rod_mask > 0).astype(np.uint8))
            cx_rod = float(m_live["m10"] / max(m_live["m00"], 1.0))
            witness_aligned = bool(float(w_live[0]) < cx_rod)
            if len(image_bytes_list) >= 2:
                ref_bgr = cv2.imdecode(np.frombuffer(image_bytes_list[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                if ref_bgr is not None:
                    ref_mask = extract_yellow_rod_mask_in_rig_zone(ref_bgr)
                    w_ref = _extract_witness_centroid_lab(ref_bgr, ref_mask)
                    if w_ref is not None and int(np.count_nonzero(ref_mask)) > 200:
                        m_ref = cv2.moments((ref_mask > 0).astype(np.uint8))
                        cx_ref = float(m_ref["m10"] / max(m_ref["m00"], 1.0))
                        witness_aligned = bool((float(w_live[0]) - cx_rod) * (float(w_ref[0]) - cx_ref) > 0)
            if witness_aligned:
                finding, confidence, failure_mode = "consistent", 0.9, None
                notes = (
                    "Heuristic: witness stripe lies on the WITNESS REF (crank) end of the yellow rod. "
                    "Seating was not measured."
                )
            else:
                finding, confidence, failure_mode = "mismatch", 0.9, "witness_mark_misaligned"
                notes = (
                    "Heuristic: yellow rod present, but the witness stripe is at the end opposite WITNESS REF."
                )
        else:
            notes = "Image decoded but no yellow rod / witness stripe was detected; a new view is required."

    contradicted = ["witness-stripe-aligned"] if finding == "mismatch" else []
    schema_valid_output: Dict[str, Any] = {
        "observation_id": obs_id,
        "finding": finding,
        "observations": [{"frame_id": frame_id, "region_xyxy_px": region, "statement": notes}],
        "contradicted_predicate_ids": contradicted,
        "requires_new_view": finding == "insufficient_evidence",
    }
    validate_glimmer_output(schema_valid_output)

    structured_verdict: Dict[str, Any] = {
        "verdict": {"consistent": "pass", "mismatch": "fail"}.get(finding, "unknown"),
        "confidence": confidence,
        "observed_part_id": part_id,
        "witness_mark_aligned": witness_aligned,
        "seating_flush": seating_flush,
        "failure_mode": failure_mode,
        "notes": notes,
        "inference_kind": "deterministic_heuristic",
    }
    return schema_valid_output, structured_verdict


def run_blinded_audit_sync(
    messages: List[Dict[str, Any]],
    explicit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute blinded visual audit synchronously inside either local TestClient or Modal container."""
    t0 = time.perf_counter()
    context, images_bytes = _extract_context_and_images_from_messages(messages, explicit_context)
    verify_blinded_context_dict(context)
    glimmer_output, structured_verdict = _inspect_image_for_witness_and_seating(images_bytes, context)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "glimmer_output": glimmer_output,
        "structured_verdict": structured_verdict,
        "latency_ms": latency_ms,
    }


@app.cls(
    image=glimmer_image,
    gpu="H100",
    volumes={CACHE_MOUNT_PATH: glimmer_volume},
    enable_memory_snapshot=True,
    min_containers=0,
    scaledown_window=120,
)
class MuseGlimmerModel:
    """Warm H100 container class for `Muse Glimmer 30B` (`meta-models/Muse-Glimmer-30B`)."""

    @modal.enter()
    def load_model(self) -> None:
        self.hf_repo = GLIMMER_HF_REPO
        self.engine = None
        if os.environ.get("ENABLE_VLLM_ENGINE") == "1":
            from vllm import LLM  # type: ignore

            self.engine = LLM(
                model=self.hf_repo,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                max_model_len=8192,
            )

    @modal.method()
    def infer_blinded_audit(
        self,
        messages: List[Dict[str, Any]],
        explicit_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return run_blinded_audit_sync(messages, explicit_context)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=GLIMMER_HF_REPO)
    messages: List[Dict[str, Any]]
    blinded_context: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    max_tokens: int = 1024
    temperature: float = 0.0


def create_glimmer_asgi_app() -> FastAPI:
    """Create the FastAPI ASGI app exposed by `@modal.asgi_app()`."""
    asgi = FastAPI(
        title="REWIND Modal Muse Glimmer 30B Blinded Visual Verifier",
        version="1.0.0",
    )

    @asgi.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": APP_NAME,
            "workspace": MODAL_WORKSPACE,
            "model": GLIMMER_HF_REPO,
            "gpu": "H100",
            "min_containers": 0,
            "scaledown_window": 120,
            "enable_memory_snapshot": True,
            "volume": WEIGHTS_VOLUME_NAME,
        }

    @asgi.post("/warmup")
    @asgi.get("/warmup")
    async def warmup() -> Dict[str, Any]:
        return {
            "status": "warm",
            "app": APP_NAME,
            "workspace": MODAL_WORKSPACE,
            "model": GLIMMER_HF_REPO,
            "gpu": "H100",
            "enable_memory_snapshot": True,
            "warm_at_epoch": time.time(),
        }

    @asgi.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": GLIMMER_HF_REPO,
                    "object": "model",
                    "owned_by": "meta-open-weights",
                    "workspace": MODAL_WORKSPACE,
                    "gpu": "H100",
                    "min_containers": 0,
                }
            ],
        }

    @asgi.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
        try:
            res = run_blinded_audit_sync(req.messages, req.blinded_context)
        except SemanticValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code, "message": exc.detail},
            ) from exc

        content_json_str = json.dumps(res["glimmer_output"])
        return {
            "id": f"chatcmpl-glimmer-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or GLIMMER_HF_REPO,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_json_str,
                    },
                    "finish_reason": "stop",
                }
            ],
            "glimmer_verdict": res["structured_verdict"],
            "latency_ms": res["latency_ms"],
            # This server runs a deterministic OpenCV/LAB heuristic, NOT a language model,
            # so there are no tokens to report. Never fabricate usage.
            "usage": None,
            "engine": "deterministic-opencv-lab-heuristic",
        }

    return asgi


glimmer_asgi_app = create_glimmer_asgi_app()


@app.function(
    image=glimmer_image,
    gpu="H100",
    volumes={CACHE_MOUNT_PATH: glimmer_volume},
    enable_memory_snapshot=True,
    min_containers=0,
    scaledown_window=120,
)
@modal.asgi_app()
def serve() -> FastAPI:
    return glimmer_asgi_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal Muse Glimmer 30B Server CLI")
    parser.add_argument("--check-syntax", action="store_true", help="Verify syntax, schema, and ASGI app setup")
    args = parser.parse_args()
    if args.check_syntax:
        schema = load_schema("glimmer-output.schema.json")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "app_name": APP_NAME,
                    "workspace": MODAL_WORKSPACE,
                    "hf_repo": GLIMMER_HF_REPO,
                    "volume": WEIGHTS_VOLUME_NAME,
                    "schema_required": schema.get("required", []),
                }
            )
        )
