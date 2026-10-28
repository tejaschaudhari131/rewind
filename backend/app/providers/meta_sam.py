from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    video_segmentation_format,
)
from PIL import Image

from ..config import build_httpx_client, settings
from ..schemas import make_artifact, sha256_hex


@dataclass
class SamDetectedInstance:
    frame_index: int
    sam_track_id: str
    part_id: str
    prompt_text: str
    box_xyxy_px: tuple[int, int, int, int]
    mask_token: str
    crop_mask: np.ndarray
    full_frame_mask: np.ndarray
    confidence: float
    area_px: int
    centroid_xy_px: tuple[float, float]
    principal_angle_deg: float

    def to_contract_object(
        self,
        *,
        provider_response_id: str,
        frame_id: str = "00000000-0000-4000-8000-000000000010",
        mask_artifact: dict[str, str] | None = None,
        crop_artifact: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        png_bytes = encode_mask_png_bytes(self.full_frame_mask)
        crop_bytes = encode_mask_png_bytes(self.crop_mask)
        m_art = mask_artifact or make_artifact(
            uri=f"evidence://masks/{provider_response_id}/track-{self.sam_track_id}.png",
            mime_type="image/png",
            data=png_bytes,
        )
        c_art = crop_artifact or make_artifact(
            uri=f"evidence://crops/{provider_response_id}/track-{self.sam_track_id}.png",
            mime_type="image/png",
            data=crop_bytes,
        )
        try:
            track_int = int(str(self.sam_track_id).split("_")[-1])
        except ValueError:
            track_int = 0
        return {
            "frame_id": frame_id,
            "sam_response_id": provider_response_id,
            "sam_track_id": track_int,
            "part_id": self.part_id.replace("_", "-"),
            "identity_status": "confirmed" if self.confidence >= 0.65 else "provisional",
            "bbox_xyxy_inclusive_px": [int(v) for v in self.box_xyxy_px],
            "mask": m_art,
            "crop": c_art,
            "mask_encoding": "full_frame_binary_png_0_255",
            "provider_mask_confidence": round(float(self.confidence), 4),
        }


@dataclass
class SamSegmentationResult:
    provider_response_id: str
    provider_request_id: str
    image_width_px: int
    image_height_px: int
    raw_output_text: str
    latency_ms: float
    instances: list[SamDetectedInstance] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_segmentation_payload(
        self,
        *,
        frame_id: str = "00000000-0000-4000-8000-000000000010",
        source_uri: str = "evidence://observations/obs-current.png",
        source_sha256: str | None = None,
        timestamp_ms: float = 0.0,
    ) -> dict[str, Any]:
        media_artifact = make_artifact(
            uri=source_uri,
            mime_type="image/png",
            sha256=source_sha256 or ("a" * 64),
        )
        return {
            "provider_request_id": self.provider_request_id,
            "provider_response_id": self.provider_response_id,
            "source_media": media_artifact,
            "frames": [
                {
                    "frame_id": frame_id,
                    "image": media_artifact,
                    "frame_index": 0,
                    "timestamp_ms": float(timestamp_ms),
                    "width_px": int(self.image_width_px),
                    "height_px": int(self.image_height_px),
                    "coordinate_space": "undistorted_source_image_px",
                }
            ],
            "objects": [
                inst.to_contract_object(
                    provider_response_id=self.provider_response_id,
                    frame_id=frame_id,
                )
                for inst in self.instances
            ],
            "billed_video_frames": None,
            "response_status": "completed",
        }


def compute_mask_moments(binary_mask: np.ndarray) -> tuple[int, tuple[float, float], float]:
    """Compute pixel area, (cx, cy) centroid, and principal axis orientation in degrees [-90, 90]."""
    ys, xs = np.nonzero(binary_mask > 0)
    area = int(len(xs))
    if area == 0:
        return 0, (0.0, 0.0), 0.0
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    if area < 3:
        return area, (cx, cy), 0.0
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    mu20 = float(np.mean(dx * dx))
    mu02 = float(np.mean(dy * dy))
    mu11 = float(np.mean(dx * dy))
    angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return area, (cx, cy), float(np.degrees(angle_rad))


def decode_sam_token_string(
    raw_text: str,
    *,
    image_width_px: int,
    image_height_px: int,
    part_id: str = "unknown_part",
    prompt_text: str = "",
) -> list[SamDetectedInstance]:
    """Decode `<0f>0<|box;x1=...|><|mask;x=0;y=0;c=...;data=H,W,~...|>` token stream into full-frame binary masks."""
    instances: list[SamDetectedInstance] = []
    if not raw_text or not raw_text.strip():
        return instances
    parser = video_segmentation_format().create_parser()
    parser.push(raw_text if raw_text.endswith("\n") else raw_text + "\n")
    finish = parser.finish(CompletedOutcome())
    for rec in finish.result.records:
        if not isinstance(rec, SegmentationMaskRecord):
            continue
        raster_bytes = decode_mask_to_raster(rec.mask)
        crop_arr = np.frombuffer(raster_bytes, dtype=np.uint8).reshape(
            (rec.mask.height, rec.mask.width)
        )
        crop_u8 = ((crop_arr > 0).astype(np.uint8)) * 255
        x1 = int(rec.bounds.left)
        y1 = int(rec.bounds.top)
        x2_excl = int(rec.bounds.right)
        y2_excl = int(rec.bounds.bottom)
        x2_incl = max(x1, x2_excl - 1)
        y2_incl = max(y1, y2_excl - 1)

        full_mask = np.zeros((image_height_px, image_width_px), dtype=np.uint8)
        cy1 = max(0, min(image_height_px, y1))
        cy2 = max(0, min(image_height_px, y1 + rec.mask.height))
        cx1 = max(0, min(image_width_px, x1))
        cx2 = max(0, min(image_width_px, x1 + rec.mask.width))
        full_mask[cy1:cy2, cx1:cx2] = crop_u8[: (cy2 - cy1), : (cx2 - cx1)]

        area, centroid, angle_deg = compute_mask_moments(full_mask)
        conf = float(rec.confidence) if rec.confidence is not None else 1.0
        frame_idx = int(rec.frame.frame_index) if rec.frame is not None else 0
        mask_token_str = f"x=0;y=0;data={rec.mask.height},{rec.mask.width},{rec.mask.payload}"
        instances.append(
            SamDetectedInstance(
                frame_index=frame_idx,
                sam_track_id=str(rec.object_id),
                part_id=part_id,
                prompt_text=prompt_text or part_id,
                box_xyxy_px=(x1, y1, x2_incl, y2_incl),
                mask_token=mask_token_str,
                crop_mask=crop_u8,
                full_frame_mask=full_mask,
                confidence=conf,
                area_px=area,
                centroid_xy_px=centroid,
                principal_angle_deg=angle_deg,
            )
        )
    return instances


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute binary mask Intersection-over-Union in [0.0, 1.0]."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} != {mask_b.shape}")
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    return float(inter) / float(union)


def encode_mask_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a full-frame binary mask (0/255 uint8) to lossless PNG bytes."""
    arr = ((mask > 0).astype(np.uint8)) * 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def format_multi_box_prompt(
    boxes_xyxy: Sequence[tuple[int, int, int, int]],
    *,
    frame_index: int = 0,
) -> str:
    """Format an explicit multi-object `<0f>1<|box;...|>,2<|box;...|>` SAM 3.1 prompt."""
    parts = [
        f"{idx}<|box;{int(x1)},{int(y1)},{int(x2)},{int(y2)}|>"
        for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy, start=1)
    ]
    return f"<{frame_index}f>" + ",".join(parts)


class MetaSamClient:
    """Live & contract-validated client for Meta `sam-3.1` (`/v1/responses`)."""

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._external_client = client

    async def segment_image(
        self,
        image: bytes | Path | str,
        *,
        prompt_text: str,
        part_id: str = "unknown_part",
        include_confidence: bool = True,
    ) -> SamSegmentationResult:
        if isinstance(image, (Path, str)):
            raw_bytes = Path(image).read_bytes()
        else:
            raw_bytes = bytes(image)
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            width_px, height_px = pil_img.size
        b64_png = base64.b64encode(raw_bytes).decode("ascii")

        payload: dict[str, Any] = {
            "model": settings.model_sam,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_png}",
                        },
                        {
                            "type": "input_text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
        }
        if include_confidence:
            payload["metadata"] = {"include_confidence": "true"}

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

        raw_text = ""
        for item in body.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    raw_text += c.get("text", "")

        instances = decode_sam_token_string(
            raw_text,
            image_width_px=width_px,
            image_height_px=height_px,
            part_id=part_id,
            prompt_text=prompt_text,
        )
        req_id = resp.headers.get("x-request-id") or f"req-{body.get('id', 'sam')}"
        return SamSegmentationResult(
            provider_response_id=str(body.get("id", "resp_sam_unknown")),
            provider_request_id=req_id,
            image_width_px=width_px,
            image_height_px=height_px,
            raw_output_text=raw_text,
            latency_ms=round(latency_ms, 2),
            instances=instances,
            usage=body.get("usage", {}),
        )

    async def segment_multiple_prompts(
        self,
        image: bytes | Path | str,
        *,
        prompts: Mapping[str, str],
        include_confidence: bool = True,
    ) -> SamSegmentationResult:
        """Segment multiple canonical parts and merge instances into one SamSegmentationResult."""
        if isinstance(image, (Path, str)):
            raw_bytes = Path(image).read_bytes()
        else:
            raw_bytes = bytes(image)
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            width_px, height_px = pil_img.size

        all_instances: list[SamDetectedInstance] = []
        raw_texts: list[str] = []
        last_resp_id = "resp_sam_batch"
        last_req_id = "req_sam_batch"
        total_latency_ms = 0.0

        async with build_httpx_client() as shared_client:
            sub = MetaSamClient(client=shared_client)
            for part_id, prompt_text in prompts.items():
                res = await sub.segment_image(
                    raw_bytes,
                    prompt_text=prompt_text,
                    part_id=part_id,
                    include_confidence=include_confidence,
                )
                last_resp_id = res.provider_response_id
                last_req_id = res.provider_request_id
                total_latency_ms += res.latency_ms
                raw_texts.append(res.raw_output_text)
                for idx, inst in enumerate(res.instances):
                    inst.sam_track_id = f"{part_id}_{inst.sam_track_id}_{idx}"
                    all_instances.append(inst)

        return SamSegmentationResult(
            provider_response_id=last_resp_id,
            provider_request_id=last_req_id,
            image_width_px=width_px,
            image_height_px=height_px,
            raw_output_text="\n".join(raw_texts),
            latency_ms=round(total_latency_ms, 2),
            instances=all_instances,
        )
