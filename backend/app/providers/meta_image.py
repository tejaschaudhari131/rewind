from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings, settings


@dataclass(slots=True)
class RepairCardResult:
    image_bytes: bytes
    b64_json: str
    latency_ms: float
    output_path: Path | None = None


class MetaImageClient:
    """Async visual repair card generator using `muse-image-1.0` (`POST /v1/images/edits`)."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg

    def _headers(self) -> dict[str, str]:
        self.cfg.require_meta_api_key()
        return {
            "Authorization": f"Bearer {self.cfg.meta_api_key}",
            "Content-Type": "application/json",
        }

    async def generate_repair_card(
        self,
        *,
        image_source: bytes | str | Path,
        prompt: str,
        output_path: str | Path | None = None,
        reasoning_strength: str = "low",
        timeout_s: float = 35.0,
    ) -> RepairCardResult:
        """Edit an overhead frame into an annotated repair instruction card."""
        raw_bytes = (
            Path(image_source).read_bytes()
            if isinstance(image_source, (str, Path))
            else bytes(image_source)
        )
        b64_in = base64.b64encode(raw_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_in}"

        req_payload: dict[str, Any] = {
            "model": self.cfg.image_model,
            "prompt": prompt,
            "images": [{"url": data_uri}],
            "reasoning_strength": reasoning_strength,
            "enable_web_search": False,
            "enable_image_search": False,
            "enable_shell": False,
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(proxy=self.cfg.proxy_url, timeout=timeout_s) as client:
            resp = await client.post(
                f"{self.cfg.meta_api_base_url}/images/edits",
                headers=self._headers(),
                json=req_payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        b64_out = str(data["data"][0]["b64_json"])
        img_bytes = base64.b64decode(b64_out)

        saved_path: Path | None = None
        if output_path is not None:
            saved_path = Path(output_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_bytes(img_bytes)

        return RepairCardResult(
            image_bytes=img_bytes,
            b64_json=b64_out,
            latency_ms=latency_ms,
            output_path=saved_path,
        )
