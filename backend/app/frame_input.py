"""Real-frame input for the furniture gate: uploaded photos, phone video frames, live camera grabs.

Images are validated (type, <= 8 MB, decodable), normalised to PNG (HEIC converted server-side when
pillow-heif is installed), downscaled so the long side is <= 1280 px, and written to a temp file that is
deleted after the gate runs. Nothing is persisted unless the caller sets save_to_eval, in which case a
copy goes to fixtures/real_footage/ (gitignored)."""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

try:  # optional HEIC/HEIF support
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:  # pragma: no cover
    HEIC_SUPPORTED = False

MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 1280
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"}
SOURCE_LABELS = {"camera": "LIVE CAMERA", "upload": "UPLOADED PHOTO", "video_frame": "UPLOADED VIDEO FRAME"}
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(os.getenv("REWIND_REAL_FOOTAGE_DIR", str(REPO_ROOT / "fixtures" / "real_footage")))


class FrameInputError(ValueError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def decode_frame(data: bytes, content_type: Optional[str], filename: str = "") -> Image.Image:
    if not data:
        raise FrameInputError(400, "empty_file")
    if len(data) > MAX_BYTES:
        raise FrameInputError(413, f"file_too_large: {len(data)} bytes > {MAX_BYTES}")
    ct = (content_type or "").lower().split(";")[0].strip()
    ext = Path(filename).suffix.lower()
    if ct not in ALLOWED_TYPES and not (ct in ("", "application/octet-stream") and ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")):
        raise FrameInputError(415, f"unsupported_type: {ct or ext or 'unknown'}")
    if (ct in ("image/heic", "image/heif") or ext in (".heic", ".heif")) and not HEIC_SUPPORTED:
        raise FrameInputError(415, "heic_not_supported_on_server: export as JPG")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise FrameInputError(400, f"not_a_decodable_image: {type(exc).__name__}") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def write_temp_png(img: Image.Image) -> Path:
    fd, name = tempfile.mkstemp(prefix="rewind_frame_", suffix=".png")
    with os.fdopen(fd, "wb") as f:
        img.save(f, format="PNG")
    return Path(name)


def save_to_eval(img: Image.Image, source: str, step_id: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = EVAL_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{step_id}_{source}.png"
    img.save(p, format="PNG")
    return p
