from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)

DEFAULT_LOCAL_PROXY = "http://127.0.0.1:8791"


def _resolve_proxy() -> str | None:
    proxy = (
        os.getenv("REWIND_PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("http_proxy")
    )
    if proxy:
        return proxy.strip()
    return DEFAULT_LOCAL_PROXY


@dataclass(frozen=True)
class Settings:
    # Secrets are excluded from repr to prevent accidental logging
    meta_api_key: str = field(
        default_factory=lambda: os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY", ""),
        repr=False,
    )
    baseten_api_key: str = field(
        default_factory=lambda: os.getenv("BASETEN_API_KEY", ""),
        repr=False,
    )
    modal_glimmer_api_key: str = field(
        default_factory=lambda: os.getenv("MODAL_GLIMMER_API_KEY")
        or os.getenv("MODAL_GLIMMER_TOKEN", ""),
        repr=False,
    )

    # Endpoints & Proxy
    meta_api_base: str = field(
        default_factory=lambda: os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    )
    meta_asr_ws_url: str = field(
        default_factory=lambda: os.getenv(
            "META_VOICE_WS_URL", "wss://api.meta.ai/v1/asr/realtime"
        )
    )
    http_proxy: str | None = field(default_factory=_resolve_proxy)
    http_timeout_s: float = 35.0
    modal_glimmer_endpoint: str = field(
        default_factory=lambda: os.getenv("MODAL_GLIMMER_ENDPOINT")
        or os.getenv("MODAL_GLIMMER_URL", "")
    )
    baseten_base_url: str = field(
        default_factory=lambda: os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
    )

    # Runtime Model IDs
    model_sam: str = "sam-3.1"
    compiler_model: str = field(
        default_factory=lambda: os.getenv("REWIND_COMPILER_MODEL", "muse-spark-1.3-contributor")
    )
    model_wonder_full: str = "muse-wonder-1.0"
    model_spark_contributor: str = "muse-spark-1.3-contributor"
    voice_model: str = "muse-voice-transcribe-1.0"
    image_model: str = "muse-image-1.0"
    baseten_spark_model: str = field(
        default_factory=lambda: os.getenv(
            "BASETEN_SPARK_MODEL", "meta-models/muse-spark-open-1026"
        )
    )
    glimmer_modal_model: str = "facebook/Muse-Glimmer-30B"

    # Gate & Metrology Thresholds (from CONTRACT.md & LOCK-IN.md)
    gate_deadline_ms: int = 2800
    calibration_max_age_ms: int = 2000
    calibration_max_corner_drift_px: float = 1.5
    calibration_max_reproj_rms_mm: float = 0.25
    sam_min_confidence: float = 0.65
    sam_min_iou: float = 0.78
    center_tolerance_px: float = 10.0
    angle_tolerance_deg: float = 8.0
    measurement_sigma_max_mm: float = 0.25

    @property
    def model_api_key(self) -> str:
        return self.meta_api_key

    @property
    def proxy_url(self) -> str | None:
        return self.http_proxy

    @property
    def meta_api_base_url(self) -> str:
        return self.meta_api_base

    @property
    def glimmer_fallback_model(self) -> str:
        return self.model_spark_contributor

    def require_meta_api_key(self) -> str:
        if not self.meta_api_key:
            raise RuntimeError("META_API_KEY is not configured in /workspace/rewind/.env")
        return self.meta_api_key

    def auth_headers(self) -> dict[str, str]:
        self.require_meta_api_key()
        return {
            "Authorization": f"Bearer {self.meta_api_key}",
            "Content-Type": "application/json",
        }


def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def build_httpx_client(*, timeout_s: float | None = None) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient automatically configured with the local tunnel proxy."""
    return httpx.AsyncClient(
        proxy=settings.http_proxy,
        timeout=timeout_s or settings.http_timeout_s,
    )
