from __future__ import annotations

import asyncio
import io
import json
import re
import socket
import ssl
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import websockets

from ..config import Settings, settings

CANONICAL_VOICE_COMMANDS: dict[str, tuple[str, ...]] = {
    "check_step": ("check step", "verify step", "check this step", "inspect step"),
    "explain_failure": (
        "why did that fail",
        "why did it fail",
        "explain failure",
        "what failed",
    ),
    "scan_flat_part": (
        "scan flat part",
        "scan part",
        "measure flat part",
        "measure link",
    ),
    "generate_replacement": (
        "generate replacement",
        "create replacement",
        "build replacement",
        "make replacement",
    ),
    "repeat_instruction": (
        "repeat instruction",
        "repeat step",
        "say that again",
        "repeat",
    ),
}

DEFAULT_VOICE_KEYWORDS: tuple[str, ...] = (
    "check step",
    "why did that fail",
    "scan flat part",
    "generate replacement",
    "repeat instruction",
)


@dataclass(slots=True)
class MappedVoiceCommand:
    intent: str
    raw_transcript: str
    matched_phrase: str | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.intent == other
        if isinstance(other, MappedVoiceCommand):
            return self.intent == other.intent
        return False


@dataclass(slots=True)
class VoiceTranscriptResult:
    session_id: str | None
    transcript: str
    command: MappedVoiceCommand
    events: list[dict[str, Any]]
    handshake_latency_ms: float
    total_latency_ms: float


def map_transcript_to_command(transcript: str) -> MappedVoiceCommand:
    """Deterministically map a spoken utterance transcript to one of the 5 canonical REWIND commands."""
    normalized = re.sub(r"[^a-z0-9\s]+", " ", (transcript or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return MappedVoiceCommand(intent="unrecognized", raw_transcript=transcript)

    if re.search(r"\b(don t|dont|do not|never|stop|cancel|not)\b", normalized):
        return MappedVoiceCommand(intent="unrecognized", raw_transcript=transcript)

    for cmd_id, phrases in CANONICAL_VOICE_COMMANDS.items():
        for phrase in phrases:
            if phrase in normalized:
                return MappedVoiceCommand(
                    intent=cmd_id,
                    raw_transcript=transcript,
                    matched_phrase=phrase,
                )
    return MappedVoiceCommand(intent="unrecognized", raw_transcript=transcript)


def _extract_pcm24k_mono_s16le(audio_source: bytes | str | Path) -> bytes:
    """Extract raw 24kHz 16-bit signed LE mono PCM bytes from a WAV file or raw PCM buffer."""
    raw = (
        Path(audio_source).read_bytes()
        if isinstance(audio_source, (str, Path))
        else bytes(audio_source)
    )
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        with wave.open(io.BytesIO(raw), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 24000:
                raise ValueError(
                    f"WAV must be 24000Hz 16-bit mono PCM; got {wf.getframerate()}Hz, "
                    f"{wf.getsampwidth() * 8}-bit, {wf.getnchannels()}ch."
                )
            return wf.readframes(wf.getnframes())
    return raw


def _open_proxy_tls_socket(
    ws_url: str, proxy_url: str | None, timeout_s: float = 15.0
) -> ssl.SSLSocket:
    parsed_ws = urlparse(ws_url)
    target_host = parsed_ws.hostname or "api.meta.ai"
    target_port = parsed_ws.port or (443 if parsed_ws.scheme == "wss" else 80)

    if proxy_url:
        parsed_proxy = urlparse(proxy_url)
        proxy_host = parsed_proxy.hostname or "127.0.0.1"
        proxy_port = parsed_proxy.port or 8791
        sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout_s)
        connect_req = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            f"Host: {target_host}:{target_port}\r\n\r\n"
        ).encode("ascii")
        sock.sendall(connect_req)
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise ConnectionError("Proxy closed connection during CONNECT handshake.")
            resp += chunk
        status_line = resp.splitlines()[0].decode("latin1", errors="replace")
        if "200" not in status_line:
            sock.close()
            raise ConnectionError(f"Proxy CONNECT failed: {status_line}")
    else:
        sock = socket.create_connection((target_host, target_port), timeout=timeout_s)

    sock.setblocking(False)
    return sock


class MetaVoiceClient:
    """Realtime WebSocket client for `muse-voice-transcribe-1.0` over HTTP CONNECT proxy."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg

    async def transcribe_pcm(
        self,
        audio_source: bytes | str | Path,
        *,
        keywords: Sequence[str] = DEFAULT_VOICE_KEYWORDS,
        mode: str = "ENDPOINTING",
        chunk_bytes: int = 9600,
        chunk_delay_s: float = 0.04,
        receive_window_s: float = 3.0,
    ) -> VoiceTranscriptResult:
        """Stream 24kHz 16-bit mono PCM audio to `wss://api.meta.ai/v1/asr/realtime` and map to command."""
        self.cfg.require_meta_api_key()
        pcm_bytes = _extract_pcm24k_mono_s16le(audio_source)

        t0 = time.perf_counter()
        tls_sock = await asyncio.to_thread(
            _open_proxy_tls_socket,
            self.cfg.meta_asr_ws_url,
            self.cfg.proxy_url,
        )

        events: list[dict[str, Any]] = []
        session_id: str | None = None
        latest_transcript = ""
        handshake_ms = 0.0

        ssl_ctx = ssl.create_default_context()
        async with websockets.connect(
            self.cfg.meta_asr_ws_url,
            sock=tls_sock,
            ssl=ssl_ctx,
            server_hostname="api.meta.ai",
        ) as ws:
            config_frame = {
                "authorization": {"accessToken": f"Bearer {self.cfg.meta_api_key}"},
                "audioEncoding": "PCM_24KHZ",
                "model": self.cfg.voice_model,
                "mode": mode,
                "keywords": list(keywords),
            }
            await ws.send(json.dumps(config_frame))
            first_msg_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            handshake_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            first_msg = json.loads(first_msg_raw)
            events.append(first_msg)
            session_id = first_msg.get("sessionId")

            for offset in range(0, len(pcm_bytes), chunk_bytes):
                await ws.send(pcm_bytes[offset : offset + chunk_bytes])
                await asyncio.sleep(chunk_delay_s)

            deadline = time.perf_counter() + receive_window_s
            while time.perf_counter() < deadline:
                remaining = max(0.05, deadline - time.perf_counter())
                try:
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(msg_raw)
                events.append(msg)
                if msg.get("type") == "transcript":
                    text_val = str(msg.get("transcript", "")).strip()
                    if text_val:
                        latest_transcript = text_val
                    if msg.get("final") is True:
                        break
                elif msg.get("type") == "speechEnd":
                    try:
                        extra_raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        extra = json.loads(extra_raw)
                        events.append(extra)
                        if extra.get("type") == "transcript" and extra.get("transcript"):
                            latest_transcript = str(extra["transcript"]).strip()
                    except asyncio.TimeoutError:
                        pass
                    break

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return VoiceTranscriptResult(
            session_id=session_id,
            transcript=latest_transcript,
            command=map_transcript_to_command(latest_transcript),
            events=events,
            handshake_latency_ms=handshake_ms,
            total_latency_ms=total_ms,
        )

    async def transcribe_wav_24khz(
        self,
        wav_path: bytes | str | Path,
        **kwargs: Any,
    ) -> VoiceTranscriptResult:
        """Convenience alias for transcribe_pcm on a 24kHz mono WAV file."""
        return await self.transcribe_pcm(wav_path, **kwargs)
