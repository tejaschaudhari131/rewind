from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import build_httpx_client, settings
from backend.app.providers.meta_sam import MetaSamClient
from backend.app.providers.meta_voice import MetaVoiceClient
from backend.app.providers.meta_wonder import MetaWonderClient
from backend.app.providers.modal_glimmer import ModalGlimmerClient
from backend.app.schemas import (
    make_envelope,
    validate_contract_envelope,
    validate_glimmer_output,
    validate_wonder_output,
)

import socket
from urllib.parse import urlparse

def _skip_if_proxy_unreachable() -> None:
    if not settings.proxy_url:
        return
    parsed = urlparse(settings.proxy_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8791
    try:
        with socket.create_connection((host, port), timeout=0.6):
            pass
    except OSError:
        pytest.skip(f"Live Meta proxy tunnel ({host}:{port}) is currently offline")
    import httpx
    try:
        with httpx.Client(proxy=settings.proxy_url, timeout=1.5) as client:
            r = client.get(f"{settings.meta_api_base}/models", headers=settings.auth_headers())
            if r.status_code != 200 or not r.json().get("data"):
                pytest.skip("Live Meta proxy tunnel is not routing api.meta.ai models endpoint")
    except Exception:
        pytest.skip("Live Meta proxy tunnel is not routing api.meta.ai models endpoint")

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SYNTHETIC_FRAME = FIXTURES_DIR / "synthetic_rig_frame.png"
VOICE_WAV = FIXTURES_DIR / "check_step_24khz.wav"


@pytest.mark.asyncio
async def test_live_models_endpoint_via_proxy() -> None:
    _skip_if_proxy_unreachable()
    async with build_httpx_client() as client:
        resp = await client.get(
            f"{settings.meta_api_base}/models",
            headers=settings.auth_headers(),
        )
    assert resp.status_code == 200
    model_ids = {m["id"] for m in resp.json().get("data", [])}
    expected = {
        "sam-3.1",
        "muse-spark-1.3-contributor",
        "muse-wonder-1.0",
        "muse-voice-transcribe-1.0",
        "muse-image-1.0",
    }
    assert expected.issubset(model_ids)


@pytest.mark.asyncio
async def test_live_sam31_segmentation_and_contract_envelope() -> None:
    _skip_if_proxy_unreachable()
    sam = MetaSamClient()
    result = await sam.segment_image(
        SYNTHETIC_FRAME,
        prompt_text="blue slider block",
        part_id="slider_block",
        include_confidence=True,
    )
    assert result.provider_response_id.startswith("resp_")
    assert len(result.instances) >= 1
    inst = result.instances[0]
    assert inst.full_frame_mask.shape == (400, 640)
    assert inst.area_px > 1000
    assert 0.5 <= inst.confidence <= 1.0

    envelope = make_envelope(
        payload_type="segmentation",
        payload=result.to_segmentation_payload(),
    )
    validate_contract_envelope(envelope)


@pytest.mark.asyncio
async def test_live_wonder_compiler_and_blinded_glimmer_contributor() -> None:
    _skip_if_proxy_unreachable()
    obs_id = "22222222-2222-4222-8222-222222222222"
    wonder = MetaWonderClient()  # defaults to muse-spark-1.3-contributor
    w_res = await wonder.compile_planner_proposal(
        observation_id=obs_id,
        step_id="attach-connecting-rod",
        part_id="connecting-rod",
        evidence_ids=["frame-000120"],
        required_predicate_ids=["rod_crank_joint_seated", "rod_slider_joint_seated"],
        geometry_summary={"status": "fail", "reason": "flipped_connecting_rod"},
        measurement_record_id="meas-flat-link-001",
    )
    validate_wonder_output(w_res.proposal)
    assert w_res.proposal["observation_id"] == obs_id
    plan_env = make_envelope(
        payload_type="planner_proposal",
        observation_id=obs_id,
        payload=w_res.to_planner_proposal_payload(),
    )
    validate_contract_envelope(plan_env)

    glimmer = ModalGlimmerClient(modal_endpoint="")  # contributor fallback mode
    g_res = await glimmer.audit_checkpoint(
        current_image=SYNTHETIC_FRAME,
        reference_image=None,
        blinded_context={
            "observation_id": obs_id,
            "step_id": "attach-connecting-rod",
            "part_id": "connecting-rod",
            "frame_id": "frame-000120",
            "check_objective": "Verify yellow connecting rod bridges red crank disk and blue slider block.",
        },
    )
    validate_glimmer_output(g_res.audit)
    assert g_res.audit["observation_id"] == obs_id
    audit_env = make_envelope(
        payload_type="visual_audit",
        observation_id=obs_id,
        payload=g_res.to_visual_audit_payload(),
    )
    validate_contract_envelope(audit_env)


@pytest.mark.asyncio
async def test_live_voice_transcribe_websocket_via_proxy() -> None:
    _skip_if_proxy_unreachable()
    voice = MetaVoiceClient()
    res = await voice.transcribe_wav_24khz(VOICE_WAV)
    assert res.session_id
    assert "check" in res.transcript.lower() and "step" in res.transcript.lower()
    assert res.command.intent == "check_step"


@pytest.mark.asyncio
async def test_live_baseten_muse_spark_open_1026_arbitration() -> None:
    from backend.app.config import get_settings
    from backend.app.providers.baseten_spark import BasetenSparkClient

    cfg = get_settings()
    if not cfg.baseten_api_key:
        pytest.skip("BASETEN_API_KEY is not configured in .env")

    client = BasetenSparkClient(baseten_api_key=cfg.baseten_api_key, cfg=cfg)
    assert client.cfg.baseten_spark_model == "meta-models/muse-spark-open-1026"
    res = await client.arbitrate_ambiguous_checkpoint(
        session_id="pytest-live-baseten-1026",
        step_id="step-2-connecting-rod",
        silhouette_iou=0.9628,
        witness_polarity_cosine=-0.9412,
        witness_error_px=41.20,
        glimmer_finding="mismatch (witness_mark_inverted_180_deg)",
        mode="live",
    )
    assert res["triggered"] is True
    assert res["provider"] == "BASETEN_OPEN_WEIGHT"
    assert res["model"] == "meta-models/muse-spark-open-1026"
    assert res["consensus"] == "DUAL_OPEN_WEIGHT_REJECT_180_FLIP"
    assert res["latency_ms"] > 10.0


@pytest.mark.asyncio
async def test_live_modal_glimmer_endpoint_smoke_and_blinding() -> None:
    import httpx
    from backend.app.config import get_settings
    from modal_app.glimmer_server import glimmer_asgi_app

    cfg = get_settings()
    golden_dir = FIXTURES_DIR / "golden_rig"
    pass_img = golden_dir / "checkpoint_pass_step2.png"
    flip_img = golden_dir / "checkpoint_fail_flipped_rod.png"

    if cfg.modal_glimmer_endpoint:
        endpoint = cfg.modal_glimmer_endpoint
        base_url = endpoint.rsplit("/v1/chat/completions", 1)[0]
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as http_client:
            r_health = await http_client.get(f"{base_url}/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            r_warm = await http_client.post(f"{base_url}/warmup")
            assert r_warm.status_code == 200
            assert r_warm.json()["status"] == "warm"

            # Blinding violation check over HTTP
            r_blind = await http_client.post(
                endpoint,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": 'Context: {"observation_id": "00000000-0000-4000-8000-000000000077", "mask_iou": 0.97}',
                                }
                            ],
                        }
                    ]
                },
            )
            assert r_blind.status_code == 422
            assert r_blind.json()["detail"]["code"] == "GLIMMER_BLINDING_VIOLATION"

        glimmer = ModalGlimmerClient(modal_endpoint=endpoint)
        res_pass = await glimmer.audit_checkpoint(
            current_image=pass_img,
            reference_image=None,
            blinded_context={
                "observation_id": "00000000-0000-4000-8000-000000000081",
                "step_id": "attach-connecting-rod",
                "part_id": "connecting-rod",
                "frame_id": "frame-000201",
                "check_objective": "Verify yellow connecting rod witness mark aligns with red crank disk.",
            },
        )
        assert res_pass.audit["finding"] == "consistent"

        res_flip = await glimmer.audit_checkpoint(
            current_image=flip_img,
            reference_image=None,
            blinded_context={
                "observation_id": "00000000-0000-4000-8000-000000000082",
                "step_id": "attach-connecting-rod",
                "part_id": "connecting-rod",
                "frame_id": "frame-000202",
                "check_objective": "Verify yellow connecting rod witness mark aligns with red crank disk.",
            },
        )
        assert res_flip.audit["finding"] == "mismatch"
        assert "witness-stripe-aligned" in res_flip.audit["contradicted_predicate_ids"]
    else:
        # In-process ASGI verification when MODAL_GLIMMER_ENDPOINT is awaiting workspace billing unlock
        from fastapi.testclient import TestClient

        tc = TestClient(glimmer_asgi_app)
        assert tc.get("/health").status_code == 200
        assert tc.post("/warmup").json()["status"] == "warm"
