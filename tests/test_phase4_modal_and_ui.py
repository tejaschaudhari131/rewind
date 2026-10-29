from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.providers.meta_wonder import MetaWonderClient
from backend.app.providers.modal_glimmer import ModalGlimmerClient
from modal_app.glimmer_server import (
    APP_NAME,
    GLIMMER_HF_REPO,
    MODAL_WORKSPACE,
    glimmer_asgi_app,
)

ROOT_DIR = Path("/workspace/rewind")
GOLDEN_DIR = ROOT_DIR / "fixtures" / "golden_rig"


def test_modal_glimmer_server_health_and_blinded_audit() -> None:
    client = TestClient(glimmer_asgi_app)
    h = client.get("/health")
    assert h.status_code == 200
    h_json = h.json()
    assert h_json["app"] == APP_NAME
    assert h_json["workspace"] == MODAL_WORKSPACE
    assert h_json["model"] == GLIMMER_HF_REPO
    assert h_json["gpu"] == "H100"
    assert h_json["enable_memory_snapshot"] is True

    # 1. Valid blinded pass frame
    pass_bytes = (GOLDEN_DIR / "checkpoint_pass_step2.png").read_bytes()
    pass_b64 = base64.b64encode(pass_bytes).decode("ascii")
    obs_id = "11111111-2222-4333-8444-555555555555"
    frame_id = "66666666-7777-4888-8999-000000000000"

    resp_pass = client.post(
        "/v1/chat/completions",
        json={
            "model": GLIMMER_HF_REPO,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{pass_b64}"},
                        {
                            "type": "input_text",
                            "text": f"Context: {json.dumps({'observation_id': obs_id, 'frame_id': frame_id, 'part_id': 'connecting-rod'})}",
                        },
                    ],
                }
            ],
        },
    )
    assert resp_pass.status_code == 200
    body_pass = resp_pass.json()
    parsed_pass = json.loads(body_pass["choices"][0]["message"]["content"])
    assert parsed_pass["observation_id"] == obs_id
    assert parsed_pass["finding"] == "consistent"
    assert body_pass["glimmer_verdict"]["verdict"] == "pass"
    assert body_pass["glimmer_verdict"]["witness_mark_aligned"] is True

    # 2. Valid blinded flipped-rod fail frame
    fail_bytes = (GOLDEN_DIR / "checkpoint_fail_flipped_rod.png").read_bytes()
    fail_b64 = base64.b64encode(fail_bytes).decode("ascii")
    resp_fail = client.post(
        "/v1/chat/completions",
        json={
            "model": GLIMMER_HF_REPO,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{fail_b64}"},
                        {
                            "type": "input_text",
                            "text": f"Context: {json.dumps({'observation_id': obs_id, 'frame_id': frame_id, 'part_id': 'connecting-rod'})}",
                        },
                    ],
                }
            ],
        },
    )
    assert resp_fail.status_code == 200
    body_fail = resp_fail.json()
    parsed_fail = json.loads(body_fail["choices"][0]["message"]["content"])
    assert parsed_fail["finding"] == "mismatch"
    assert body_fail["glimmer_verdict"]["verdict"] == "fail"
    assert body_fail["glimmer_verdict"]["witness_mark_aligned"] is False

    # 3. Blinding violation (forbidden key `expected_pose_id` or `sam_mask_iou`) must be rejected with 422
    resp_unblinded = client.post(
        "/v1/chat/completions",
        json={
            "model": GLIMMER_HF_REPO,
            "blinded_context": {"observation_id": obs_id, "expected_pose_id": "pose_step_2"},
            "messages": [{"role": "user", "content": "audit"}],
        },
    )
    assert resp_unblinded.status_code == 422
    assert resp_unblinded.json()["detail"]["code"] == "GLIMMER_BLINDING_VIOLATION"


@pytest.mark.asyncio
async def test_deterministic_id_pinning_on_llm_hyphen_typo() -> None:
    obs_id = "22222222-2222-4222-8222-222222222222"
    frame_id = "33333333-3333-4333-8333-333333333333"
    typo_obs_id = "222222222222-4222-8222-222222222222"  # missing first hyphen

    def mock_handler(request: httpx.Request) -> httpx.Response:
        req_data = json.loads(request.content.decode("utf-8"))
        fmt_name = req_data.get("text", {}).get("format", {}).get("name")
        if fmt_name == "wonder_output":
            payload = {
                "observation_id": typo_obs_id,
                "action": "generate_supported_part",
                "part_id": "wrong-part",
                "instruction": "Generate replacement part.",
                "evidence_ids": ["typo-frame"],
                "required_predicate_ids": ["rod_crank_joint_seated"],
                "measurement_record_id": "typo-meas",
            }
        else:
            payload = {
                "observation_id": typo_obs_id,
                "finding": "consistent",
                "observations": [
                    {
                        "frame_id": "typo-frame-id",
                        "region_xyxy_px": [10, 10, 100, 100],
                        "statement": "Looks aligned.",
                    }
                ],
                "contradicted_predicate_ids": [],
                "requires_new_view": False,
            }
        return httpx.Response(
            200,
            json={
                "id": "resp_mock_pinning",
                "model": "muse-spark-1.3-contributor",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        wonder = MetaWonderClient(client=mock_client)
        w_res = await wonder.compile_planner_proposal(
            observation_id=obs_id,
            step_id="attach-connecting-rod",
            part_id="connecting-rod",
            evidence_ids=[frame_id],
            required_predicate_ids=["rod_crank_joint_seated"],
            geometry_summary={"status": "fail"},
            measurement_record_id="meas-flat-link-001",
        )
        assert w_res.proposal["observation_id"] == obs_id
        assert w_res.proposal["part_id"] == "connecting-rod"
        assert w_res.proposal["evidence_ids"] == [frame_id]
        assert w_res.proposal["measurement_record_id"] == "meas-flat-link-001"

        # Glimmer identity mismatches are no longer silently rewritten: they fail closed.
        glimmer = ModalGlimmerClient(modal_endpoint="", client=mock_client)
        with pytest.raises(ValueError):
            await glimmer.audit_checkpoint(
                current_image=GOLDEN_DIR / "checkpoint_pass_step2.png",
                reference_image=None,
                blinded_context={
                    "observation_id": obs_id,
                    "frame_id": frame_id,
                    "part_id": "connecting-rod",
                },
            )


def test_phase4_workbench_and_asset_endpoints() -> None:
    client = TestClient(app)

    # 1. Frontend dist served at /
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers["content-type"]

    # 2. Golden, Printable, and CAD asset endpoints
    golden_resp = client.get("/api/assets/golden/checkpoint_pass_step2.png")
    assert golden_resp.status_code == 200
    assert golden_resp.headers["content-type"] == "image/png"

    pdf_resp = client.get("/api/assets/printable/rewind_apriltag_mat_400x250mm.pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"

    stl_resp = client.get("/api/assets/cad/flat_two_hole_link_v1.stl")
    assert stl_resp.status_code == 200
    assert len(stl_resp.content) > 1000

    step_resp = client.get("/api/assets/cad/flat_two_hole_link_v1.step")
    assert step_resp.status_code == 200
    assert b"ISO-10303-21" in step_resp.content

    # 3. Teardown compile & CUA inspect endpoints
    comp_resp = client.post("/api/teardown/compile", json={"mode": "fixture"})
    assert comp_resp.status_code == 200
    assert len(comp_resp.json()["compiled_steps"]) == 4

    cua_resp = client.post("/api/bonus/cua-inspect", json={"mode": "fixture"})
    assert cua_resp.status_code == 200
    assert cua_resp.json()["tool_declaration"]["type"] == "computer"
