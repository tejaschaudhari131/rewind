"""Real-frame endpoint: validation, honest source labels, no persistence by default, token gate."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import frame_input
from backend.app.checkpoints import default_checkpoint_set
from backend.app.main import app

CS = default_checkpoint_set()
client = TestClient(app)
FAIL = Path(CS.scenarios["ikea_fail_holes_out"]).read_bytes()
PASS = Path(CS.scenarios["ikea_pass_step3"]).read_bytes()


@pytest.fixture(autouse=True)
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "real_footage"
    monkeypatch.setattr(frame_input, "EVAL_DIR", d)
    monkeypatch.delenv("REWIND_ACCESS_TOKEN", raising=False)
    return d


def post(data, ct="image/png", name="f.png", **form):
    form.setdefault("mode", "fixture")  # never hit live APIs from tests
    return client.post("/api/furniture/checkpoint/verify_frame", files={"file": (name, data, ct)}, data=form)


@pytest.mark.parametrize("source,label", [("camera", "LIVE CAMERA"), ("upload", "UPLOADED PHOTO"), ("video_frame", "UPLOADED VIDEO FRAME")])
def test_label_reflects_source_and_not_fixture(source, label):
    r = post(FAIL, source=source)
    assert r.status_code == 200
    j = r.json()
    assert j["input_label"] == label and j["input_source"] == source
    assert j["synthetic_fixture"] is False
    assert j["decision"] == "rejected"  # same gate as fixtures: flipped panel is a STOP


def test_pass_frame_in_fixture_mode_never_accepts_or_ticks():
    j = post(PASS, source="upload", check_off_manual="true").json()
    assert j["decision"] == "unknown" and j["stage2"]["measured"] is False
    assert j["manual"]["status"] == "not_updated"


def test_no_persistence_by_default(eval_dir):
    j = post(FAIL).json()
    assert j["persisted"] is False and "saved_path" not in j
    assert not eval_dir.exists() or not any(eval_dir.iterdir())


def test_save_to_eval_persists(eval_dir):
    j = post(FAIL, save_to_eval="true", source="camera").json()
    assert j["persisted"] is True
    files = list(eval_dir.glob("*.png"))
    assert len(files) == 1 and "camera" in files[0].name


def test_jpeg_accepted_and_downscaled():
    buf = io.BytesIO(); Image.open(io.BytesIO(FAIL)).resize((2560, 1440)).save(buf, "JPEG")
    j = post(buf.getvalue(), ct="image/jpeg", name="p.jpg").json()
    assert max(j["frame_size"]) <= 1280


@pytest.mark.parametrize("data,ct,name,code", [
    (b"", "image/png", "e.png", 400),
    (b"not an image", "image/png", "x.png", 400),
    (b"GIF89a", "image/gif", "x.gif", 415),
    (b"%PDF-1.4", "application/pdf", "x.pdf", 415),
])
def test_validation_rejects_bad_input(data, ct, name, code):
    assert post(data, ct=ct, name=name).status_code == code


def test_size_cap():
    assert post(b"\x89PNG" + b"0" * (frame_input.MAX_BYTES + 10)).status_code == 413


def test_access_token_gate(monkeypatch):
    monkeypatch.setenv("REWIND_ACCESS_TOKEN", "s3cret-test")
    assert client.get("/api/furniture/config").status_code == 401
    assert client.get("/api/furniture/config", headers={"x-rewind-token": "wrong"}).status_code == 401
    assert client.get("/api/furniture/config?t=s3cret-test").status_code == 200
    assert client.get("/api/furniture/config", headers={"x-rewind-token": "s3cret-test"}).status_code == 200


def test_preflight_shape():
    j = client.get("/api/furniture/preflight").json()
    assert "meta_reachable" in j and "api_key" not in str(j).lower().replace("has_meta_key", "").replace("has_baseten_key", "")
