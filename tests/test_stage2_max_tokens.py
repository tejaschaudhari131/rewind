import asyncio
from backend.app import furniture_gate as fg
from backend.app.providers import baseten_vision as bv


def test_stage2_max_tokens_raised():
    assert fg.STAGE2_MAX_TOKENS >= 4096


def test_stage2_auditor_built_with_raised_budget(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, *, max_tokens, **kw):
            seen["max_tokens"] = max_tokens

        async def audit_checkpoint(self, **kw):
            await asyncio.sleep(10)

    monkeypatch.setattr(bv, "BasetenVisionAuditor", Fake)
    monkeypatch.setattr(fg, "STAGE2_DEADLINE_S", 0.05)
    from backend.app.checkpoints import default_checkpoint_set
    cfg = default_checkpoint_set().step("step-3-left-side-panel")
    s1 = {"provisional_verdict": "provisional_pass", "observation_id": "x"}
    out = asyncio.run(fg.run_stage2(cfg.reference_image, cfg, s1, mode="live"))
    assert seen["max_tokens"] >= 4096
    assert out["error"] == "stage2_timeout" and out["decision"] == "unknown" and out["measured"] is False
