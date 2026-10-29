"""Live probe: does Baseten muse-spark-open-1026 accept image_url parts, and does it catch the flipped rod?

Usage: python -m scripts.probe_baseten_vision [--runs 3] [--mode full|crop|both]
Never prints keys. Writes JSON results to docs/probes/baseten_vision_<ts>.json.
"""
from __future__ import annotations
import argparse, asyncio, base64, json, time
from pathlib import Path
import cv2
from backend.app.providers.baseten_vision import BasetenVisionAuditor
from backend.app.geometry import evaluate_elevated_checkpoint, GOLDEN_REF_PASS_IMAGE_PATH
from backend.app.calibration import compute_planar_calibration

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "golden_rig"
CASES = {"pass": ("checkpoint_pass_step2.png", "consistent"), "fail_flipped_rod": ("checkpoint_fail_flipped_rod.png", "mismatch")}

def crops(path: Path):
    img = cv2.imread(str(path))
    m = evaluate_elevated_checkpoint(img, cal=compute_planar_calibration(img))
    r, l = m.paired_crop_ref_b64, m.paired_crop_live_b64
    if not r or not l:
        return None
    dec = lambda s: base64.b64decode(s.split(",", 1)[-1])
    return dec(r), dec(l)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=3); ap.add_argument("--mode", default="both")
    a = ap.parse_args()
    aud = BasetenVisionAuditor(); out = []
    ref_full = GOLDEN_REF_PASS_IMAGE_PATH.read_bytes()
    for name, (fn, expected) in CASES.items():
        modes = ["full", "crop"] if a.mode == "both" else [a.mode]
        for mode in modes:
            if mode == "full":
                ref, cur = ref_full, (FIX / fn).read_bytes()
            else:
                c = crops(FIX / fn)
                if c is None:
                    out.append({"case": name, "mode": mode, "error": "no paired crops"}); continue
                ref, cur = c
            for i in range(a.runs):
                rec = {"case": name, "mode": mode, "run": i + 1, "expected": expected}
                try:
                    r = await aud.audit_checkpoint(current_image=cur, reference_image=ref,
                        blinded_context={"observation_id": "00000000-0000-4000-8000-000000000002", "frame_id": "probe"})
                    u = r.usage
                    rec.update(finding=r.audit["finding"], correct=r.audit["finding"] == expected, latency_ms=r.latency_ms,
                               prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                               total_tokens=u.get("total_tokens"), request_id=u.get("request_id"), raw_verdict=u.get("raw_verdict"))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(json.dumps({k: v for k, v in rec.items() if k != "raw_verdict"}), flush=True)
                out.append(rec)
    d = ROOT / "docs" / "probes"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"baseten_vision_{time.strftime('%Y%m%dT%H%M%S')}.json"; p.write_text(json.dumps(out, indent=2)); print("wrote", p)

if __name__ == "__main__":
    asyncio.run(main())
