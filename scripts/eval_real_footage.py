#!/usr/bin/env python3
"""Run labelled frames (or frames sampled from clips) through the furniture gate and print an honest table.

Usage:
  python3 scripts/eval_real_footage.py --labels footage/labels.csv [--mode fixture|live] [--every-n 15] [--out eval.json]
labels.csv columns: path,expected   (expected in {correct, holes_out, unmeasurable}); path may be an image or video.
Without --labels, evaluates the synthetic fixtures (clearly marked SYNTHETIC).
Live mode calls SAM + Baseten per frame (~$0.01-0.02/frame); --max-live caps calls.
"""
from __future__ import annotations

import argparse, asyncio, csv, json, sys, tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.checkpoints import default_checkpoint_set  # noqa: E402
from backend.app.furniture_gate import verify_furniture_checkpoint  # noqa: E402

EXPECT = {"correct": "accepted", "holes_out": "rejected", "unmeasurable": "unknown"}


def frames(path: Path, every_n: int):
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        yield path
        return
    cap, i, tmp = cv2.VideoCapture(str(path)), 0, Path(tempfile.mkdtemp())
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            out = tmp / f"{path.stem}_{i:05d}.png"
            cv2.imwrite(str(out), f)
            yield out
        i += 1


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels"); ap.add_argument("--mode", default="fixture", choices=["fixture", "live"])
    ap.add_argument("--every-n", type=int, default=15); ap.add_argument("--max-live", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    cs = default_checkpoint_set(); cfg = cs.step()
    if a.labels:
        rows = [(Path(r["path"]), r["expected"]) for r in csv.DictReader(open(a.labels))]
        source = "REAL FOOTAGE"
    else:
        m = {"ikea_pass_step3": "correct", "ikea_fail_holes_out": "holes_out", "ikea_unmeasurable_blur": "unmeasurable"}
        rows = [(cs.scenarios[k], v) for k, v in m.items()]
        source = "SYNTHETIC FIXTURES (not real footage)"
    results, live_calls = [], 0
    for p, exp in rows:
        for f in frames(p, a.every_n):
            mode = a.mode
            if mode == "live":
                if live_calls >= a.max_live:
                    mode = "fixture"
                live_calls += 1
            o = await verify_furniture_checkpoint(f, cfg, mode=mode)
            s1 = o["stage1"]
            results.append({"frame": str(f), "expected": exp, "mode": mode,
                            "provisional": s1.get("provisional_verdict"), "final": o["decision"],
                            "holes": (s1.get("hole_pattern_witness") or {}).get("status"),
                            "tape": (s1.get("tape_edge_witness") or {}).get("status"),
                            "iou": s1.get("panel_silhouette_iou"), "stage1_ms": s1.get("stage1_elapsed_ms"),
                            "stage2_ms": (o.get("stage2") or {}).get("wall_ms"),
                            "model_finding": (o.get("stage2") or {}).get("model_finding")})
    print(f"# {source} | mode={a.mode} | step={cfg.display_label}")
    print("| frame | expected | provisional | final | holes | tape | IoU | s1 ms | s2 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {Path(r['frame']).name} | {r['expected']} | {r['provisional']} | {r['final']} | {r['holes']} | "
              f"{r['tape']} | {r['iou'] if r['iou'] is None else round(r['iou'], 3)} | {r['stage1_ms']} | {r['stage2_ms']} |")
    wrong_accept = sum(1 for r in results if r["final"] == "accepted" and r["expected"] != "correct")
    caught = sum(1 for r in results if r["expected"] == "holes_out" and r["final"] == "rejected")
    n_out = sum(1 for r in results if r["expected"] == "holes_out")
    print(f"\nfalse accepts: {wrong_accept} | holes-out caught: {caught}/{n_out} | frames: {len(results)}")
    if a.mode != "live":
        print("NOTE: fixture mode never confirms a pass (stage 2 not run); 'correct' frames end as unknown by design.")
    if a.out:
        Path(a.out).write_text(json.dumps({"source": source, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
