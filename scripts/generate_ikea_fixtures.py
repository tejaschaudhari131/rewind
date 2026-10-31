"""Generate SYNTHETIC flat-pack side-panel fixtures (not real footage) so the furniture flow is demoable offline.

Frames (1280x720, overhead view of a panel lying on a desk):
  reference_step3.png          correct: cam-lock + shelf-pin holes visible, blue tape at the LEFT end of the front edge
  checkpoint_pass_step3.png    correct, with small pose jitter / lighting change
  checkpoint_fail_holes_out.png panel mounted holes-facing-OUT: blank face visible, tape moved to the RIGHT end
  checkpoint_unmeasurable_blur.png  heavy motion blur / dim: witness must fail closed (insufficient evidence)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "ikea"
W, H = 1280, 720
PANEL_W, PANEL_H = 820, 300


def _desk(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (62, 66, 72), np.uint8).astype(np.int16)
    img += rng.integers(-6, 7, size=(H, W, 1), dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def render(*, holes_visible: bool, tape_left: bool, dx: int = 0, dy: int = 0, angle: float = 0.0,
           gain: float = 1.0, seed: int = 0) -> np.ndarray:
    img = _desk(seed)
    panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    panel[:] = (196, 214, 226)  # birch/white-ish laminate (BGR)
    rng = np.random.default_rng(seed + 7)
    grain = rng.normal(0, 3, size=(PANEL_H, PANEL_W, 1))
    panel = np.clip(panel.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    # finished (edge-banded) front edge along the bottom long side
    cv2.rectangle(panel, (0, PANEL_H - 10), (PANEL_W - 1, PANEL_H - 1), (176, 190, 200), -1)
    if holes_visible:
        for x in (50, PANEL_W - 50):  # cam-lock holes
            for y in (70, PANEL_H - 70):
                cv2.circle(panel, (x, y), 15, (52, 58, 66), -1)
                cv2.circle(panel, (x, y), 15, (110, 120, 130), 2)
        for x in range(170, PANEL_W - 160, 64):  # shelf-pin row
            cv2.circle(panel, (x, 150), 6, (60, 66, 74), -1)
    tx = 40 if tape_left else PANEL_W - 40 - 90
    cv2.rectangle(panel, (tx, PANEL_H - 26), (tx + 90, PANEL_H - 4), (205, 120, 30), -1)  # blue painter's tape
    # paste panel with rotation
    canvas = np.zeros_like(img)
    mask = np.zeros((H, W), np.uint8)
    x0, y0 = (W - PANEL_W) // 2 + dx, (H - PANEL_H) // 2 + dy
    canvas[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = panel
    mask[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = 255
    M = cv2.getRotationMatrix2D((W / 2 + dx, H / 2 + dy), angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H))
    mask = cv2.warpAffine(mask, M, (W, H))
    img[mask > 127] = canvas[mask > 127]
    # soft drop shadow edge + lighting gradient
    grad = np.linspace(0.92, 1.05, W, dtype=np.float32)[None, :, None]
    img = np.clip(img.astype(np.float32) * grad * gain, 0, 255).astype(np.uint8)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "reference_step3.png"), render(holes_visible=True, tape_left=True, seed=1))
    cv2.imwrite(str(OUT / "checkpoint_pass_step3.png"),
                render(holes_visible=True, tape_left=True, dx=14, dy=-9, angle=1.5, gain=0.96, seed=2))
    cv2.imwrite(str(OUT / "checkpoint_fail_holes_out.png"),
                render(holes_visible=False, tape_left=False, dx=-11, dy=6, angle=-1.2, gain=1.02, seed=3))
    blur = render(holes_visible=True, tape_left=True, dx=5, seed=4, gain=0.35)
    k = np.zeros((1, 61), np.float32); k[0, :] = 1 / 61
    cv2.imwrite(str(OUT / "checkpoint_unmeasurable_blur.png"), cv2.filter2D(blur, -1, k))
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
