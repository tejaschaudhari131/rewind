"""Real-part witness for a black metal bracket (e.g. Zinus bed-frame leg bracket #7).

DETERMINISTIC_LOCAL, no model calls. The correct face carries a green/teal part-number sticker next to the
threaded insert; the blank face has none. Witness: the largest green-label blob (after scaling the frame to a
fixed width) must be at least `min_label_area_px` and sit on a dark (black-metal) surround. Thresholds are in
the step JSON, tuned on 4 real phone photos (3 correct, 1 blank face); they are not tuned per file.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

DEFAULTS = {"work_width": 1000, "hue_lo": 35, "hue_hi": 100, "sat_min": 70, "val_min": 60,
            "min_label_area_px": 300, "surround_px": 25, "max_surround_mean_v": 140}


def _scaled(img: np.ndarray, width: int) -> np.ndarray:
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def check_part_label_witness(live_bgr: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = {**DEFAULTS, **(params or {})}
    im = _scaled(live_bgr, int(p["work_width"]))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] >= p["hue_lo"]) & (hsv[..., 0] <= p["hue_hi"]) & (hsv[..., 1] > p["sat_min"])
         & (hsv[..., 2] > p["val_min"])).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    out: Dict[str, Any] = {"check": "part_label_face", "producer": "DETERMINISTIC_LOCAL", "measured": True,
                           "label_area_px": 0, "surround_mean_v": None}
    if n <= 1:
        return {**out, "status": "fail", "reason": "part_label_not_visible_blank_face"}
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in st[i])
    out.update(label_area_px=area, label_bbox_px=[x, y, w, h])
    if area < p["min_label_area_px"]:
        return {**out, "status": "fail", "reason": "part_label_not_visible_blank_face"}
    r = int(p["surround_px"])
    ring = np.zeros(g.shape, np.uint8)
    cv2.rectangle(ring, (max(0, x - r), max(0, y - r)), (x + w + r, y + h + r), 1, -1)
    ring[y:y + h, x:x + w] = 0
    sv = float(hsv[..., 2][ring.astype(bool)].mean())
    out["surround_mean_v"] = round(sv, 1)
    if sv > p["max_surround_mean_v"]:
        return {**out, "status": "unknown", "reason": "label_not_on_dark_bracket"}
    return {**out, "status": "pass", "reason": "part_label_visible_on_bracket"}


def bracket_png(img_bgr: np.ndarray, max_side: int = 768) -> Optional[bytes]:
    s = max_side / max(img_bgr.shape[:2])
    im = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    ok, buf = cv2.imencode(".png", im)
    return buf.tobytes() if ok else None
