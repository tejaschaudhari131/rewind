import cv2
import numpy as np

from backend.app.bracket_witness import check_part_label_witness
from backend.app.checkpoints import load_checkpoint_set


def _bar(label: bool) -> np.ndarray:
    img = np.full((600, 1000, 3), (60, 120, 170), np.uint8)  # wood-ish background (BGR)
    cv2.rectangle(img, (100, 250), (900, 330), (20, 20, 20), -1)  # black bar
    if label:
        cv2.circle(img, (860, 290), 18, (150, 160, 30), -1)  # teal sticker
    return img


def test_label_face_passes():
    assert check_part_label_witness(_bar(True))["status"] == "pass"


def test_blank_face_fails():
    r = check_part_label_witness(_bar(False))
    assert r["status"] == "fail" and r["reason"] == "part_label_not_visible_blank_face"


def test_zinus_step_configured():
    s = load_checkpoint_set().step("step-7-zinus-leg-f")
    assert s.audit_profile == "metal_bracket" and s.sam_concept_prompt == "black metal bar"
