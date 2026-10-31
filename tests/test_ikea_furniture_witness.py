import json

import cv2
import numpy as np
import pytest

from backend.app.checkpoints import default_checkpoint_set, load_checkpoint_set
from backend.app.geometry import (check_hole_pattern_witness, check_tape_edge_witness, extract_panel_mask,
                                  panel_mask_iou)

CS = default_checkpoint_set()
CFG = CS.step()
REF = cv2.imread(str(CFG.reference_image))


def img(name):
    return cv2.imread(str(CS.scenarios[name]))


def test_loader_reads_step_config():
    assert CFG.sam_concept_prompt == "wooden panel" and CFG.display_label.startswith("Step 3 of 8")
    assert CS.synthetic is True and set(CS.scenarios) >= {"ikea_pass_step3", "ikea_fail_holes_out", "ikea_unmeasurable_blur"}


def test_loader_rejects_missing_concept_prompt(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"steps": [{"step_id": "s", "step_number": 1, "title": "t", "part_label": "p",
                                        "reference_image": "x.png"}]}))
    with pytest.raises(ValueError):
        load_checkpoint_set(p)


def test_pass_frame():
    live = img("ikea_pass_step3")
    assert check_hole_pattern_witness(REF, live, CFG.witness["hole_pattern"])["status"] == "pass"
    assert check_tape_edge_witness(REF, live, CFG.witness["tape_edge"])["status"] == "pass"


def test_holes_out_fails_despite_matching_outline():
    live = img("ikea_fail_holes_out")
    assert panel_mask_iou(REF, live) >= CFG.panel_iou_min  # outline alone would pass
    assert check_hole_pattern_witness(REF, live, CFG.witness["hole_pattern"])["status"] == "fail"
    assert check_tape_edge_witness(REF, live, CFG.witness["tape_edge"])["status"] == "fail"


def test_unmeasurable_is_insufficient_not_pass_or_fail():
    live = img("ikea_unmeasurable_blur")
    assert check_hole_pattern_witness(REF, live, CFG.witness["hole_pattern"])["status"] == "insufficient_evidence"
    assert check_tape_edge_witness(REF, live, CFG.witness["tape_edge"])["status"] == "insufficient_evidence"


def test_no_panel_fails_closed():
    blank = np.full_like(REF, 40)
    assert extract_panel_mask(blank) is None
    assert check_hole_pattern_witness(REF, blank, CFG.witness["hole_pattern"])["status"] == "insufficient_evidence"
    assert check_tape_edge_witness(REF, blank, CFG.witness["tape_edge"])["status"] == "insufficient_evidence"
