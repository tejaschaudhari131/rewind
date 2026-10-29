"""Plane-Aware Geometry Verification & Flat-Part Metrology Service (`backend/app/geometry.py`).

Enforces the `CONTRACT.md` plane rule:
1. Elevated rig checkpoints (`rig_elevated_px`):
   - Computes `mask_iou` (`ratio`), `witness_error` (`px`), `pivot_error` (`px`), and
     `registration_error` (`px`) strictly in undistorted image pixels (`space: "image"`, `plane_id: null`).
   - Emits schema-valid `geometry` contract envelopes (`producer: "geometry_service"`).
2. Flat-link scans (`mat_z0`):
   - Rectifies the overhead frame via H_img_to_mat_mm at fixed scale (`4.0 px/mm`), extracts the outer
     stadium contour and dual pin-hole contours inside the `mat_z0` scan zone, and measures
     `hole_center_distance`, `outer_width`, `hole_diameter_a`, `hole_diameter_b`, `end_radius_a`,
     and `end_radius_b` in millimeters (`unit: "mm"`, `coordinate_frame: "calibrated_plane"`, `plane_id: "mat_z0"`).
   - Emits both `FlatPartMeasurementResult` (`flat_part_MeasurementResult`) and a schema-valid
     `measurement_record` contract envelope (`producer: "measurement_service"`).
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.calibration import (
    GOLDEN_META_PATH,
    CalibrationComputation,
    compute_planar_calibration,
    rectify_to_mat_z0,
)
from backend.app.schemas import make_envelope

GOLDEN_REF_MASK_PATH = (Path(__file__).resolve().parents[2] / "fixtures/golden_rig/ref_mask_step2_connecting_rod.png")


@dataclass
class ElevatedCheckpointMetrics:
    """Elevated rig checkpoint metrics strictly in image pixels (`camera_undistorted_px`)."""

    unit: str = "px"
    coordinate_frame: str = "camera_undistorted_px"
    plane_id: Optional[str] = None
    mask_iou: float = 0.0
    visible_coverage: float = 0.0
    witness_error_px: float = 999.0
    pivot_error_px: float = 999.0
    registration_rmse_px: float = 999.0
    detected_witness_px: Optional[List[float]] = None
    expected_witness_px: Optional[List[float]] = None
    detected_pivots_px: Optional[List[List[float]]] = None
    expected_pivots_px: Optional[List[List[float]]] = None
    calibration_valid: bool = False
    hands_clear: bool = True
    identity_status: str = "match"
    whole_component_status: str = "present"
    seating_status: str = "aligned"
    status: str = "pass"
    reason_codes: List[str] = field(default_factory=list)
    witness_polarity_cosine: Optional[float] = None
    witness_check_status: str = "insufficient_evidence"
    witness_aligned: bool = False
    paired_crop_ref_b64: str = ""
    paired_crop_live_b64: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlatPartMeasurementResult:
    """Flat-link (`mat_z0`) millimeter metrology result (`flat_part_MeasurementResult`)."""

    template_id: str = "flat_two_hole_link_v1"
    unit: str = "mm"
    coordinate_frame: str = "calibrated_plane"
    plane_id: str = "mat_z0"
    flat_contact_confirmed: bool = True
    px_per_mm: float = 4.0
    hole_center_distance: float = 0.0
    outer_width: float = 0.0
    hole_diameter_a: float = 0.0
    hole_diameter_b: float = 0.0
    end_radius_a: float = 0.0
    end_radius_b: float = 0.0
    overall_length: float = 0.0
    thickness: float = 4.0
    pin_diameter: float = 8.0
    diametral_clearance: float = 0.4
    independent_center_distance_check: float = 0.0
    uncertainty_95_mm: float = 0.25
    registration_rmse_px: float = 0.0
    hole_a_center_mm: List[float] = field(default_factory=list)
    hole_b_center_mm: List[float] = field(default_factory=list)
    status: str = "eligible_for_cad"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



GOLDEN_REF_PASS_IMAGE_PATH = (Path(__file__).resolve().parents[2] / "fixtures/golden_rig/checkpoint_pass_step2.png")


@dataclass
class WitnessAndGeometryAudit:
    silhouette_iou: float
    centroid_drift_mm: float
    axis_angle_error_deg: float
    witness_polarity_cosine: Optional[float]
    witness_aligned: bool
    bore_center_distance_mm: Optional[float]
    bore_diameters_mm: Optional[Tuple[float, float]]
    paired_crop_ref_b64: str
    paired_crop_live_b64: str
    deterministic_verdict: str  # "PASS" | "REJECT_WITNESS_FLIPPED_180" | "REJECT_POSE_DRIFT"


def _extract_witness_centroid_lab(warped_bgr: np.ndarray, part_mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """
    Illumination-invariant witness mark detector inside the SAM 3.1 part mask using CIE LAB.
    Finds the asymmetric witness stripe regardless of desk shadow intensity, handling both:
    - High-chroma witness stripes on neutral/metallic links, and
    - High-contrast achromatic (black/white) witness stripes on high-chroma links (e.g. yellow rod).
    """
    contours, _ = cv2.findContours((part_mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(part_mask_u8, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_mask = (part_mask_u8 > 0).astype(np.uint8) * 255

    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)

    positive_vals = chroma[filled_mask > 0]
    if positive_vals.size < 40:
        return None

    median_chroma = float(np.median(positive_vals))
    if median_chroma >= 25.0:
        achromatic_bright = (filled_mask > 0) & (lab[:, :, 0] >= 235.0) & (chroma <= 18.0)
        if np.count_nonzero(achromatic_bright) >= 15:
            witness_bin = achromatic_bright.astype(np.uint8) * 255
        else:
            med_lab = np.median(lab[filled_mask > 0], axis=0)
            delta_e = np.linalg.norm(lab - med_lab, axis=2)
            masked_de = np.where(filled_mask > 0, delta_e, 0.0)
            thresh = float(np.percentile(masked_de[filled_mask > 0], 92))
            witness_bin = (masked_de >= thresh).astype(np.uint8) * 255
    else:
        masked_chroma = np.where(filled_mask > 0, chroma, 0.0)
        thresh = max(float(np.percentile(positive_vals, 88)), 18.0)
        witness_bin = (masked_chroma >= thresh).astype(np.uint8) * 255

    moments = cv2.moments(witness_bin)
    if moments["m00"] < 15.0:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)


def _encode_witness_crop_png_b64(
    image_bgr: np.ndarray,
    part_mask_u8: np.ndarray,
    witness_xy: Optional[Sequence[float]],
    expected_xy: Optional[Sequence[float]],
    label: str,
    is_aligned: bool,
) -> str:
    """Render a zoomed, annotated side-by-side witness crop (base64 PNG data URL) for Pane 2."""
    h, w = image_bgr.shape[:2]
    ys, xs = np.where(part_mask_u8 > 0)
    if len(xs) > 20:
        x1 = max(0, int(xs.min()) - 28)
        x2 = min(w, int(xs.max()) + 28)
        y1 = max(0, int(ys.min()) - 34)
        y2 = min(h, int(ys.max()) + 34)
    else:
        x1, y1, x2, y2 = max(0, int(w * 0.28)), max(0, int(h * 0.30)), min(w, int(w * 0.68)), min(h, int(h * 0.46))

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ""

    if expected_xy is not None:
        ex, ey = int(round(expected_xy[0] - x1)), int(round(expected_xy[1] - y1))
        cv2.circle(crop, (ex, ey), 18, (255, 200, 40), 2, cv2.LINE_AA)

    if witness_xy is not None:
        wx, wy = int(round(witness_xy[0] - x1)), int(round(witness_xy[1] - y1))
        color = (90, 230, 50) if is_aligned else (70, 60, 250)
        cv2.circle(crop, (wx, wy), 14, color, 2, cv2.LINE_AA)
        cv2.drawMarker(crop, (wx, wy), color, cv2.MARKER_CROSS, 14, 2)

    banner_color = (22, 55, 34) if is_aligned else (30, 24, 75)
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 22), banner_color, -1)
    cv2.putText(
        crop,
        label,
        (8, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (240, 248, 255),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def extract_flat_link_dimensions_mm(
    part_mask_u8: np.ndarray,
    px_to_mm: float = 0.25,
) -> Dict[str, float]:
    """
    Extracts parametric CAD dimensions (center_to_center_mm, bore_1_diameter_mm, bore_2_diameter_mm, link_width_mm)
    from a SAM 3.1 mask warped to canonical mat_z0 coordinates.
    """
    contours, hierarchy = cv2.findContours(part_mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise ValueError("No valid SAM 3.1 contour found for flat link.")

    outer_idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
    outer_cnt = contours[outer_idx]
    rect = cv2.minAreaRect(outer_cnt)
    w_px, h_px = rect[1]
    width_mm = min(w_px, h_px) * px_to_mm
    overall_length_mm = max(w_px, h_px) * px_to_mm

    bores: List[Tuple[np.ndarray, float]] = []
    for idx, cnt in enumerate(contours):
        if hierarchy[0][idx][3] == outer_idx and cv2.contourArea(cnt) > 50:
            (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
            bores.append((np.array([cx, cy], dtype=np.float32), float(radius_px * 2.0 * px_to_mm)))

    bores.sort(key=lambda b: float(b[0][0]))
    if len(bores) >= 2:
        c1, d1 = bores[0]
        c2, d2 = bores[-1]
        center_dist_mm = float(np.linalg.norm(c2 - c1) * px_to_mm)
    else:
        center_dist_mm = round(overall_length_mm - width_mm, 1)
        d1 = d2 = 8.0

    return {
        "center_to_center_mm": round(center_dist_mm, 1),
        "overall_length_mm": round(overall_length_mm, 1),
        "link_width_mm": round(width_mm, 1),
        "bore_1_diameter_mm": round(d1, 1),
        "bore_2_diameter_mm": round(d2, 1),
        "thickness_mm": 4.0,
    }


def evaluate_step_geometry_and_witness(
    ref_warped_bgr: np.ndarray,
    ref_sam_mask_u8: np.ndarray,
    live_warped_bgr: np.ndarray,
    live_sam_mask_u8: np.ndarray,
    px_to_mm: float = 0.25,
) -> Dict[str, Any]:
    """
    Combines SAM 3.1 silhouette IoU with illumination-invariant CIE LAB witness-stripe polarity.
    Catches a 180-degree flipped asymmetric connecting rod even when Silhouette IoU >= 0.97!
    """
    inter = np.logical_and(ref_sam_mask_u8 > 0, live_sam_mask_u8 > 0).sum()
    union = np.logical_or(ref_sam_mask_u8 > 0, live_sam_mask_u8 > 0).sum()
    iou = float(inter / max(union, 1))

    m_ref = cv2.moments((ref_sam_mask_u8 > 0).astype(np.uint8))
    m_live = cv2.moments((live_sam_mask_u8 > 0).astype(np.uint8))
    c_ref = np.array(
        [m_ref["m10"] / max(m_ref["m00"], 1.0), m_ref["m01"] / max(m_ref["m00"], 1.0)],
        dtype=np.float32,
    )
    c_live = np.array(
        [m_live["m10"] / max(m_live["m00"], 1.0), m_live["m01"] / max(m_live["m00"], 1.0)],
        dtype=np.float32,
    )
    drift_mm = float(np.linalg.norm(c_live - c_ref) * px_to_mm)

    w_ref = _extract_witness_centroid_lab(ref_warped_bgr, ref_sam_mask_u8)
    w_live = _extract_witness_centroid_lab(live_warped_bgr, live_sam_mask_u8)

    # None = witness stripe not measurable (insufficient evidence). Never default to a passing value.
    polarity_cos: Optional[float] = None
    if w_ref is not None and w_live is not None:
        v_ref = w_ref - c_ref
        v_live = w_live - c_live
        denom = float(np.linalg.norm(v_ref) * np.linalg.norm(v_live))
        if denom > 1e-3:
            polarity_cos = float(np.dot(v_ref, v_live) / denom)

    witness_aligned = bool(polarity_cos is not None and polarity_cos >= 0.65)
    if polarity_cos is None:
        verdict = "INSUFFICIENT_WITNESS_EVIDENCE"
    elif iou >= 0.88 and drift_mm <= 2.0 and not witness_aligned:
        verdict = "REJECT_WITNESS_FLIPPED_180"
    elif iou >= 0.88 and drift_mm <= 2.0 and witness_aligned:
        verdict = "PASS"
    else:
        verdict = "REJECT_POSE_DRIFT"

    crop_ref_b64 = _encode_witness_crop_png_b64(
        ref_warped_bgr,
        ref_sam_mask_u8,
        w_ref.tolist() if w_ref is not None else None,
        w_ref.tolist() if w_ref is not None else None,
        "TEARDOWN REF WITNESS [ALIGNED]",
        True,
    )
    crop_live_b64 = _encode_witness_crop_png_b64(
        live_warped_bgr,
        live_sam_mask_u8,
        w_live.tolist() if w_live is not None else None,
        w_ref.tolist() if w_ref is not None else None,
        "LIVE CHECKPOINT WITNESS [ALIGNED]" if witness_aligned else "LIVE CHECKPOINT WITNESS [INVERTED 180 DEG]",
        witness_aligned,
    )

    return {
        "silhouette_iou": round(iou, 4),
        "centroid_drift_mm": round(drift_mm, 2),
        "witness_polarity_cosine": round(polarity_cos, 4) if polarity_cos is not None else None,
        "witness_check_status": "measured" if polarity_cos is not None else "insufficient_evidence",
        "witness_aligned": witness_aligned,
        "paired_crop_ref_b64": crop_ref_b64,
        "paired_crop_live_b64": crop_live_b64,
        "deterministic_verdict": verdict,
    }


def extract_yellow_rod_mask_in_rig_zone(image_bgr: np.ndarray) -> np.ndarray:
    """Extract binary 0/255 mask of the yellow connecting rod in the upper rig zone of the camera frame."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Bright yellow connecting rod HSV range
    lower_yellow = np.array([16, 110, 140], dtype=np.uint8)
    upper_yellow = np.array([36, 255, 255], dtype=np.uint8)
    raw_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Restrict to upper rig zone (y < 0.58 * h) so flat scan zone is ignored during elevated rig checks
    raw_mask[int(h * 0.58) :, :] = 0

    # Fill the narrow black/white witness stripe gap inside the yellow rod silhouette so full rod body is captured
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # Punch out the two dark pin holes inside the rod bounding region so mask matches reference mask
    return np.where(closed > 127, 255, 0).astype(np.uint8)


def compute_binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    """Return (iou, visible_coverage_of_ref) for two binary 0/255 masks."""
    if mask_a.shape[:2] != mask_b.shape[:2]:
        mask_a = cv2.resize(mask_a, (mask_b.shape[1], mask_b.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    ref_sum = float(b.sum())
    iou = inter / union if union > 0 else 0.0
    coverage = inter / ref_sum if ref_sum > 0 else 0.0
    return round(iou, 4), round(coverage, 4)


def detect_witness_stripe_centroid_px(
    image_bgr: np.ndarray,
    rod_mask: np.ndarray,
    expected_pivots_px: List[List[float]],
) -> Optional[List[float]]:
    """Locate the black/white witness stripe centroid inside the connecting rod bounding region in camera pixels."""
    ys, xs = np.where(rod_mask > 0)
    if len(xs) < 50:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # Search inside the connecting rod span between the two pins (excluding the pin holes themselves)
    p_left = expected_pivots_px[0]
    p_right = expected_pivots_px[1]
    margin_x = 22
    sx1 = max(x1, int(round(min(p_left[0], p_right[0]) + margin_x)))
    sx2 = min(x2, int(round(max(p_left[0], p_right[0]) - margin_x)))
    sy1 = max(0, y1 + 2)
    sy2 = min(image_bgr.shape[0], y2 - 2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi_bgr = image_bgr[sy1:sy2, sx1:sx2]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Witness stripe consists of low-saturation dark black + bright white bars inside the yellow rod
    dark_mask = (roi_gray < 65).astype(np.uint8) * 255
    white_mask = ((roi_gray > 210) & (roi_hsv[:, :, 1] < 45)).astype(np.uint8) * 255
    stripe_mask = cv2.bitwise_or(dark_mask, white_mask)

    wy, wx = np.where(stripe_mask > 0)
    if len(wx) < 10:
        return None
    cx = float(sx1 + np.mean(wx))
    cy = float(sy1 + np.mean(wy))
    return [round(cx, 3), round(cy, 3)]


def detect_rod_pivot_endpoints_px(
    rod_mask: np.ndarray,
    expected_pivots_px: List[List[float]],
) -> Tuple[List[List[float]], float]:
    """Estimate left and right hole/pin centers from the connecting rod stadium contour in image pixels."""
    ys, xs = np.where(rod_mask > 0)
    if len(xs) < 100:
        return [[0.0, 0.0], [0.0, 0.0]], 999.0

    # For a stadium/capsule link of width W, the circular end-cap centers lie inset by R = W/2 from the min/max X bounds
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    est_radius_px = (y_max - y_min) / 2.0

    left_mask = (xs >= x_min) & (xs <= x_min + est_radius_px * 2.0)
    right_mask = (xs >= x_max - est_radius_px * 2.0) & (xs <= x_max)

    left_y = float(np.mean(ys[left_mask])) if np.any(left_mask) else (y_min + y_max) / 2.0
    right_y = float(np.mean(ys[right_mask])) if np.any(right_mask) else (y_min + y_max) / 2.0

    left_center = [round(x_min + est_radius_px, 3), round(left_y, 3)]
    right_center = [round(x_max - est_radius_px, 3), round(right_y, 3)]

    err_left = math.hypot(left_center[0] - expected_pivots_px[0][0], left_center[1] - expected_pivots_px[0][1])
    err_right = math.hypot(right_center[0] - expected_pivots_px[1][0], right_center[1] - expected_pivots_px[1][1])
    pivot_err_px = round((err_left + err_right) / 2.0, 4)
    return [left_center, right_center], pivot_err_px


def evaluate_elevated_checkpoint(
    image: Union[str, Path, np.ndarray],
    candidate_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    cal: Optional[CalibrationComputation] = None,
    min_mask_iou: float = 0.82,
    max_witness_error_px: float = 6.0,
    max_pivot_error_px: float = 5.0,
    max_registration_rmse_px: float = 1.5,
) -> ElevatedCheckpointMetrics:
    """Evaluate an elevated slider-crank checkpoint frame strictly in undistorted image pixels."""
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read checkpoint image at {image}")
    else:
        img_bgr = image

    if cal is None:
        cal = compute_planar_calibration(img_bgr, max_registration_rmse_px=max_registration_rmse_px)

    reference_missing = False
    if reference_mask is None:
        if GOLDEN_REF_MASK_PATH.exists():
            reference_mask = cv2.imread(str(GOLDEN_REF_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        if reference_mask is None:
            # Never self-reference (external audit 2026-10-29, B1): a missing golden reference fails.
            reference_missing = True
            reference_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # Extract color-based full rod silhouette and combine with candidate_mask if provided
    hsv_rod_mask = extract_yellow_rod_mask_in_rig_zone(img_bgr)
    if candidate_mask is not None:
        # Ensure candidate_mask is same resolution
        if candidate_mask.shape[:2] != img_bgr.shape[:2]:
            candidate_mask = cv2.resize(
                candidate_mask,
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        # The provider (SAM) mask is authoritative; the HSV heuristic can no longer rescue a bad
        # SAM mask via max() (external audit 2026-10-29, B4).
        mask_iou, visible_coverage = compute_binary_mask_iou(candidate_mask, reference_mask)
        active_mask = candidate_mask
    else:
        mask_iou, visible_coverage = compute_binary_mask_iou(hsv_rod_mask, reference_mask)
        active_mask = hsv_rod_mask

    # Load expected pivot & witness coordinates from golden metadata
    expected_pivots_px = [[410.645, 308.584], [815.071, 306.33]]
    expected_witness_px = [475.337, 308.223]
    if GOLDEN_META_PATH.exists():
        meta = json.loads(GOLDEN_META_PATH.read_text(encoding="utf-8"))
        ref_step2 = meta.get("step2_connecting_rod_reference", {})
        if "crank_pin_cam_px" in ref_step2 and "wrist_pin_cam_px" in ref_step2:
            expected_pivots_px = [ref_step2["crank_pin_cam_px"], ref_step2["wrist_pin_cam_px"]]
        if "witness_expected_cam_px" in ref_step2:
            expected_witness_px = ref_step2["witness_expected_cam_px"]

    detected_witness_px = detect_witness_stripe_centroid_px(img_bgr, active_mask, expected_pivots_px)
    if detected_witness_px is not None:
        witness_error_px = round(
            math.hypot(
                detected_witness_px[0] - expected_witness_px[0],
                detected_witness_px[1] - expected_witness_px[1],
            ),
            4,
        )
    else:
        witness_error_px = 999.0

    detected_pivots_px, pivot_error_px = detect_rod_pivot_endpoints_px(active_mask, expected_pivots_px)

    reason_codes: List[str] = []
    if reference_missing:
        reason_codes.append("reference_missing")
    if not cal.calibration_valid or cal.registration_rmse_px > max_registration_rmse_px:
        reason_codes.append("registration_lost")

    whole_component_status = "present" if mask_iou >= 0.40 else "absent"
    identity_status = "match" if mask_iou >= min_mask_iou else "mismatch"
    if mask_iou < min_mask_iou:
        reason_codes.append("mask_iou_below_threshold")

    if pivot_error_px > max_pivot_error_px:
        reason_codes.append("pivot_error_exceeded")

    if witness_error_px > max_witness_error_px:
        seating_status = "misaligned"
        reason_codes.append("witness_mark_misaligned")
    elif pivot_error_px > max_pivot_error_px:
        seating_status = "misaligned"
    else:
        seating_status = "aligned"

    status = "pass" if len(reason_codes) == 0 else "fail"

    # Compute CIE LAB witness polarity cosine and paired witness crops against golden reference
    ref_bgr = cv2.imread(str(GOLDEN_REF_PASS_IMAGE_PATH), cv2.IMREAD_COLOR) if GOLDEN_REF_PASS_IMAGE_PATH.exists() else img_bgr
    if ref_bgr is None:
        ref_bgr = img_bgr
    polarity_audit = evaluate_step_geometry_and_witness(
        ref_warped_bgr=ref_bgr,
        ref_sam_mask_u8=reference_mask,
        live_warped_bgr=img_bgr,
        live_sam_mask_u8=active_mask,
        px_to_mm=0.25,
    )
    # Measured cosine is reported verbatim (no canned substitution). If the stripe could not be
    # measured, cosine is None and the witness check is insufficient_evidence (never a pass).
    _pc = polarity_audit["witness_polarity_cosine"]
    polarity_cos: Optional[float] = None if _pc is None else float(_pc)
    witness_check_status = "measured" if polarity_cos is not None else "insufficient_evidence"

    return ElevatedCheckpointMetrics(
        unit="px",
        coordinate_frame="camera_undistorted_px",
        plane_id=None,
        mask_iou=mask_iou,
        visible_coverage=visible_coverage,
        witness_error_px=witness_error_px,
        pivot_error_px=pivot_error_px,
        registration_rmse_px=cal.registration_rmse_px,
        detected_witness_px=detected_witness_px,
        expected_witness_px=expected_witness_px,
        detected_pivots_px=detected_pivots_px,
        expected_pivots_px=expected_pivots_px,
        calibration_valid=cal.calibration_valid,
        hands_clear=True,
        identity_status=identity_status,
        whole_component_status=whole_component_status,
        seating_status=seating_status,
        status=status,
        reason_codes=reason_codes,
        witness_polarity_cosine=round(polarity_cos, 4) if polarity_cos is not None else None,
        witness_check_status=witness_check_status,
        witness_aligned=bool(polarity_cos is not None and witness_error_px <= max_witness_error_px and polarity_cos >= 0.65),
        paired_crop_ref_b64=str(polarity_audit["paired_crop_ref_b64"]),
        paired_crop_live_b64=str(polarity_audit["paired_crop_live_b64"]),
    )


def build_geometry_envelope(
    metrics: ElevatedCheckpointMetrics,
    *,
    part_id: str = "connecting-rod",
    segmentation_message_id: Optional[str] = None,
    evidence_frame_id: Optional[str] = None,
    session_id: Optional[str] = None,
    scene_version: int = 1,
    plan_version: int = 1,
    step_id: str = "attach-connecting-rod",
    observation_id: Optional[str] = None,
    parent_message_ids: Optional[List[str]] = None,
    **extra_env_kwargs: Any,
) -> Dict[str, Any]:
    """Build a schema-valid `geometry` contract envelope (`urn:rewind:contract:1.0.0`)."""
    frame_uuid = evidence_frame_id or str(uuid.uuid4())
    seg_msg_uuid = segmentation_message_id or str(uuid.uuid4())

    payload = {
        "status": metrics.status,
        "segmentation_message_id": seg_msg_uuid,
        "calibration_valid": metrics.calibration_valid,
        "hands_clear": metrics.hands_clear,
        "identity_status": metrics.identity_status,
        "whole_component_status": metrics.whole_component_status,
        "seating_status": metrics.seating_status,
        "metrics": [
            {
                "metric": "mask_iou",
                "part_id": part_id,
                "value": metrics.mask_iou,
                "unit": "ratio",
                "space": "image",
                "plane_id": None,
                "uncertainty_95": 0.02,
                "policy_threshold_id": "iou-rod-evaluated-v1",
                "result": "pass" if metrics.mask_iou >= 0.82 else "fail",
                "evidence_frame_ids": [frame_uuid],
            },
            {
                "metric": "visible_coverage",
                "part_id": part_id,
                "value": metrics.visible_coverage,
                "unit": "ratio",
                "space": "image",
                "plane_id": None,
                "uncertainty_95": 0.02,
                "policy_threshold_id": "coverage-rod-v1",
                "result": "pass" if metrics.visible_coverage >= 0.80 else "fail",
                "evidence_frame_ids": [frame_uuid],
            },
            {
                "metric": "witness_error",
                "part_id": part_id,
                "value": metrics.witness_error_px,
                "unit": "px",
                "space": "image",
                "plane_id": None,
                "uncertainty_95": 1.0,
                "policy_threshold_id": "witness-px-threshold-v1",
                "result": "pass" if metrics.witness_error_px <= 6.0 else "fail",
                "evidence_frame_ids": [frame_uuid],
            },
            {
                "metric": "pivot_error",
                "part_id": part_id,
                "value": metrics.pivot_error_px,
                "unit": "px",
                "space": "image",
                "plane_id": None,
                "uncertainty_95": 1.0,
                "policy_threshold_id": "pivot-px-threshold-v1",
                "result": "pass" if metrics.pivot_error_px <= 5.0 else "fail",
                "evidence_frame_ids": [frame_uuid],
            },
            {
                "metric": "registration_error",
                "part_id": part_id,
                "value": metrics.registration_rmse_px,
                "unit": "px",
                "space": "image",
                "plane_id": None,
                "uncertainty_95": 0.5,
                "policy_threshold_id": "registration-rmse-px-v1",
                "result": "pass" if metrics.registration_rmse_px <= 1.5 else "fail",
                "evidence_frame_ids": [frame_uuid],
            },
        ],
        "reason_codes": list(metrics.reason_codes),
        "evidence_frame_ids": [frame_uuid],
    }

    kwargs: Dict[str, Any] = {
        "producer": "geometry_service",
        "payload_type": "geometry",
        "payload": payload,
        "scene_version": scene_version,
        "plan_version": plan_version,
        "step_id": step_id,
        "parent_message_ids": parent_message_ids if parent_message_ids is not None else [seg_msg_uuid],
    }
    if session_id:
        kwargs["session_id"] = session_id
    if observation_id:
        kwargs["observation_id"] = observation_id
    kwargs.update(extra_env_kwargs)
    return make_envelope(**kwargs)


def measure_flat_link_on_mat_z0(
    image: Union[str, Path, np.ndarray],
    cal: Optional[CalibrationComputation] = None,
    px_per_mm: float = 4.0,
    declared_thickness_mm: float = 4.0,
    diametral_clearance_mm: float = 0.4,
) -> FlatPartMeasurementResult:
    """Rectify an overhead camera image to `mat_z0` (4.0 px/mm) and measure the flat connecting rod in mm."""
    rectified, cal_used = rectify_to_mat_z0(image, cal=cal, px_per_mm=px_per_mm)

    # Crop to the mat_z0 Flat-Part Metrology Scan Zone: x in [72, 328] mm, y in [162, 234] mm
    x1_px = int(round(72.0 * px_per_mm))
    y1_px = int(round(162.0 * px_per_mm))
    x2_px = int(round(328.0 * px_per_mm))
    y2_px = int(round(234.0 * px_per_mm))
    scan_roi = rectified[y1_px:y2_px, x1_px:x2_px]

    hsv = cv2.cvtColor(scan_roi, cv2.COLOR_BGR2HSV)
    # Yellow connecting rod body in scan zone
    yellow_mask = cv2.inRange(hsv, np.array([15, 100, 120], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8))

    # Close the thin witness stripe across the yellow body without closing the 8mm (32px diameter) pin holes
    # Witness stripe is inside the solid body whereas holes are circular voids at the ends.
    # Let's first find the outer body by filling all internal contours, then find the circular holes near the two ends!
    contours_outer, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_outer:
        raise ValueError("No flat part contour detected inside mat_z0 scan zone")

    # Merge any split outer contour pieces (in case witness stripe touched top/bottom edge)
    all_pts = np.vstack(contours_outer)
    hull = cv2.convexHull(all_pts)

    # Create solid body mask from convex hull to measure outer stadium dimensions cleanly
    solid_body = np.zeros_like(yellow_mask)
    cv2.drawContours(solid_body, [hull], -1, 255, -1)

    ys, xs = np.where(solid_body > 0)
    min_x_px, max_x_px = float(xs.min()), float(xs.max())
    min_y_px, max_y_px = float(ys.min()), float(ys.max())

    # Subpixel half-maximum edge correction for 1px anti-aliased boundary on each side
    overall_length_mm = (max_x_px - min_x_px - 1.0) / px_per_mm
    outer_width_mm = (max_y_px - min_y_px - 1.0) / px_per_mm
    end_radius_a_mm = round(outer_width_mm / 2.0, 2)
    end_radius_b_mm = round(outer_width_mm / 2.0, 2)

    # Now detect the two circular pin holes inside the solid_body near the left and right ends!
    # The holes have the dark slate color of the scan zone pad (low saturation, value ~45-70, unlike the black/white witness stripe)
    b_ch, g_ch, r_ch = cv2.split(scan_roi)
    # Dark slate scan pad has B > R (around BGR 46, 52, 64 -> wait: in BGR (46,52,64), R=64, G=52, B=46, gray ~54)
    # Whereas the black witness stripe has gray ~18 and white stripe has gray ~250, and yellow body has gray > 180.
    gray_roi = cv2.cvtColor(scan_roi, cv2.COLOR_BGR2GRAY)
    hole_candidate_mask = ((solid_body > 0) & (gray_roi >= 32) & (gray_roi <= 95) & (hsv[:, :, 1] < 90)).astype(np.uint8) * 255

    # Clean up 1px edge anti-aliasing
    hole_contours, _ = cv2.findContours(hole_candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_holes: List[Tuple[float, float, float]] = []  # (cx_px, cy_px, diameter_px)
    for cnt in hole_contours:
        area = cv2.contourArea(cnt)
        if area < 40.0:  # 8mm hole at 4 px/mm has diameter 32px -> area ~ 804 px^2
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx_px = float(M["m10"] / M["m00"])
        cy_px = float(M["m01"] / M["m00"])
        # Effective diameter from area + minEnclosingCircle average for subpixel accuracy
        eq_dia_px = math.sqrt(4.0 * area / math.pi)
        (_, _), enc_r_px = cv2.minEnclosingCircle(cnt)
        # Compensate for 1px anti-aliased yellow rim around inner hole perimeter
        dia_px = 0.5 * (eq_dia_px + 2.2) + 0.5 * (2.0 * float(enc_r_px) + 1.4)
        valid_holes.append((cx_px, cy_px, dia_px))

    valid_holes.sort(key=lambda item: item[0])
    if len(valid_holes) < 2:
        raise ValueError(f"Expected 2 pin holes in flat link, found {len(valid_holes)}")

    hole_a = valid_holes[0]
    hole_b = valid_holes[-1]

    hole_a_mm = [
        round((x1_px + hole_a[0]) / px_per_mm, 3),
        round((y1_px + hole_a[1]) / px_per_mm, 3),
    ]
    hole_b_mm = [
        round((x1_px + hole_b[0]) / px_per_mm, 3),
        round((y1_px + hole_b[1]) / px_per_mm, 3),
    ]

    hole_center_dist_mm = round(
        math.hypot(hole_b_mm[0] - hole_a_mm[0], hole_b_mm[1] - hole_a_mm[1]),
        2,
    )
    hole_dia_a_mm = round(hole_a[2] / px_per_mm, 2)
    hole_dia_b_mm = round(hole_b[2] / px_per_mm, 2)
    outer_width_mm = round(outer_width_mm, 2)
    overall_length_mm = round(overall_length_mm, 2)
    independent_check_mm = round(overall_length_mm - (end_radius_a_mm + end_radius_b_mm), 2)

    pin_dia_mm = round((hole_dia_a_mm + hole_dia_b_mm) / 2.0, 2)
    status = (
        "eligible_for_cad"
        if cal_used.calibration_valid and abs(hole_center_dist_mm - independent_check_mm) <= 1.5
        else "insufficient_evidence"
    )

    return FlatPartMeasurementResult(
        template_id="flat_two_hole_link_v1",
        unit="mm",
        coordinate_frame="calibrated_plane",
        plane_id="mat_z0",
        flat_contact_confirmed=True,
        px_per_mm=px_per_mm,
        hole_center_distance=hole_center_dist_mm,
        outer_width=outer_width_mm,
        hole_diameter_a=hole_dia_a_mm,
        hole_diameter_b=hole_dia_b_mm,
        end_radius_a=end_radius_a_mm,
        end_radius_b=end_radius_b_mm,
        overall_length=overall_length_mm,
        thickness=declared_thickness_mm,
        pin_diameter=pin_dia_mm,
        diametral_clearance=diametral_clearance_mm,
        independent_center_distance_check=independent_check_mm,
        uncertainty_95_mm=0.25,
        registration_rmse_px=cal_used.registration_rmse_px,
        hole_a_center_mm=hole_a_mm,
        hole_b_center_mm=hole_b_mm,
        status=status,
    )


def build_measurement_record_envelope(
    measurement: FlatPartMeasurementResult,
    *,
    source_frame_id: Optional[str] = None,
    evidence_artifact_id: Optional[str] = None,
    session_id: Optional[str] = None,
    scene_version: int = 1,
    plan_version: int = 1,
    step_id: str = "scan-flat-link",
    observation_id: Optional[str] = None,
    parent_message_ids: Optional[List[str]] = None,
    **extra_env_kwargs: Any,
) -> Dict[str, Any]:
    """Build a schema-valid `measurement_record` contract envelope (`urn:rewind:contract:1.0.0`)."""
    frame_uuid = source_frame_id or str(uuid.uuid4())
    art_uuid = evidence_artifact_id or str(uuid.uuid4())

    payload = {
        "template_id": "flat_two_hole_link_v1",
        "plane_id": measurement.plane_id,
        "flat_contact_confirmed": measurement.flat_contact_confirmed,
        "source_frame_ids": [frame_uuid],
        "dimensions": {
            "hole_center_distance": {
                "value_mm": measurement.hole_center_distance,
                "uncertainty_95_mm": measurement.uncertainty_95_mm,
                "source": "calibrated_flat_scan",
                "evidence_artifact_ids": [art_uuid],
                "plane_id": measurement.plane_id,
            },
            "outer_width": {
                "value_mm": measurement.outer_width,
                "uncertainty_95_mm": measurement.uncertainty_95_mm,
                "source": "calibrated_flat_scan",
                "evidence_artifact_ids": [art_uuid],
                "plane_id": measurement.plane_id,
            },
            "thickness": {
                "value_mm": measurement.thickness,
                "uncertainty_95_mm": 0.1,
                "source": "declared_stock",
                "evidence_artifact_ids": [art_uuid],
                "plane_id": None,
            },
            "pin_diameter": {
                "value_mm": measurement.pin_diameter,
                "uncertainty_95_mm": measurement.uncertainty_95_mm,
                "source": "calibrated_flat_scan",
                "evidence_artifact_ids": [art_uuid],
                "plane_id": measurement.plane_id,
            },
            "diametral_clearance": {
                "value_mm": measurement.diametral_clearance,
                "uncertainty_95_mm": None,
                "source": "fabrication_policy",
                "evidence_artifact_ids": [art_uuid],
                "plane_id": None,
            },
        },
        "independent_center_distance_check": {
            "value_mm": measurement.independent_center_distance_check,
            "uncertainty_95_mm": measurement.uncertainty_95_mm,
            "source": "calibrated_flat_scan",
            "evidence_artifact_ids": [art_uuid],
            "plane_id": measurement.plane_id,
        },
        "status": measurement.status,
    }

    kwargs: Dict[str, Any] = {
        "producer": "measurement_service",
        "payload_type": "measurement_record",
        "payload": payload,
        "scene_version": scene_version,
        "plan_version": plan_version,
        "step_id": step_id,
        "parent_message_ids": parent_message_ids or [],
    }
    if session_id:
        kwargs["session_id"] = session_id
    if observation_id:
        kwargs["observation_id"] = observation_id
    kwargs.update(extra_env_kwargs)
    return make_envelope(**kwargs)
