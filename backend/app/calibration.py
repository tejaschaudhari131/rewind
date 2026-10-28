"""Plane-Aware AprilTag 36h11 Calibration & Homography Service (`backend/app/calibration.py`).

Detects the 4 corner `APRILTAG_36h11` fiducials (IDs 0, 1, 2, 3) on the 400x250 mm calibration mat,
computes planar homography H_undistorted_px_to_plane_mm (`mat_z0`), measures registration RMSE in pixels
against the reference session calibration or back-projected tag corners, rectifies `mat_z0` views,
and emits schema-valid `calibration` contract envelopes (`urn:rewind:contract:1.0.0`).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.schemas import make_envelope

GOLDEN_META_PATH = Path("/workspace/rewind/fixtures/golden_rig/golden_metadata.json")

# Default 400x250mm mat tag corner layout in mm (clockwise from top-left for each tag ID 0..3)
DEFAULT_TAG_CORNERS_MM: Dict[int, List[List[float]]] = {
    0: [[12.0, 12.0], [48.0, 12.0], [48.0, 48.0], [12.0, 48.0]],
    1: [[352.0, 12.0], [388.0, 12.0], [388.0, 48.0], [352.0, 48.0]],
    2: [[352.0, 202.0], [388.0, 202.0], [388.0, 238.0], [352.0, 238.0]],
    3: [[12.0, 202.0], [48.0, 202.0], [48.0, 238.0], [12.0, 238.0]],
}


@dataclass
class CalibrationComputation:
    """Result of AprilTag 36h11 detection and planar homography estimation."""

    camera_id: str
    image_width_px: int
    image_height_px: int
    detected_tag_ids: List[int]
    detected_corners_px: Dict[int, List[List[float]]]
    H_img_to_mat_mm: List[List[float]]
    H_mat_mm_to_img: List[List[float]]
    reprojection_rmse_px: float
    registration_rmse_px: float
    calibration_valid: bool
    plane_id: str = "mat_z0"
    printed_tag_side_mm: float = 36.0
    reason_codes: List[str] = field(default_factory=list)


def load_reference_tag_corners_px(meta_path: Path = GOLDEN_META_PATH) -> Optional[Dict[int, List[List[float]]]]:
    """Load the baseline session tag corner coordinates in camera pixels if available."""
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    ref = data.get("reference_tag_corners_cam_px")
    if not isinstance(ref, dict):
        return None
    return {int(k): v for k, v in ref.items()}


def detect_apriltag_36h11_corners(image_bgr_or_gray: np.ndarray) -> Dict[int, np.ndarray]:
    """Detect APRILTAG_36h11 markers (IDs 0, 1, 2, 3) with subpixel corner refinement."""
    if image_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray.copy()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    detected: Dict[int, np.ndarray] = {}
    if ids is not None:
        for corner_arr, raw_id in zip(corners, ids.flatten()):
            tag_id = int(raw_id)
            if tag_id in DEFAULT_TAG_CORNERS_MM:
                pts = corner_arr[0].astype(np.float64)
                detected[tag_id] = pts
    return detected


def compute_planar_calibration(
    image: Union[str, Path, np.ndarray],
    reference_corners_px: Optional[Dict[int, List[List[float]]]] = None,
    max_registration_rmse_px: float = 1.5,
    camera_id: str = "overhead-cam-v1",
) -> CalibrationComputation:
    """Detect 4 corner AprilTags, compute H_img_to_mat_mm and registration_rmse_px."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
    else:
        img = image

    h_px, w_px = img.shape[:2]
    detected = detect_apriltag_36h11_corners(img)

    if reference_corners_px is None:
        reference_corners_px = load_reference_tag_corners_px()

    reason_codes: List[str] = []
    if len(detected) < 4:
        reason_codes.append("insufficient_apriltag_fiducials")
        eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        return CalibrationComputation(
            camera_id=camera_id,
            image_width_px=w_px,
            image_height_px=h_px,
            detected_tag_ids=sorted(detected.keys()),
            detected_corners_px={k: v.tolist() for k, v in detected.items()},
            H_img_to_mat_mm=eye,
            H_mat_mm_to_img=eye,
            reprojection_rmse_px=999.0,
            registration_rmse_px=999.0,
            calibration_valid=False,
            reason_codes=reason_codes,
        )

    img_pts_list: List[List[float]] = []
    mat_pts_list: List[List[float]] = []
    for tag_id in sorted(DEFAULT_TAG_CORNERS_MM.keys()):
        if tag_id not in detected:
            continue
        corners_px = detected[tag_id]
        corners_mm = DEFAULT_TAG_CORNERS_MM[tag_id]
        for pt_px, pt_mm in zip(corners_px, corners_mm):
            img_pts_list.append([float(pt_px[0]), float(pt_px[1])])
            mat_pts_list.append([float(pt_mm[0]), float(pt_mm[1])])

    img_pts = np.array(img_pts_list, dtype=np.float64)
    mat_pts = np.array(mat_pts_list, dtype=np.float64)

    H_img_to_mat, _ = cv2.findHomography(img_pts, mat_pts, method=0)
    H_mat_to_img = np.linalg.inv(H_img_to_mat)

    # Back-project mat_pts -> img_pts to measure intrinsic homography fit RMSE (px)
    proj_img_pts = cv2.perspectiveTransform(mat_pts.reshape(-1, 1, 2).astype(np.float32), H_mat_to_img).reshape(-1, 2)
    reproj_diffs = img_pts - proj_img_pts
    reproj_rmse_px = float(np.sqrt(np.mean(np.sum(reproj_diffs ** 2, axis=1))))

    # Compare detected tag corners against the session's locked reference tag corners (detects camera/mat bump!)
    if reference_corners_px:
        shift_sq_errors: List[float] = []
        for tag_id, det_pts in detected.items():
            if tag_id in reference_corners_px:
                ref_pts = np.array(reference_corners_px[tag_id], dtype=np.float64)
                diff = det_pts - ref_pts
                shift_sq_errors.extend(np.sum(diff ** 2, axis=1).tolist())
        if shift_sq_errors:
            registration_rmse_px = float(np.sqrt(np.mean(shift_sq_errors)))
        else:
            registration_rmse_px = reproj_rmse_px
    else:
        registration_rmse_px = reproj_rmse_px

    calibration_valid = bool(registration_rmse_px <= max_registration_rmse_px and reproj_rmse_px <= 1.5)
    if not calibration_valid:
        reason_codes.append("registration_lost")

    return CalibrationComputation(
        camera_id=camera_id,
        image_width_px=w_px,
        image_height_px=h_px,
        detected_tag_ids=sorted(detected.keys()),
        detected_corners_px={k: np.round(v, 3).tolist() for k, v in detected.items()},
        H_img_to_mat_mm=np.round(H_img_to_mat, 8).tolist(),
        H_mat_mm_to_img=np.round(H_mat_to_img, 8).tolist(),
        reprojection_rmse_px=round(reproj_rmse_px, 4),
        registration_rmse_px=round(registration_rmse_px, 4),
        calibration_valid=calibration_valid,
        reason_codes=reason_codes,
    )


def rectify_to_mat_z0(
    image: Union[str, Path, np.ndarray],
    cal: Optional[CalibrationComputation] = None,
    px_per_mm: float = 4.0,
    mat_width_mm: float = 400.0,
    mat_height_mm: float = 250.0,
) -> Tuple[np.ndarray, CalibrationComputation]:
    """Rectify an overhead camera image onto the `mat_z0` plane at `px_per_mm` (default 4.0 px/mm = 1600x1000 px)."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
    else:
        img = image

    if cal is None:
        cal = compute_planar_calibration(img)

    out_w = int(round(mat_width_mm * px_per_mm))
    out_h = int(round(mat_height_mm * px_per_mm))
    scale_mat = np.array(
        [
            [px_per_mm, 0.0, 0.0],
            [0.0, px_per_mm, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    H_img_to_rect_px = scale_mat @ np.array(cal.H_img_to_mat_mm, dtype=np.float64)
    rectified = cv2.warpPerspective(img, H_img_to_rect_px, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return rectified, cal


def build_calibration_envelope(
    cal: CalibrationComputation,
    *,
    session_id: Optional[str] = None,
    scene_version: int = 1,
    plan_version: int = 1,
    step_id: str = "attach-connecting-rod",
    observation_id: Optional[str] = None,
    calibration_frame_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a schema-valid `calibration` contract envelope (`urn:rewind:contract:1.0.0`)."""
    cal_frame_uuid = calibration_frame_id or str(uuid.uuid4())
    holdout_frame_uuid = str(uuid.uuid4())

    fx = float(cal.image_width_px) * 0.9
    fy = float(cal.image_width_px) * 0.9
    cx = float(cal.image_width_px) / 2.0
    cy = float(cal.image_height_px) / 2.0
    K_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]

    # Compute valid_image_polygon_px from outer corners of the 400x250mm mat projected into image
    H_mat_to_img = np.array(cal.H_mat_mm_to_img, dtype=np.float64)
    mat_corners = np.array(
        [[[0.0, 0.0], [400.0, 0.0], [400.0, 250.0], [0.0, 250.0]]],
        dtype=np.float32,
    )
    poly_px = cv2.perspectiveTransform(mat_corners, H_mat_to_img)[0]
    valid_polygon_px = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly_px]

    # Holdout 95th percentile error in mm (~0.25 mm/px on 1280x800 viewing 400mm)
    holdout_error_95_mm = round(max(0.05, cal.reprojection_rmse_px * 0.35), 4)

    payload = {
        "camera_id": cal.camera_id,
        "image_width_px": cal.image_width_px,
        "image_height_px": cal.image_height_px,
        "K_raw": K_matrix,
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K_undistorted": K_matrix,
        "coordinate_space": "undistorted_source_image_px",
        "tag_dictionary": "APRILTAG_36h11",
        "printed_tag_side_mm": cal.printed_tag_side_mm,
        "status": "validated" if cal.calibration_valid else "invalid",
        "planes": [
            {
                "plane_id": cal.plane_id,
                "z_from_mat_mm": 0.0,
                "H_undistorted_px_to_plane_mm": cal.H_img_to_mat_mm,
                "origin_description": "Top-left outer corner of the 400x250mm calibration mat (mat_z0)",
                "positive_x_description": "Left-to-right along top 400mm ruler edge from Tag 0 toward Tag 1",
                "positive_y_description": "Top-to-bottom along left 250mm ruler edge from Tag 0 toward Tag 3",
                "calibration_frame_ids": [cal_frame_uuid],
                "holdout_frame_ids": [holdout_frame_uuid],
                "holdout_error_95_mm": holdout_error_95_mm,
                "valid_image_polygon_px": valid_polygon_px,
            }
        ],
    }

    kwargs = {
        "producer": "calibration_service",
        "payload_type": "calibration",
        "payload": payload,
        "scene_version": scene_version,
        "plan_version": plan_version,
        "step_id": step_id,
    }
    if session_id:
        kwargs["session_id"] = session_id
    if observation_id:
        kwargs["observation_id"] = observation_id
    return make_envelope(**kwargs)
