#!/usr/bin/env python3
"""Generate Phase 2 Printable AprilTag Calibration Mat + 1:1 Cutouts + Synthetic Golden Rig Dataset.

Outputs:
1. /workspace/rewind/fixtures/printable/:
   - rewind_apriltag_mat_400x250mm.png (High-res 8 px/mm = 3200x2000 px)
   - rewind_apriltag_mat_400x250mm.svg (Exact 400x250mm vector SVG)
   - rewind_apriltag_mat_400x250mm.pdf (Exact 1:1 400x250mm printable PDF)
   - rewind_apriltag_mat_letter_tiles.pdf (3-page US Letter tiled 1:1 printable PDF with alignment crosshairs)
   - rewind_slider_crank_cutouts_1to1.pdf (Exact 1:1 printable cutout templates for all 6 slider-crank parts)
2. /workspace/rewind/fixtures/golden_rig/:
   - step_0_complete.png .. step_4_base.png (1280x800 perspective-warped overhead camera frames)
   - checkpoint_pass_step2.png (Yellow connecting rod properly installed & witness stripe aligned)
   - checkpoint_fail_flipped_rod.png (Yellow connecting rod flipped 180 deg: IoU >= 0.82 but witness stripe misaligned)
   - checkpoint_fail_camera_bump.png (Camera/mat shifted causing registration_rmse_px > 1.5)
   - flat_scan_mat_z0.png (Yellow connecting rod laid flat inside mat_z0 scan zone for px->mm rectification)
   - ref_mask_step2_connecting_rod.png (Binary reference segmentation mask in 1280x800 camera frame)
   - teardown_clip.mp4 (H.264 MP4 video of the slider-crank teardown)
   - golden_metadata.json (Ground-truth homography, tag coordinates, witness/pivot locations, and part specs)
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas

ROOT = Path("/workspace/rewind")
PRINTABLE_DIR = ROOT / "fixtures" / "printable"
GOLDEN_DIR = ROOT / "fixtures" / "golden_rig"

# Mat dimensions in mm
MAT_W_MM = 400.0
MAT_H_MM = 250.0
MAT_SCALE_PX_PER_MM = 8.0  # 3200 x 2000 high-res master texture
TAG_SIDE_MM = 36.0

# Exact tag placements on mat_z0 (in mm, origin at top-left of mat, +X right, +Y down)
# Each entry gives (x_min_mm, y_min_mm, x_max_mm, y_max_mm)
APRILTAG_LAYOUT_MM: Dict[int, Tuple[float, float, float, float]] = {
    0: (12.0, 12.0, 48.0, 48.0),      # Top-left
    1: (352.0, 12.0, 388.0, 48.0),    # Top-right
    2: (352.0, 202.0, 388.0, 238.0),  # Bottom-right
    3: (12.0, 202.0, 48.0, 238.0),    # Bottom-left
}

# Slider-crank assembly key geometry in mat_z0 mm coordinates (upper zone)
CRANK_CENTER_MM = (85.0, 95.0)
CRANK_PIN_MM = (115.0, 95.0)      # Left hole of connecting rod in step 2
WRIST_PIN_MM = (265.0, 95.0)      # Right hole of connecting rod in step 2 (150.0 mm apart!)
ROD_HOLE_DIST_MM = 150.0
ROD_WIDTH_MM = 25.0
ROD_THICKNESS_MM = 4.0
ROD_HOLE_DIA_MM = 8.0
ROD_END_RADIUS_MM = 12.5

# Witness stripe offset from left hole (CRANK_PIN_MM) along rod centerline (+24.0 mm from left hole)
WITNESS_OFFSET_FROM_A_MM = 24.0

# Flat-part scan zone on mat_z0 (lower zone)
SCAN_ZONE_MM = (68.0, 152.0, 332.0, 238.0)  # x1, y1, x2, y2
FLAT_ROD_HOLE_A_MM = (125.0, 195.0)
FLAT_ROD_HOLE_B_MM = (275.0, 195.0)  # Exact 150.0 mm center-to-center inside scan zone

# Camera frame dimensions
CAM_W_PX = 1280
CAM_H_PX = 800


def _mm_to_px(x_mm: float, y_mm: float, scale: float = MAT_SCALE_PX_PER_MM) -> Tuple[int, int]:
    return int(round(x_mm * scale)), int(round(y_mm * scale))



CANONICAL_TAG36H11_HEX: Dict[int, int] = {
    0: 0xD5D628584,
    1: 0xD97F18B49,
    2: 0xDD280910E,
    3: 0xE479E9C98,
}


def render_canonical_apriltag_36h11(tag_id: int, side_px: int = 120) -> np.ndarray:
    """
    Renders a canonical APRILTAG_36h11 marker image with a 1-cell white quiet zone,
    1-cell black border, and 6x6 data payload (total 8x8 cells).
    """
    code = CANONICAL_TAG36H11_HEX[tag_id]
    grid = np.ones((8, 8), dtype=np.uint8) * 255
    grid[0, :] = 0
    grid[7, :] = 0
    grid[:, 0] = 0
    grid[:, 7] = 0
    for bit_idx in range(36):
        row = 1 + (bit_idx // 6)
        col = 1 + (bit_idx % 6)
        bit_val = (code >> (35 - bit_idx)) & 1
        grid[row, col] = 255 if bit_val == 1 else 0
    return cv2.resize(grid, (side_px, side_px), interpolation=cv2.INTER_NEAREST)


def generate_apriltag_marker(tag_id: int, size_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, size_px)
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def render_base_mat_image(scale: float = MAT_SCALE_PX_PER_MM) -> np.ndarray:
    """Render the 400x250mm calibration mat at `scale` px/mm."""
    w_px = int(round(MAT_W_MM * scale))
    h_px = int(round(MAT_H_MM * scale))
    mat = np.full((h_px, w_px, 3), 248, dtype=np.uint8)

    # Outer border
    cv2.rectangle(mat, (0, 0), (w_px - 1, h_px - 1), (40, 40, 40), max(2, int(scale * 0.8)))

    # Subtle 10mm engineering grid
    for x_mm in range(10, int(MAT_W_MM), 10):
        x_px = int(round(x_mm * scale))
        color = (222, 226, 230) if x_mm % 50 != 0 else (195, 202, 210)
        thickness = 1 if x_mm % 50 != 0 else max(1, int(scale * 0.25))
        cv2.line(mat, (x_px, 0), (x_px, h_px), color, thickness)

    for y_mm in range(10, int(MAT_H_MM), 10):
        y_px = int(round(y_mm * scale))
        color = (222, 226, 230) if y_mm % 50 != 0 else (195, 202, 210)
        thickness = 1 if y_mm % 50 != 0 else max(1, int(scale * 0.25))
        cv2.line(mat, (0, y_px), (w_px, y_px), color, thickness)

    # Millimeter ruler ticks along top and left edges
    for x_mm in range(0, int(MAT_W_MM) + 1, 2):
        x_px = int(round(x_mm * scale))
        tick_h_mm = 4.5 if x_mm % 10 == 0 else (3.0 if x_mm % 5 == 0 else 1.8)
        cv2.line(mat, (x_px, 0), (x_px, int(round(tick_h_mm * scale))), (50, 50, 50), 1)
        if x_mm % 50 == 0 and 50 <= x_mm <= 350:
            cv2.putText(
                mat,
                f"{x_mm}mm",
                (x_px - int(8 * scale), int(round(8.5 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.045 * scale,
                (40, 40, 40),
                max(1, int(0.18 * scale)),
                cv2.LINE_AA,
            )

    for y_mm in range(0, int(MAT_H_MM) + 1, 2):
        y_px = int(round(y_mm * scale))
        tick_w_mm = 4.5 if y_mm % 10 == 0 else (3.0 if y_mm % 5 == 0 else 1.8)
        cv2.line(mat, (0, y_px), (int(round(tick_w_mm * scale)), y_px), (50, 50, 50), 1)

    # Upper zone: Fixed Rig Mount Footprints (z = elevated rig)
    rig_x1, rig_y1 = _mm_to_px(56.0, 48.0, scale)
    rig_x2, rig_y2 = _mm_to_px(344.0, 142.0, scale)
    cv2.rectangle(mat, (rig_x1, rig_y1), (rig_x2, rig_y2), (120, 130, 140), max(1, int(0.3 * scale)))
    cv2.putText(
        mat,
        "SLIDER-CRANK RIG MOUNT ZONE (ELEVATED PIXEL CHECKPOINT SPACE)",
        _mm_to_px(62.0, 55.0, scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.042 * scale,
        (70, 80, 90),
        max(1, int(0.18 * scale)),
        cv2.LINE_AA,
    )

    # Footprint outlines for crank_support and linear_guide
    cs_x1, cs_y1 = _mm_to_px(62.0, 66.0, scale)
    cs_x2, cs_y2 = _mm_to_px(110.0, 124.0, scale)
    cv2.rectangle(mat, (cs_x1, cs_y1), (cs_x2, cs_y2), (90, 140, 95), max(1, int(0.3 * scale)))

    lg_x1, lg_y1 = _mm_to_px(170.0, 74.0, scale)
    lg_x2, lg_y2 = _mm_to_px(332.0, 116.0, scale)
    cv2.rectangle(mat, (lg_x1, lg_y1), (lg_x2, lg_y2), (130, 130, 135), max(1, int(0.3 * scale)))

    # Witness reference target marker printed on the mat above crank pin +24mm
    wit_x_mm = CRANK_PIN_MM[0] + WITNESS_OFFSET_FROM_A_MM
    wit_ref_pt1 = _mm_to_px(wit_x_mm, 73.0, scale)
    wit_ref_pt2 = _mm_to_px(wit_x_mm, 81.0, scale)
    cv2.line(mat, wit_ref_pt1, wit_ref_pt2, (20, 20, 20), max(2, int(0.6 * scale)))
    cv2.putText(
        mat,
        "WITNESS REF",
        _mm_to_px(wit_x_mm - 12.0, 70.0, scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.036 * scale,
        (20, 20, 20),
        max(1, int(0.16 * scale)),
        cv2.LINE_AA,
    )

    # Lower zone: Flat-Part Metrology Scan Zone (plane_id: mat_z0)
    # Dark high-contrast slate pad so yellow flat parts & inner holes segment cleanly
    sz_x1, sz_y1 = _mm_to_px(SCAN_ZONE_MM[0], SCAN_ZONE_MM[1], scale)
    sz_x2, sz_y2 = _mm_to_px(SCAN_ZONE_MM[2], SCAN_ZONE_MM[3], scale)
    cv2.rectangle(mat, (sz_x1, sz_y1), (sz_x2, sz_y2), (46, 52, 64), -1)
    cv2.rectangle(mat, (sz_x1, sz_y1), (sz_x2, sz_y2), (25, 30, 38), max(2, int(0.45 * scale)))
    cv2.putText(
        mat,
        "MAT_Z0 FLAT-PART METROLOGY SCAN ZONE (Z = 0.0 MM | CALIBRATED_PLANE)",
        _mm_to_px(SCAN_ZONE_MM[0] + 6.0, SCAN_ZONE_MM[1] + 7.0, scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.040 * scale,
        (215, 222, 232),
        max(1, int(0.16 * scale)),
        cv2.LINE_AA,
    )

    # Render the 4 corner APRILTAG_36h11 markers with white quiet zones
    for tag_id, (x1_mm, y1_mm, x2_mm, y2_mm) in APRILTAG_LAYOUT_MM.items():
        # Quiet zone (4mm margin around tag)
        qx1, qy1 = _mm_to_px(x1_mm - 4.0, y1_mm - 4.0, scale)
        qx2, qy2 = _mm_to_px(x2_mm + 4.0, y2_mm + 4.0, scale)
        cv2.rectangle(mat, (qx1, qy1), (qx2, qy2), (255, 255, 255), -1)

        tx1, ty1 = _mm_to_px(x1_mm, y1_mm, scale)
        tx2, ty2 = _mm_to_px(x2_mm, y2_mm, scale)
        tag_w_px = tx2 - tx1
        tag_h_px = ty2 - ty1
        marker_img = generate_apriltag_marker(tag_id, tag_w_px)
        if marker_img.shape[0] != tag_h_px or marker_img.shape[1] != tag_w_px:
            marker_img = cv2.resize(marker_img, (tag_w_px, tag_h_px), interpolation=cv2.INTER_NEAREST)
        mat[ty1:ty2, tx1:tx2] = marker_img

        label_y = y2_mm + 6.5 if y1_mm < 100 else y1_mm - 5.0
        cv2.putText(
            mat,
            f"TAG36h11 ID:{tag_id}",
            _mm_to_px(x1_mm, label_y, scale),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.036 * scale,
            (30, 30, 30),
            max(1, int(0.16 * scale)),
            cv2.LINE_AA,
        )

    # Title banner at top center
    cv2.putText(
        mat,
        "REWIND CALIBRATION MAT v1.0 | 400.0 x 250.0 mm | APRILTAG_36h11 | PLANE: mat_z0",
        _mm_to_px(64.0, 20.0, scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.048 * scale,
        (25, 25, 25),
        max(1, int(0.22 * scale)),
        cv2.LINE_AA,
    )
    return mat


def _draw_rounded_link(
    canvas: np.ndarray,
    hole_a_mm: Tuple[float, float],
    hole_b_mm: Tuple[float, float],
    width_mm: float,
    hole_dia_mm: float,
    bgr_color: Tuple[int, int, int],
    scale: float,
    draw_witness: bool = True,
    flipped_witness: bool = False,
    hole_fill_bgr: Tuple[int, int, int] = (46, 52, 64),
    draw_pins: bool = False,
) -> None:
    """Draw a rounded-end two-hole link (stadium/capsule shape) with witness mark."""
    ax, ay = _mm_to_px(hole_a_mm[0], hole_a_mm[1], scale)
    bx, by = _mm_to_px(hole_b_mm[0], hole_b_mm[1], scale)
    radius_px = int(round((width_mm / 2.0) * scale))
    hole_r_px = int(round((hole_dia_mm / 2.0) * scale))

    # Draw shadow for depth realism when on elevated rig
    if draw_pins:
        sh_offset = max(2, int(round(1.6 * scale)))
        cv2.line(canvas, (ax + sh_offset, ay + sh_offset), (bx + sh_offset, by + sh_offset), (85, 90, 95), radius_px * 2, cv2.LINE_AA)

    # Stadium outer body (thick line with round caps + end circles)
    cv2.line(canvas, (ax, ay), (bx, by), bgr_color, radius_px * 2, cv2.LINE_AA)
    cv2.circle(canvas, (ax, ay), radius_px, bgr_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), radius_px, bgr_color, -1, cv2.LINE_AA)

    # Crisp dark border around link
    # Approximate stadium contour for outline
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    ux, uy = (dx / length, dy / length) if length > 0 else (1.0, 0.0)
    nx, ny = -uy, ux

    # Witness stripe on top surface of connecting rod
    if draw_witness:
        # When properly oriented, witness stripe is at +24mm from Hole A (left hole).
        # When flipped 180 degrees, witness stripe is at -24mm from Hole B (right end)!
        if not flipped_witness:
            wx_mm = hole_a_mm[0] + WITNESS_OFFSET_FROM_A_MM
            wy_mm = hole_a_mm[1]
        else:
            wx_mm = hole_b_mm[0] - WITNESS_OFFSET_FROM_A_MM
            wy_mm = hole_b_mm[1]
        wx, wy = _mm_to_px(wx_mm, wy_mm, scale)
        half_w_px = int(round(3.0 * scale))
        half_h_px = int(round((width_mm * 0.44) * scale))
        # High-contrast black witness block with white center stripe
        cv2.rectangle(
            canvas,
            (wx - half_w_px, wy - half_h_px),
            (wx + half_w_px, wy + half_h_px),
            (18, 18, 18),
            -1,
        )
        inner_w_px = max(1, int(round(1.1 * scale)))
        cv2.rectangle(
            canvas,
            (wx - inner_w_px, wy - half_h_px),
            (wx + inner_w_px, wy + half_h_px),
            (250, 250, 250),
            -1,
        )

    # Through-holes at both ends
    cv2.circle(canvas, (ax, ay), hole_r_px, hole_fill_bgr, -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), hole_r_px, hole_fill_bgr, -1, cv2.LINE_AA)

    if draw_pins:
        pin_r_px = max(2, int(round(2.8 * scale)))
        cv2.circle(canvas, (ax, ay), pin_r_px, (205, 210, 215), -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), pin_r_px, (205, 210, 215), -1, cv2.LINE_AA)


def render_rig_state_on_mat(
    parts_present: List[str],
    flipped_rod: bool = False,
    flat_scan_rod: bool = False,
    scale: float = MAT_SCALE_PX_PER_MM,
) -> np.ndarray:
    """Render specified slider-crank parts on the 400x250mm mat canvas."""
    canvas = render_base_mat_image(scale=scale)

    # 1. green crank_support & gray linear_guide (Base mounts)
    if "crank_support" in parts_present:
        x1, y1 = _mm_to_px(62.0, 66.0, scale)
        x2, y2 = _mm_to_px(110.0, 124.0, scale)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (75, 165, 80), -1)  # BGR green
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 100, 45), max(2, int(0.4 * scale)))
        cx, cy = _mm_to_px(*CRANK_CENTER_MM, scale)
        cv2.circle(canvas, (cx, cy), int(round(5.0 * scale)), (60, 65, 70), -1, cv2.LINE_AA)

    if "linear_guide" in parts_present:
        x1, y1 = _mm_to_px(170.0, 74.0, scale)
        x2, y2 = _mm_to_px(332.0, 116.0, scale)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (145, 150, 155), -1)  # BGR gray
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 95, 100), max(2, int(0.4 * scale)))
        # Inner slot track
        sx1, sy1 = _mm_to_px(180.0, 89.0, scale)
        sx2, sy2 = _mm_to_px(322.0, 101.0, scale)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (70, 75, 80), -1)

    # 2. red crank_disk
    if "crank_disk" in parts_present:
        cx, cy = _mm_to_px(*CRANK_CENTER_MM, scale)
        r_px = int(round(36.0 * scale))
        cv2.circle(canvas, (cx, cy), r_px, (55, 65, 215), -1, cv2.LINE_AA)  # BGR red
        cv2.circle(canvas, (cx, cy), r_px, (30, 35, 140), max(2, int(0.4 * scale)), cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(round(4.5 * scale)), (210, 215, 220), -1, cv2.LINE_AA)
        px, py = _mm_to_px(*CRANK_PIN_MM, scale)
        cv2.circle(canvas, (px, py), int(round(4.0 * scale)), (45, 50, 55), -1, cv2.LINE_AA)

    # 3. blue slider_block
    if "slider_block" in parts_present:
        bx1, by1 = _mm_to_px(242.0, 78.0, scale)
        bx2, by2 = _mm_to_px(288.0, 112.0, scale)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (215, 115, 45), -1)  # BGR blue
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (140, 70, 20), max(2, int(0.4 * scale)))
        wx, wy = _mm_to_px(*WRIST_PIN_MM, scale)
        cv2.circle(canvas, (wx, wy), int(round(4.0 * scale)), (45, 50, 55), -1, cv2.LINE_AA)

    # 4. yellow connecting_rod (installed on rig)
    if "connecting_rod" in parts_present:
        _draw_rounded_link(
            canvas,
            hole_a_mm=CRANK_PIN_MM,
            hole_b_mm=WRIST_PIN_MM,
            width_mm=ROD_WIDTH_MM,
            hole_dia_mm=ROD_HOLE_DIA_MM,
            bgr_color=(35, 210, 245),  # High-contrast BGR yellow
            scale=scale,
            draw_witness=True,
            flipped_witness=flipped_rod,
            hole_fill_bgr=(50, 55, 60),
            draw_pins=True,
        )

    # 5. purple output_paddle (attached to right end of slider_block)
    if "output_paddle" in parts_present:
        px1, py1 = _mm_to_px(284.0, 82.0, scale)
        px2, py2 = _mm_to_px(326.0, 108.0, scale)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (185, 70, 155), -1)  # BGR purple
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (120, 40, 100), max(2, int(0.4 * scale)))

    # 6. Flat scan mode: removed yellow connecting_rod laid flat on mat_z0 scan zone
    if flat_scan_rod:
        _draw_rounded_link(
            canvas,
            hole_a_mm=FLAT_ROD_HOLE_A_MM,
            hole_b_mm=FLAT_ROD_HOLE_B_MM,
            width_mm=ROD_WIDTH_MM,
            hole_dia_mm=ROD_HOLE_DIA_MM,
            bgr_color=(35, 215, 250),  # Bright yellow on dark slate scan zone
            scale=scale,
            draw_witness=True,
            flipped_witness=False,
            hole_fill_bgr=(46, 52, 64),  # Same dark slate as scan zone background
            draw_pins=False,
        )

    return canvas


def get_camera_perspective_homography(
    scale: float = MAT_SCALE_PX_PER_MM,
    bump_offset_px: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (H_mat_px_to_cam_px, H_cam_px_to_mat_mm) for a realistic angled overhead camera.

    The camera views the 400x250mm mat on a dark workbench with mild perspective foreshortening.
    """
    w_mat_px = MAT_W_MM * scale
    h_mat_px = MAT_H_MM * scale
    src_mat_px = np.array(
        [
            [0.0, 0.0],
            [w_mat_px, 0.0],
            [w_mat_px, h_mat_px],
            [0.0, h_mat_px],
        ],
        dtype=np.float64,
    )
    dx, dy = bump_offset_px
    # Mild perspective trapezoid inside 1280x800 frame
    dst_cam_px = np.array(
        [
            [118.0 + dx, 68.0 + dy],     # Top-left (slightly further from camera)
            [1162.0 + dx, 62.0 - dy * 0.4],   # Top-right
            [1210.0 - dx * 0.5, 736.0 + dy],  # Bottom-right (closer to camera)
            [70.0 + dx, 742.0 - dy * 0.3],    # Bottom-left
        ],
        dtype=np.float64,
    )
    H_mat_px_to_cam_px = cv2.getPerspectiveTransform(src_mat_px.astype(np.float32), dst_cam_px.astype(np.float32))

    src_mat_mm = np.array(
        [
            [0.0, 0.0],
            [MAT_W_MM, 0.0],
            [MAT_W_MM, MAT_H_MM],
            [0.0, MAT_H_MM],
        ],
        dtype=np.float32,
    )
    H_cam_px_to_mat_mm = cv2.getPerspectiveTransform(dst_cam_px.astype(np.float32), src_mat_mm)
    return H_mat_px_to_cam_px, H_cam_px_to_mat_mm


def warp_mat_to_camera_frame(
    mat_img: np.ndarray,
    bump_offset_px: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Warp the high-res mat + parts onto a 1280x800 overhead workbench camera image."""
    # Workbench wood/slate texture background
    bench = np.full((CAM_H_PX, CAM_W_PX, 3), (36, 40, 46), dtype=np.uint8)
    H_mat_px_to_cam_px, _ = get_camera_perspective_homography(
        scale=MAT_SCALE_PX_PER_MM,
        bump_offset_px=bump_offset_px,
    )
    warped = cv2.warpPerspective(
        mat_img,
        H_mat_px_to_cam_px,
        (CAM_W_PX, CAM_H_PX),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_TRANSPARENT,
        dst=bench,
    )
    return warped


def render_connecting_rod_reference_mask_cam_px() -> np.ndarray:
    """Render the exact binary reference mask (0/255) for `connecting_rod` in 1280x800 camera pixels."""
    w_px = int(round(MAT_W_MM * MAT_SCALE_PX_PER_MM))
    h_px = int(round(MAT_H_MM * MAT_SCALE_PX_PER_MM))
    mask_mat = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    _draw_rounded_link(
        mask_mat,
        hole_a_mm=CRANK_PIN_MM,
        hole_b_mm=WRIST_PIN_MM,
        width_mm=ROD_WIDTH_MM,
        hole_dia_mm=ROD_HOLE_DIA_MM,
        bgr_color=(255, 255, 255),
        scale=MAT_SCALE_PX_PER_MM,
        draw_witness=False,
        hole_fill_bgr=(0, 0, 0),
        draw_pins=False,
    )
    H_mat_px_to_cam_px, _ = get_camera_perspective_homography(scale=MAT_SCALE_PX_PER_MM)
    warped = cv2.warpPerspective(
        mask_mat[:, :, 0],
        H_mat_px_to_cam_px,
        (CAM_W_PX, CAM_H_PX),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return np.where(warped > 127, 255, 0).astype(np.uint8)


def save_printable_svg_and_pdfs(mat_img: np.ndarray) -> None:
    """Save 1:1 printable SVG, 400x250mm full-size PDF, US Letter tiled PDF, and 1:1 cutouts PDF."""
    PRINTABLE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PRINTABLE_DIR / "rewind_apriltag_mat_400x250mm.png"
    cv2.imwrite(str(png_path), mat_img)

    # 1. Exact 400x250mm SVG embedding the lossless PNG at exact physical dimensions
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="400mm" height="250mm"
     viewBox="0 0 400 250">
  <title>REWIND Calibration Mat v1.0 (400x250mm, APRILTAG_36h11, plane_id: mat_z0)</title>
  <image x="0" y="0" width="400" height="250"
         preserveAspectRatio="none"
         href="data:image/png;base64,{png_b64}" />
</svg>
"""
    (PRINTABLE_DIR / "rewind_apriltag_mat_400x250mm.svg").write_text(svg_content, encoding="utf-8")

    # 2. Exact 1:1 400x250mm PDF
    pdf_path = PRINTABLE_DIR / "rewind_apriltag_mat_400x250mm.pdf"
    c = pdf_canvas.Canvas(str(pdf_path), pagesize=(400 * mm, 250 * mm))
    c.setTitle("REWIND Printable AprilTag Calibration Mat (400x250mm 1:1)")
    c.drawImage(ImageReader(str(png_path)), 0, 0, width=400 * mm, height=250 * mm)
    c.showPage()
    c.save()

    # 3. US Letter Tiled 1:1 PDF (2 pages landscape 279.4 x 215.9 mm with overlap & registration crosshairs)
    tiled_pdf_path = PRINTABLE_DIR / "rewind_apriltag_mat_letter_tiles.pdf"
    page_w, page_h = letter[1], letter[0]  # Landscape US Letter: 279.4mm x 215.9mm
    ct = pdf_canvas.Canvas(str(tiled_pdf_path), pagesize=(page_w, page_h))
    ct.setTitle("REWIND Calibration Mat - US Letter 1:1 Tiled Print")
    tiles = [
        ("Tile 1 of 2 (Left Half: X=0..220mm) - Print at 100% / Actual Size", 0.0, 220.0),
        ("Tile 2 of 2 (Right Half: X=180..400mm, splice at X=200mm) - Print at 100% / Actual Size", 180.0, 400.0),
    ]
    for title, x_start_mm, x_end_mm in tiles:
        x1_px = int(round(x_start_mm * MAT_SCALE_PX_PER_MM))
        x2_px = int(round(x_end_mm * MAT_SCALE_PX_PER_MM))
        crop = mat_img[:, x1_px:x2_px]
        tmp_crop_path = PRINTABLE_DIR / "_tmp_tile.png"
        cv2.imwrite(str(tmp_crop_path), crop)
        tile_w_mm = x_end_mm - x_start_mm
        # Scale height if needed or place top/bottom with registration mark; here draw 1:1 crop of 220x190mm active area
        ct.setFont("Helvetica-Bold", 9)
        ct.drawString(12 * mm, (page_h - 10 * mm), title)
        ct.drawImage(
            ImageReader(str(tmp_crop_path)),
            12 * mm,
            8 * mm,
            width=tile_w_mm * mm,
            height=195 * mm,
            preserveAspectRatio=True,
        )
        # Registration crosshair at X=200mm splice line
        splice_x_pt = (12.0 + (200.0 - x_start_mm)) * mm
        ct.setStrokeColorRGB(0.85, 0.1, 0.1)
        ct.setDash(3, 3)
        ct.line(splice_x_pt, 5 * mm, splice_x_pt, page_h - 14 * mm)
        ct.showPage()
        if tmp_crop_path.exists():
            tmp_crop_path.unlink()
    ct.save()

    # 4. 1:1 Printable Slider-Crank Cutouts Template PDF
    cutouts_pdf_path = PRINTABLE_DIR / "rewind_slider_crank_cutouts_1to1.pdf"
    cc = pdf_canvas.Canvas(str(cutouts_pdf_path), pagesize=(page_w, page_h))
    cc.setTitle("REWIND 1:1 Printable Slider-Crank Cutouts Template")

    # Header
    cc.setFont("Helvetica-Bold", 12)
    cc.drawString(15 * mm, 200 * mm, "REWIND 1:1 Printable Slider-Crank Cutout Templates (Print at 100% / Actual Size)")
    cc.setFont("Helvetica", 9)
    cc.drawString(
        15 * mm,
        194 * mm,
        "Verify 100.0 mm scale bar below with a ruler before cutting foamboard / cardboard / acrylic stock (4.0 mm thickness).",
    )

    # 100.0 mm calibration scale bar
    cc.setLineWidth(1.5)
    cc.line(15 * mm, 188 * mm, 115 * mm, 188 * mm)
    cc.line(15 * mm, 186 * mm, 15 * mm, 190 * mm)
    cc.line(115 * mm, 186 * mm, 115 * mm, 190 * mm)
    cc.drawString(48 * mm, 189.5 * mm, "100.0 mm EXACT SCALE BAR")

    # Part 1: Yellow Connecting Rod (flat_two_hole_link_v1: 150.0mm hole-to-hole, 25.0mm width, 8.0mm holes, 12.5mm ends)
    rod_ax, rod_ay = 35 * mm, 155 * mm
    rod_bx, rod_by = 185 * mm, 155 * mm
    cc.setFillColorRGB(0.96, 0.84, 0.14)
    cc.setStrokeColorRGB(0.15, 0.15, 0.15)
    cc.roundRect(rod_ax - 12.5 * mm, rod_ay - 12.5 * mm, 175.0 * mm, 25.0 * mm, 12.5 * mm, stroke=1, fill=1)
    # Witness stripe at +24.0mm from left hole
    cc.setFillColorRGB(0.08, 0.08, 0.08)
    cc.rect(rod_ax + 21.0 * mm, rod_ay - 11.0 * mm, 6.0 * mm, 22.0 * mm, stroke=0, fill=1)
    cc.setFillColorRGB(0.98, 0.98, 0.98)
    cc.rect(rod_ax + 23.0 * mm, rod_ay - 11.0 * mm, 2.0 * mm, 22.0 * mm, stroke=0, fill=1)
    # 8.0 mm pin holes
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.circle(rod_ax, rod_ay, 4.0 * mm, stroke=1, fill=1)
    cc.circle(rod_bx, rod_by, 4.0 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(0.1, 0.1, 0.1)
    cc.setFont("Helvetica-Bold", 8)
    cc.drawString(rod_ax + 34 * mm, rod_ay - 2.5 * mm, "3. connecting_rod (YELLOW) | L=150.0mm C-C, W=25.0mm, D=8.0mm")

    # Part 2: Red Crank Disk (D=72.0mm, r=36.0mm, crank throw=30.0mm)
    disk_cx, disk_cy = 55 * mm, 95 * mm
    cc.setFillColorRGB(0.85, 0.25, 0.22)
    cc.circle(disk_cx, disk_cy, 36.0 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.circle(disk_cx, disk_cy, 4.0 * mm, stroke=1, fill=1)
    cc.circle(disk_cx + 30.0 * mm, disk_cy, 4.0 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.drawString(disk_cx - 24 * mm, disk_cy - 18 * mm, "2. crank_disk (RED) R=36mm")

    # Part 3: Green Crank Support (48 x 58 mm)
    cc.setFillColorRGB(0.30, 0.65, 0.32)
    cc.rect(105 * mm, 66 * mm, 48 * mm, 58 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.circle(128 * mm, 95 * mm, 4.0 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(0.1, 0.1, 0.1)
    cc.drawString(107 * mm, 72 * mm, "1. crank_support (GREEN)")

    # Part 4: Blue Slider Block (46 x 34 mm)
    cc.setFillColorRGB(0.18, 0.45, 0.85)
    cc.rect(165 * mm, 78 * mm, 46 * mm, 34 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.circle(188 * mm, 95 * mm, 4.0 * mm, stroke=1, fill=1)
    cc.drawString(167 * mm, 82 * mm, "4. slider_block (BLUE)")

    # Part 5: Purple Output Paddle (42 x 26 mm)
    cc.setFillColorRGB(0.62, 0.28, 0.72)
    cc.rect(220 * mm, 82 * mm, 42 * mm, 26 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.drawString(222 * mm, 93 * mm, "6. output_paddle")

    # Part 6: Gray Linear Guide (162 x 42 mm)
    cc.setFillColorRGB(0.60, 0.62, 0.64)
    cc.rect(35 * mm, 14 * mm, 162 * mm, 42 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(1.0, 1.0, 1.0)
    cc.rect(45 * mm, 29 * mm, 142 * mm, 12 * mm, stroke=1, fill=1)
    cc.setFillColorRGB(0.1, 0.1, 0.1)
    cc.drawString(65 * mm, 18 * mm, "5. linear_guide (GRAY) | 162.0 x 42.0 mm (12.0 mm slot)")

    cc.showPage()
    cc.save()


def generate_golden_rig_dataset() -> Dict[str, object]:
    """Render all golden overhead camera keyframes, checkpoint frames, flat-scan frame, and MP4 teardown clip."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    mat_base = render_base_mat_image(scale=MAT_SCALE_PX_PER_MM)
    save_printable_svg_and_pdfs(mat_base)

    # Teardown keyframes (step_0_complete -> step_4_base)
    teardown_steps = {
        "step_0_complete.png": [
            "crank_support",
            "linear_guide",
            "crank_disk",
            "slider_block",
            "connecting_rod",
            "output_paddle",
        ],
        "step_1_no_paddle.png": [
            "crank_support",
            "linear_guide",
            "crank_disk",
            "slider_block",
            "connecting_rod",
        ],
        "step_2_no_rod.png": [
            "crank_support",
            "linear_guide",
            "crank_disk",
            "slider_block",
        ],
        "step_3_no_slider_or_crank.png": [
            "crank_support",
            "linear_guide",
            "crank_disk",
        ],
        "step_4_base.png": [
            "crank_support",
            "linear_guide",
        ],
    }

    rendered_frames_paths: List[Path] = []
    for filename, parts in teardown_steps.items():
        mat_state = render_rig_state_on_mat(parts_present=parts)
        cam_frame = warp_mat_to_camera_frame(mat_state)
        out_path = GOLDEN_DIR / filename
        cv2.imwrite(str(out_path), cam_frame)
        rendered_frames_paths.append(out_path)

    # Checkpoint 1: Pass Step 2 (yellow connecting rod properly installed & witness stripe aligned)
    pass_parts = ["crank_support", "linear_guide", "crank_disk", "slider_block", "connecting_rod"]
    mat_pass = render_rig_state_on_mat(parts_present=pass_parts, flipped_rod=False)
    cam_pass = warp_mat_to_camera_frame(mat_pass)
    cv2.imwrite(str(GOLDEN_DIR / "checkpoint_pass_step2.png"), cam_pass)

    # Checkpoint 2: Fail Flipped Rod (yellow connecting rod installed 180 deg backwards!)
    mat_flipped = render_rig_state_on_mat(parts_present=pass_parts, flipped_rod=True)
    cam_flipped = warp_mat_to_camera_frame(mat_flipped)
    cv2.imwrite(str(GOLDEN_DIR / "checkpoint_fail_flipped_rod.png"), cam_flipped)

    # Checkpoint 3: Fail Camera Bump (mat/camera shifted by (14.0, -9.0) px so registration_rmse_px > 1.5)
    cam_bumped = warp_mat_to_camera_frame(mat_pass, bump_offset_px=(14.0, -9.0))
    cv2.imwrite(str(GOLDEN_DIR / "checkpoint_fail_camera_bump.png"), cam_bumped)

    # Checkpoint 4: Flat-Part Metrology Scan on mat_z0 (connecting rod laid flat inside scan zone)
    base_plus_flat = render_rig_state_on_mat(
        parts_present=["crank_support", "linear_guide", "crank_disk", "slider_block"],
        flat_scan_rod=True,
    )
    cam_flat_scan = warp_mat_to_camera_frame(base_plus_flat)
    cv2.imwrite(str(GOLDEN_DIR / "flat_scan_mat_z0.png"), cam_flat_scan)

    # Reference binary mask for connecting_rod in camera pixel space (1280x800)
    ref_mask = render_connecting_rod_reference_mask_cam_px()
    cv2.imwrite(str(GOLDEN_DIR / "ref_mask_step2_connecting_rod.png"), ref_mask)

    # Generate H.264 MP4 teardown clip using ffmpeg (2 seconds per keyframe = 10s @ 12 fps)
    mp4_path = GOLDEN_DIR / "teardown_clip.mp4"
    tmp_concat = GOLDEN_DIR / "_ffmpeg_concat.txt"
    lines = []
    for p in rendered_frames_paths:
        lines.append(f"file '{p.name}'")
        lines.append("duration 1.0")
    lines.append(f"file '{rendered_frames_paths[-1].name}'")
    tmp_concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(tmp_concat),
            "-vf",
            "fps=12,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            str(mp4_path),
        ],
        cwd=str(GOLDEN_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tmp_concat.exists():
        tmp_concat.unlink()

    # Compute exact camera-pixel coordinates of pivots & witness stripe via H_mat_px_to_cam_px
    H_mat_px_to_cam_px, H_cam_px_to_mat_mm = get_camera_perspective_homography(scale=MAT_SCALE_PX_PER_MM)

    def _project_mm_to_cam_px(pt_mm: Tuple[float, float]) -> List[float]:
        pt_mat_px = np.array([[[pt_mm[0] * MAT_SCALE_PX_PER_MM, pt_mm[1] * MAT_SCALE_PX_PER_MM]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt_mat_px, H_mat_px_to_cam_px)[0][0]
        return [round(float(out[0]), 3), round(float(out[1]), 3)]

    crank_pin_cam_px = _project_mm_to_cam_px(CRANK_PIN_MM)
    wrist_pin_cam_px = _project_mm_to_cam_px(WRIST_PIN_MM)
    witness_expected_cam_px = _project_mm_to_cam_px((CRANK_PIN_MM[0] + WITNESS_OFFSET_FROM_A_MM, CRANK_PIN_MM[1]))
    witness_flipped_cam_px = _project_mm_to_cam_px((WRIST_PIN_MM[0] - WITNESS_OFFSET_FROM_A_MM, WRIST_PIN_MM[1]))

    # Tag corner ground truth in mm (clockwise from top-left, matching cv2.aruco.ArucoDetector)
    tag_corners_mm: Dict[str, List[List[float]]] = {}
    tag_corners_cam_px: Dict[str, List[List[float]]] = {}
    for tag_id, (x1, y1, x2, y2) in APRILTAG_LAYOUT_MM.items():
        pts_mm = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        tag_corners_mm[str(tag_id)] = pts_mm
        tag_corners_cam_px[str(tag_id)] = [_project_mm_to_cam_px((px, py)) for px, py in pts_mm]

    metadata = {
        "schema_version": "1.0.0",
        "mat_dimensions_mm": {"width_mm": MAT_W_MM, "height_mm": MAT_H_MM},
        "plane_id": "mat_z0",
        "tag_dictionary": "APRILTAG_36h11",
        "printed_tag_side_mm": TAG_SIDE_MM,
        "tag_corners_mm": tag_corners_mm,
        "reference_tag_corners_cam_px": tag_corners_cam_px,
        "camera_resolution_px": {"width_px": CAM_W_PX, "height_px": CAM_H_PX},
        "H_cam_px_to_mat_mm_ground_truth": H_cam_px_to_mat_mm.tolist(),
        "step2_connecting_rod_reference": {
            "part_id": "connecting-rod",
            "step_id": "attach-connecting-rod",
            "crank_pin_cam_px": crank_pin_cam_px,
            "wrist_pin_cam_px": wrist_pin_cam_px,
            "witness_expected_cam_px": witness_expected_cam_px,
            "witness_flipped_cam_px": witness_flipped_cam_px,
            "reference_mask_uri": "fixtures/golden_rig/ref_mask_step2_connecting_rod.png",
        },
        "flat_scan_zone_mm": {
            "x_min_mm": SCAN_ZONE_MM[0],
            "y_min_mm": SCAN_ZONE_MM[1],
            "x_max_mm": SCAN_ZONE_MM[2],
            "y_max_mm": SCAN_ZONE_MM[3],
            "ground_truth_part": {
                "template_id": "flat_two_hole_link_v1",
                "hole_center_distance_mm": ROD_HOLE_DIST_MM,
                "outer_width_mm": ROD_WIDTH_MM,
                "thickness_mm": ROD_THICKNESS_MM,
                "hole_diameter_a_mm": ROD_HOLE_DIA_MM,
                "hole_diameter_b_mm": ROD_HOLE_DIA_MM,
                "end_radius_a_mm": ROD_END_RADIUS_MM,
                "end_radius_b_mm": ROD_END_RADIUS_MM,
            },
        },
    }
    meta_path = GOLDEN_DIR / "golden_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = generate_golden_rig_dataset()
    print("Generated printable mat + cutouts + golden rig dataset successfully.")
    print("Crank pin cam px:", meta["step2_connecting_rod_reference"]["crank_pin_cam_px"])
    print("Wrist pin cam px:", meta["step2_connecting_rod_reference"]["wrist_pin_cam_px"])
    print("Witness expected cam px:", meta["step2_connecting_rod_reference"]["witness_expected_cam_px"])
    print("Witness flipped cam px:", meta["step2_connecting_rod_reference"]["witness_flipped_cam_px"])
