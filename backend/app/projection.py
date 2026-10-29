"""Height-plane projection helpers (adopted from external audit 2026-10-29, B2).

A 2-D mat homography (Z=0) cannot measure features on an elevated top face. These helpers
require *measured* intrinsics, distortion and a world-to-camera pose. REWIND does not yet ship
measured intrinsics, so production flat measurements remain mat_z0-only; these functions are
used by tests and are the integration point once a real calibration exists.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def pixels_to_height_plane_mm(pixels_raw, K, distortion, R_world_to_camera,
                              t_world_to_camera_mm, height_mm):
    """Map RAW distorted pixels to world XY on world Z=height_mm.

    Requires measured camera intrinsics/distortion and a physically calibrated
    world-to-camera pose. A 2-D mat homography alone cannot supply these inputs.
    The caller must establish which visible contour lies on this plane.
    """
    p = np.asarray(pixels_raw, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera_mm, dtype=np.float64).reshape(3)
    dist = np.asarray(distortion, dtype=np.float64)
    if not all(np.isfinite(x).all() for x in (p, K, R, t, dist)) or not math.isfinite(height_mm):
        raise ValueError('Non-finite camera/plane input')
    if abs(np.linalg.det(K)) < 1e-12 or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError('Invalid intrinsic matrix or rotation')
    normalized = cv2.undistortPoints(p, K, dist).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
    rays_world = rays_camera @ R
    camera_center = -R.T @ t
    if np.any(np.abs(rays_world[:, 2]) < 1e-9):
        raise ValueError('Ray parallel to plane')
    distance = (float(height_mm) - camera_center[2]) / rays_world[:, 2]
    if np.any(distance <= 0):
        raise ValueError('Plane intersection behind camera')
    world = camera_center + distance[:, None] * rays_world
    return world[:, :2]


def top_plane_homography(K_undistorted, R_world_to_camera,
                         t_world_to_camera_mm, height_mm):
    """Map world XY on Z=h to UNDISTORTED pixels, not raw webcam pixels."""
    K = np.asarray(K_undistorted, dtype=float).reshape(3, 3)
    R = np.asarray(R_world_to_camera, dtype=float).reshape(3, 3)
    t = np.asarray(t_world_to_camera_mm, dtype=float).reshape(3)
    return K @ np.column_stack((R[:, 0], R[:, 1], t + height_mm * R[:, 2]))

