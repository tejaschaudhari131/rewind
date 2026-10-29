"""Proposed integration helpers, NOT an applied patch to REWIND.

Projection and deadline helpers are exercised by verify_recommendations.py.
The optional CadQuery path needs an installed CAD kernel and integration testing.
"""
from __future__ import annotations
import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable
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


@dataclass(frozen=True)
class ProviderCompletion:
    status: str
    elapsed_ms: float
    values: tuple[Any, ...] = ()


def _consume_finished_task(task):
    if not task.cancelled():
        task.exception()


async def collect_before_deadline(*jobs: Awaitable[Any], started_at: float,
                                  deadline_seconds: float = 2.8):
    """Bound waiting; elapsed includes earlier capture/local work.

    Jobs MUST be pure producers, never state writers. Timeout/cancellation does
    not authorize late writes. Commit still needs a separate token/epoch CAS.
    Move blocking CPU work off the event loop. Network/browser return latency
    and scheduler stalls cannot be hard-bounded by this coroutine.
    """
    loop = asyncio.get_running_loop()
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        remaining = max(0.0, started_at + deadline_seconds - loop.time())
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        elapsed = (loop.time() - started_at) * 1000
        if pending or elapsed > deadline_seconds * 1000:
            return ProviderCompletion('unknown_timeout', elapsed)
        try:
            values = tuple(task.result() for task in tasks)
        except Exception:
            return ProviderCompletion('unknown_provider_error', elapsed)
        return ProviderCompletion('complete', elapsed, values)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
            task.add_done_callback(_consume_finished_task)


def export_analytic_link(step_path: str | Path, *, center_distance_mm: float,
                         width_mm: float, thickness_mm: float,
                         pin_diameter_mm: float, diametral_clearance_mm: float,
                         minimum_wall_mm: float):
    """Optional CadQuery implementation. No fake STEP fallback.

    Inputs must be fetched from an eligible immutable measurement by the caller.
    Retain the existing STL-only path when CadQuery is unavailable, explicitly
    reporting STEP unavailable. Minimum wall is a rig policy, not a safety claim.
    """
    import cadquery as cq
    L, W, T = center_distance_mm, width_mm, thickness_mm
    D = pin_diameter_mm + diametral_clearance_mm
    bounds = [(L,40,180),(W,8,40),(T,2,12),(pin_diameter_mm,3,10),(diametral_clearance_mm,.1,1.5)]
    if not all(math.isfinite(v) and low <= v <= high for v,low,high in bounds):
        raise ValueError('Parameters outside supported rig envelope')
    if not math.isfinite(minimum_wall_mm) or minimum_wall_mm <= 0 or (W-D)/2 < minimum_wall_mm:
        raise ValueError('Insufficient wall thickness')
    body = cq.Workplane('XY').center(L/2,0).rect(L,W).extrude(T)
    for x in (0,L):
        body = body.union(cq.Workplane('XY').center(x,0).circle(W/2).extrude(T))
    for x in (0,L):
        bore = cq.Workplane('XY', origin=(0,0,-1)).center(x,0).circle(D/2).extrude(T+2)
        body = body.cut(bore)
    body = body.clean()
    if body.solids().size() != 1 or not body.val().isValid():
        raise ValueError('Kernel rejected generated solid')
    path = Path(step_path)
    cq.exporters.export(body, str(path), exportType='STEP')
    loaded = cq.importers.importStep(str(path))
    if loaded.solids().size() != 1 or not loaded.val().isValid():
        raise ValueError('STEP re-import failed solid validity')
    solid = loaded.val()
    bb = solid.BoundingBox()
    expected_volume = (L*W + math.pi*(W/2)**2 - 2*math.pi*(D/2)**2)*T
    if not np.allclose([bb.xlen,bb.ylen,bb.zlen], [L+W,W,T], atol=1e-5, rtol=0):
        raise ValueError('Re-imported extents differ')
    if not math.isclose(solid.Volume(), expected_volume, rel_tol=1e-7, abs_tol=1e-5):
        raise ValueError('Re-imported volume differs')
    # Integration must additionally inspect cylindrical face radii/axes and the
    # two through-bores at (0,0) and (L,0); volume/extents alone are insufficient.
    return loaded
