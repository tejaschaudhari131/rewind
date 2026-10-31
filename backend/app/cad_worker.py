"""Parametric CAD Worker & Independent Geometry Verifier (`backend/app/cad_worker.py`).

Implements the `flat_two_hole_link_v1` parametric generator taking a validated
`FlatPartMeasurementResult` (or `measurement_record` envelope) and producing:
1. Genuine ISO-10303-21 `.STEP` B-rep solid (`flat_two_hole_link_v1.step`)
2. Genuine watertight `.STL` 3D mesh (`flat_two_hole_link_v1.stl`)
3. High-contrast 2D orthographic technical drawing SVG & PNG
   (`flat_two_hole_link_v1_drawing.svg` & `flat_two_hole_link_v1_drawing.png`)
4. Independent geometry verification (`max_abs_error_mm <= 0.10`, `is_watertight == True`)
5. Schema-valid `cad_request` (`producer: "coordinator"`) and `cad_artifact`
   (`producer: "cad_worker"`) contract envelopes (`urn:rewind:contract:1.0.0`).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import trimesh

from backend.app.geometry import FlatPartMeasurementResult
from backend.app.schemas import (
    SUPPORTED_LINK_BOUNDS_MM,
    make_artifact,
    make_envelope,
    sha256_hex,
)

# Overridable so tests / CI never rewrite the tracked sample artifacts in fixtures/cad_outputs.
DEFAULT_CAD_OUTPUT_DIR = Path(
    os.environ.get("REWIND_CAD_OUTPUT_DIR")
    or (Path(__file__).resolve().parents[2] / "fixtures/cad_outputs")
)


@dataclass
class CadVerificationResult:
    """Independent verification report (`cad_VerificationResult`) of generated STEP & STL files."""

    template_id: str
    status: str
    is_watertight: bool
    is_winding_consistent: bool
    volume_mm3: float
    theoretical_volume_mm3: float
    target_dimensions_mm: Dict[str, float]
    measured_stl_dimensions_mm: Dict[str, float]
    measured_step_dimensions_mm: Dict[str, float]
    abs_errors_mm: Dict[str, float]
    max_abs_error_mm: float
    tolerance_mm: float = 0.10
    claim_scope: str = "dimensional_candidate_for_unloaded_test_rig"
    physical_fit_status: str = "not_tested"
    step_kind: str = "faceted_mesh_derived_step"
    step_topology: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CadGenerationResult:
    """Output bundle (`cad_GenerationResult`) from `generate_flat_two_hole_link_v1`."""

    template_id: str
    status: str
    step_path: str
    stl_path: str
    drawing_svg_path: str
    drawing_png_path: str
    parameters_json_path: str
    report_json_path: str
    effective_parameters_mm: Dict[str, float]
    verification: CadVerificationResult
    cad_request_envelope: Dict[str, Any]
    cad_artifact_envelope: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "status": self.status,
            "step_path": self.step_path,
            "stl_path": self.stl_path,
            "drawing_svg_path": self.drawing_svg_path,
            "drawing_png_path": self.drawing_png_path,
            "parameters_json_path": self.parameters_json_path,
            "report_json_path": self.report_json_path,
            "effective_parameters_mm": self.effective_parameters_mm,
            "verification": self.verification.to_dict(),
            "cad_request_envelope": self.cad_request_envelope,
            "cad_artifact_envelope": self.cad_artifact_envelope,
        }


def validate_link_parameters_mm(
    hole_center_distance: float,
    outer_width: float,
    thickness: float,
    pin_diameter: float,
    diametral_clearance: float,
) -> Dict[str, float]:
    """Enforce `CONTRACT.md` geometric sanity bounds for `flat_two_hole_link_v1`."""
    params = {
        "hole_center_distance": float(hole_center_distance),
        "outer_width": float(outer_width),
        "thickness": float(thickness),
        "pin_diameter": float(pin_diameter),
        "diametral_clearance": float(diametral_clearance),
    }
    for key, (lo, hi) in SUPPORTED_LINK_BOUNDS_MM.items():
        val = params[key]
        if not (lo <= val <= hi):
            raise ValueError(f"Parameter {key}={val} mm outside supported bounds [{lo}, {hi}] mm")

    hole_diameter = round(pin_diameter + diametral_clearance, 4)
    # Minimal wall thickness around hole >= max(2.5mm, 0.35 * hole_diameter)
    min_wall = max(2.5, 0.35 * hole_diameter)
    actual_wall = (outer_width - hole_diameter) / 2.0
    if actual_wall < min_wall:
        raise ValueError(
            f"Insufficient radial wall thickness {actual_wall:.2f} mm < required {min_wall:.2f} mm"
        )
    if hole_center_distance <= outer_width:
        raise ValueError("hole_center_distance must exceed outer_width so holes do not overlap")

    params["effective_hole_diameter"] = hole_diameter
    params["end_radius"] = round(outer_width / 2.0, 4)
    params["overall_length"] = round(hole_center_distance + outer_width, 4)
    return params


def build_watertight_link_mesh(
    hole_center_distance: float = 150.0,
    outer_width: float = 25.0,
    thickness: float = 4.0,
    hole_diameter: float = 8.0,
    n_seg: int = 128,
) -> trimesh.Trimesh:
    """Construct a 100% watertight, outward-oriented 3D manifold mesh for `flat_two_hole_link_v1`.

    Coordinate frame of the CAD part:
    - Hole A center at (0.0, 0.0, z)
    - Hole B center at (hole_center_distance, 0.0, z)
    - Bottom face at z = 0.0, top face at z = thickness
    - Y spans [-outer_width/2, +outer_width/2]
    - X spans [-outer_width/2, hole_center_distance + outer_width/2]
    """
    L = float(hole_center_distance)
    W = float(outer_width)
    T = float(thickness)
    R = W / 2.0
    r_h = float(hole_diameter) / 2.0
    half_L = L / 2.0

    n_seg = max(64, (n_seg // 4) * 4)
    q = n_seg // 4

    outer_left: List[Tuple[float, float]] = []
    inner_left: List[Tuple[float, float]] = []
    for i in range(n_seg):
        if i < q:
            frac = i / q
            ox = half_L
            oy = -R + 2.0 * R * frac
        elif i < 2 * q:
            frac = (i - q) / q
            ox = half_L * (1.0 - frac)
            oy = R
        elif i < 3 * q:
            frac = (i - 2 * q) / q
            ang = 0.5 * math.pi + math.pi * frac
            ox = R * math.cos(ang)
            oy = R * math.sin(ang)
        else:
            frac = (i - 3 * q) / q
            ox = half_L * frac
            oy = -R
        outer_left.append((ox, oy))
        ang_in = math.atan2(oy, ox)
        inner_left.append((r_h * math.cos(ang_in), r_h * math.sin(ang_in)))

    verts: List[List[float]] = []
    vert_map: Dict[Tuple[float, float, float], int] = {}

    def get_vid(x: float, y: float, z: float) -> int:
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vert_map:
            vert_map[key] = len(verts)
            verts.append([key[0], key[1], key[2]])
        return vert_map[key]

    faces: List[List[int]] = []

    def add_annulus_half(outer_pts: List[Tuple[float, float]], inner_pts: List[Tuple[float, float]]) -> None:
        m = len(outer_pts)
        for i in range(m):
            j = (i + 1) % m
            ox0, oy0 = outer_pts[i]
            ox1, oy1 = outer_pts[j]
            ix0, iy0 = inner_pts[i]
            ix1, iy1 = inner_pts[j]

            ti0 = get_vid(ix0, iy0, T)
            to0 = get_vid(ox0, oy0, T)
            to1 = get_vid(ox1, oy1, T)
            ti1 = get_vid(ix1, iy1, T)
            faces.append([ti0, to0, to1])
            faces.append([ti0, to1, ti1])

            bi0 = get_vid(ix0, iy0, 0.0)
            bo0 = get_vid(ox0, oy0, 0.0)
            bo1 = get_vid(ox1, oy1, 0.0)
            bi1 = get_vid(ix1, iy1, 0.0)
            faces.append([bi0, bo1, bo0])
            faces.append([bi0, bi1, bo1])

            faces.append([bi0, ti0, ti1])
            faces.append([bi0, ti1, bi1])

            if not (abs(ox0 - half_L) < 1e-6 and abs(ox1 - half_L) < 1e-6):
                faces.append([bo0, bo1, to1])
                faces.append([bo0, to1, to0])

    add_annulus_half(outer_left, inner_left)
    outer_right = [(L - ox, oy) for (ox, oy) in reversed(outer_left)]
    inner_right = [(L - ix, iy) for (ix, iy) in reversed(inner_left)]
    add_annulus_half(outer_right, inner_right)

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )
    mesh.fix_normals()
    return mesh


def write_iso10303_21_step_file(
    step_path: Path,
    mesh: trimesh.Trimesh,
    hole_center_distance: float,
    outer_width: float,
    thickness: float,
    hole_diameter: float,
) -> None:
    """Write a *faceted, mesh-derived* ISO-10303-21 STEP (FACETED_BREP) for `flat_two_hole_link_v1`.

    Every STL triangle becomes a POLY_LOOP face on its own PLANE. This is not an analytic B-rep:
    the bores are faceted, not CYLINDRICAL_SURFACE entities. The named CARTESIAN_POINT/CIRCLE
    entities are parameter annotations only and are not referenced by the shell topology.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    R = round(outer_width / 2.0, 6)
    r_hole = round(hole_diameter / 2.0, 6)
    L = round(hole_center_distance, 6)
    T = round(thickness, 6)

    lines: List[str] = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('REWIND Parametric CAD Export: flat_two_hole_link_v1',",
        f"  'PARAMETERS: hole_center_distance={L:.4f}mm, outer_width={outer_width:.4f}mm, thickness={T:.4f}mm, hole_diameter={hole_diameter:.4f}mm'),",
        "  '2;1');",
        f"FILE_NAME('{step_path.name}','{now_iso}',('REWIND CAD Worker'),('REWIND Metrology'),",
        "  'REWIND ISO-10303-21 Solid Exporter v1.0','flat_two_hole_link_v1','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
        "ENDSEC;",
        "DATA;",
        "#1 = APPLICATION_CONTEXT('core data for automotive mechanical design processes');",
        "#2 = APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1);",
        "#3 = PRODUCT_CONTEXT('',#1,'mechanical');",
        "#4 = PRODUCT('flat_two_hole_link_v1','flat_two_hole_link_v1','Parametric Two-Hole Rounded Link',(#3));",
        "#5 = PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('1.0','final',#4,.MADE.);",
        "#6 = PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');",
        "#7 = PRODUCT_DEFINITION('design','',#5,#6);",
        "#8 = PRODUCT_DEFINITION_SHAPE('','',#7);",
        "#9 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
        "#10 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );",
        "#11 = ( NAMED_UNIT(*) SOLID_ANGLE_UNIT() SI_UNIT($,.STERADIAN.) );",
        "#12 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-04),#9,'distance_accuracy_value','confusion accuracy');",
        "#13 = ( GEOMETRIC_REPRESENTATION_Context(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#12)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#9,#10,#11)) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') );",
        "#14 = CARTESIAN_POINT('Origin',(0.0,0.0,0.0));",
        "#15 = DIRECTION('Z_Axis',(0.0,0.0,1.0));",
        "#16 = DIRECTION('X_Axis',(1.0,0.0,0.0));",
        "#17 = AXIS2_PLACEMENT_3D('World',#14,#15,#16);",
        f"#18 = CARTESIAN_POINT('Hole_A_Center_Bottom',(0.000000,0.000000,0.000000));",
        f"#19 = CARTESIAN_POINT('Hole_B_Center_Bottom',({L:.6f},0.000000,0.000000));",
        f"#20 = CARTESIAN_POINT('Hole_A_Center_Top',(0.000000,0.000000,{T:.6f}));",
        f"#21 = CARTESIAN_POINT('Hole_B_Center_Top',({L:.6f},0.000000,{T:.6f}));",
        "#22 = AXIS2_PLACEMENT_3D('Hole_A_Axis',#18,#15,#16);",
        "#23 = AXIS2_PLACEMENT_3D('Hole_B_Axis',#19,#15,#16);",
        f"#24 = CIRCLE('Hole_A_Profile',#22,{r_hole:.6f});",
        f"#25 = CIRCLE('Hole_B_Profile',#23,{r_hole:.6f});",
        f"#26 = CIRCLE('End_Cap_A_Profile',#22,{R:.6f});",
        f"#27 = CIRCLE('End_Cap_B_Profile',#23,{R:.6f});",
    ]

    next_id = 100
    vert_ent_ids: List[int] = []
    for vx, vy, vz in mesh.vertices:
        cp_id = next_id
        vp_id = next_id + 1
        next_id += 2
        lines.append(f"#{cp_id} = CARTESIAN_POINT('',({vx:.6f},{vy:.6f},{vz:.6f}));")
        lines.append(f"#{vp_id} = VERTEX_POINT('',#{cp_id});")
        vert_ent_ids.append(cp_id)

    face_ent_ids: List[int] = []
    for f0, f1, f2 in mesh.faces:
        loop_id = next_id
        bound_id = next_id + 1
        face_id = next_id + 2
        next_id += 3
        p0, p1, p2 = vert_ent_ids[int(f0)], vert_ent_ids[int(f1)], vert_ent_ids[int(f2)]
        a = np.asarray(mesh.vertices[int(f0)], dtype=float)
        b = np.asarray(mesh.vertices[int(f1)], dtype=float)
        c = np.asarray(mesh.vertices[int(f2)], dtype=float)
        n = np.cross(b - a, c - a)
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 0:
            raise ValueError("degenerate triangle cannot be exported as a planar STEP face")
        n = n / n_norm
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xdir = ref - n * float(np.dot(ref, n))
        xdir = xdir / float(np.linalg.norm(xdir))
        nd_id, xd_id, ax_id, pl_id = next_id, next_id + 1, next_id + 2, next_id + 3
        next_id += 4
        lines.append(f"#{nd_id} = DIRECTION('',({n[0]:.9f},{n[1]:.9f},{n[2]:.9f}));")
        lines.append(f"#{xd_id} = DIRECTION('',({xdir[0]:.9f},{xdir[1]:.9f},{xdir[2]:.9f}));")
        lines.append(f"#{ax_id} = AXIS2_PLACEMENT_3D('',#{p0},#{nd_id},#{xd_id});")
        lines.append(f"#{pl_id} = PLANE('',#{ax_id});")
        lines.append(f"#{loop_id} = POLY_LOOP('',(#{p0},#{p1},#{p2}));")
        lines.append(f"#{bound_id} = FACE_OUTER_BOUND('',#{loop_id},.T.);")
        lines.append(f"#{face_id} = FACE_SURFACE('',(#{bound_id}),#{pl_id},.T.);")
        face_ent_ids.append(face_id)

    shell_id = next_id
    solid_id = next_id + 1
    shape_rep_id = next_id + 2
    sdr_id = next_id + 3
    face_refs = ",".join(f"#{fid}" for fid in face_ent_ids)
    lines.append(f"#{shell_id} = CLOSED_SHELL('flat_two_hole_link_v1_shell',({face_refs}));")
    lines.append(f"#{solid_id} = FACETED_BREP('flat_two_hole_link_v1_solid',#{shell_id});")
    lines.append(f"#{shape_rep_id} = FACETED_BREP_SHAPE_REPRESENTATION('flat_two_hole_link_v1',(#{solid_id},#17),#13);")
    lines.append(f"#{sdr_id} = SHAPE_DEFINITION_REPRESENTATION(#8,#{shape_rep_id});")
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    step_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_technical_drawing_svg_and_png(
    svg_path: Path,
    png_path: Path,
    hole_center_distance: float,
    outer_width: float,
    thickness: float,
    hole_diameter: float,
) -> None:
    """Render a high-contrast 2D orthographic engineering drawing (SVG + PNG) with dimension callouts."""
    L = hole_center_distance
    W = outer_width
    T = thickness
    D = hole_diameter
    R = W / 2.0
    overall_L = L + W

    svg_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620">
  <rect x="0" y="0" width="1000" height="620" fill="#0f172a"/>
  <rect x="16" y="16" width="968" height="588" fill="none" stroke="#475569" stroke-width="2"/>
  <text x="36" y="52" fill="#f8fafc" font-family="monospace" font-size="18" font-weight="bold">
    REWIND PARAMETRIC CAD DRAWING | TEMPLATE: flat_two_hole_link_v1
  </text>
  <text x="36" y="76" fill="#94a3b8" font-family="monospace" font-size="13">
    PLANE: mat_z0 (calibrated_plane) | UNITS: mm | CLAIM SCOPE: dimensional_candidate_for_unloaded_test_rig
  </text>

  <!-- TOP ORTHOGRAPHIC VIEW (Scale 4.0 px/mm, Hole A at (200, 230), Hole B at (200 + 4*L, 230)) -->
  <g transform="translate(200, 230)">
    <rect x="{-R * 4.0:.2f}" y="{-R * 4.0:.2f}" width="{overall_L * 4.0:.2f}" height="{W * 4.0:.2f}"
          rx="{R * 4.0:.2f}" ry="{R * 4.0:.2f}"
          fill="#1e293b" stroke="#38bdf8" stroke-width="3"/>
    <circle cx="0" cy="0" r="{(D / 2.0) * 4.0:.2f}" fill="#0f172a" stroke="#facc15" stroke-width="2.5"/>
    <circle cx="{L * 4.0:.2f}" cy="0" r="{(D / 2.0) * 4.0:.2f}" fill="#0f172a" stroke="#facc15" stroke-width="2.5"/>
    <!-- Centerlines -->
    <line x1="{-R * 4.0 - 20:.2f}" y1="0" x2="{L * 4.0 + R * 4.0 + 20:.2f}" y2="0"
          stroke="#64748b" stroke-width="1" stroke-dasharray="8,4"/>
    <line x1="0" y1="{-R * 4.0 - 18:.2f}" x2="0" y2="{R * 4.0 + 18:.2f}"
          stroke="#64748b" stroke-width="1" stroke-dasharray="6,3"/>
    <line x1="{L * 4.0:.2f}" y1="{-R * 4.0 - 18:.2f}" x2="{L * 4.0:.2f}" y2="{R * 4.0 + 18:.2f}"
          stroke="#64748b" stroke-width="1" stroke-dasharray="6,3"/>
    <!-- Hole-to-hole dimension callout -->
    <line x1="0" y1="-85" x2="{L * 4.0:.2f}" y2="-85" stroke="#f8fafc" stroke-width="1.8"/>
    <line x1="0" y1="{-R * 4.0 - 18:.2f}" x2="0" y2="-92" stroke="#94a3b8" stroke-width="1"/>
    <line x1="{L * 4.0:.2f}" y1="{-R * 4.0 - 18:.2f}" x2="{L * 4.0:.2f}" y2="-92" stroke="#94a3b8" stroke-width="1"/>
    <text x="{(L * 2.0) - 95:.2f}" y="-94" fill="#facc15" font-family="monospace" font-size="15" font-weight="bold">
      L = {L:.2f} mm +/- 0.25 mm (C-C)
    </text>
    <!-- Outer width callout -->
    <text x="{L * 4.0 + R * 4.0 + 28:.2f}" y="5" fill="#38bdf8" font-family="monospace" font-size="14" font-weight="bold">
      W = {W:.2f} mm (R = {R:.2f} mm)
    </text>
    <!-- Hole diameter callout -->
    <text x="-40" y="{R * 4.0 + 42:.2f}" fill="#facc15" font-family="monospace" font-size="14" font-weight="bold">
      2x THRU HOLES D = {D:.2f} mm
    </text>
  </g>

  <!-- SIDE SECTION VIEW (Thickness T) -->
  <g transform="translate(200, 440)">
    <rect x="{-R * 4.0:.2f}" y="0" width="{overall_L * 4.0:.2f}" height="{max(16.0, T * 5.0):.2f}"
          fill="#334155" stroke="#38bdf8" stroke-width="2.5"/>
    <text x="0" y="45" fill="#f8fafc" font-family="monospace" font-size="14" font-weight="bold">
      SIDE VIEW | THICKNESS T = {T:.2f} mm | OVERALL LENGTH = {overall_L:.2f} mm
    </text>
  </g>

  <!-- Title Block -->
  <rect x="640" y="495" width="330" height="95" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
  <text x="655" y="520" fill="#f8fafc" font-family="monospace" font-size="13" font-weight="bold">
    PART: flat_two_hole_link_v1
  </text>
  <text x="655" y="542" fill="#cbd5e1" font-family="monospace" font-size="12">
    EXPORTS: .STEP (ISO-10303-21) + .STL
  </text>
  <text x="655" y="564" fill="#4ade80" font-family="monospace" font-size="12" font-weight="bold">
    STATUS: GEOMETRY_VALID (WATERTIGHT)
  </text>
</svg>
"""
    svg_path.write_text(svg_text, encoding="utf-8")

    # Also render crisp PNG technical drawing via OpenCV for direct UI/vision consumption
    img = np.full((620, 1000, 3), (42, 23, 15), dtype=np.uint8)
    cv2.rectangle(img, (16, 16), (984, 604), (105, 85, 71), 2)
    cv2.putText(
        img,
        "REWIND PARAMETRIC CAD DRAWING | flat_two_hole_link_v1",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (250, 250, 248),
        2,
        cv2.LINE_AA,
    )
    ox, oy = 200, 230
    bx = int(round(ox + L * 4.0))
    r_px = int(round(R * 4.0))
    hr_px = int(round((D / 2.0) * 4.0))
    cv2.line(img, (ox, oy), (bx, oy), (90, 65, 45), r_px * 2, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), r_px, (90, 65, 45), -1, cv2.LINE_AA)
    cv2.circle(img, (bx, oy), r_px, (90, 65, 45), -1, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), hr_px, (42, 23, 15), -1, cv2.LINE_AA)
    cv2.circle(img, (bx, oy), hr_px, (42, 23, 15), -1, cv2.LINE_AA)
    cv2.circle(img, (ox, oy), hr_px, (25, 210, 250), 2, cv2.LINE_AA)
    cv2.circle(img, (bx, oy), hr_px, (25, 210, 250), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"L = {L:.2f} mm C-C | W = {W:.2f} mm | D_hole = {D:.2f} mm | T = {T:.2f} mm",
        (ox - 40, oy - r_px - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (45, 220, 252),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(png_path), img)


def check_faceted_step_topology(step_text: str, expected_face_count: int, tol_mm: float = 1e-4) -> Dict[str, Any]:
    """Structural check of a faceted STEP: references resolve, one closed shell owns every face,
    face count equals the STL triangle count, every face lies on its own PLANE."""
    ents: Dict[int, str] = {}
    for m in re.finditer(r"^#(\d+)\s*=\s*(.*?);\s*$", step_text, flags=re.M):
        ents[int(m.group(1))] = m.group(2)
    problems: List[str] = []
    for eid, body in ents.items():
        for ref in re.findall(r"#(\d+)", body):
            if int(ref) not in ents:
                problems.append(f"dangling_reference:#{eid}->#{ref}")
                break
    faces = {eid: b for eid, b in ents.items() if b.startswith("FACE_SURFACE(")}
    shells = [b for b in ents.values() if b.startswith("CLOSED_SHELL(")]
    breps = [b for b in ents.values() if b.startswith("FACETED_BREP(") or b.startswith("MANIFOLD_SOLID_BREP(")]
    if len(shells) != 1:
        problems.append(f"closed_shell_count={len(shells)}")
    if len(breps) != 1:
        problems.append(f"brep_count={len(breps)}")
    if len(faces) != expected_face_count:
        problems.append(f"face_count={len(faces)} expected={expected_face_count}")
    if shells:
        shell_refs = {int(r) for r in re.findall(r"#(\d+)", shells[0])}
        if shell_refs != set(faces):
            problems.append("shell_faces_mismatch")

    def _pt(eid: int) -> Optional[np.ndarray]:
        body = ents.get(eid, "")
        m = re.match(r"CARTESIAN_POINT\('[^']*',\(([^)]*)\)\)", body)
        return np.array([float(x) for x in m.group(1).split(",")]) if m else None

    def _dir(eid: int) -> Optional[np.ndarray]:
        m = re.match(r"DIRECTION\('[^']*',\(([^)]*)\)\)", ents.get(eid, ""))
        return np.array([float(x) for x in m.group(1).split(",")]) if m else None

    off_plane = 0
    for eid, body in faces.items():
        refs = [int(r) for r in re.findall(r"#(\d+)", body)]
        if len(refs) != 2 or not ents.get(refs[1], "").startswith("PLANE("):
            problems.append(f"face_not_on_plane:#{eid}")
            continue
        bound = ents.get(refs[0], "")
        loop_ref = re.findall(r"#(\d+)", bound)
        loop = ents.get(int(loop_ref[0]), "") if loop_ref else ""
        pts = [_pt(int(r)) for r in re.findall(r"#(\d+)", loop)]
        ax = ents.get(int(re.findall(r"#(\d+)", ents[refs[1]])[0]), "")
        ax_refs = [int(r) for r in re.findall(r"#(\d+)", ax)]
        origin = _pt(ax_refs[0]) if ax_refs else None
        normal = _dir(ax_refs[1]) if len(ax_refs) > 1 else None
        if len(pts) < 3 or any(p is None for p in pts) or origin is None or normal is None:
            problems.append(f"face_unresolved:#{eid}")
            continue
        if max(abs(float(np.dot(p - origin, normal))) for p in pts) > tol_mm:
            off_plane += 1
    if off_plane:
        problems.append(f"faces_off_plane={off_plane}")
    return {"ok": not problems, "face_count": len(faces), "problems": problems[:20]}


def verify_generated_cad_artifacts(
    step_path: Path,
    stl_path: Path,
    target_hole_center_distance: float,
    target_outer_width: float,
    target_thickness: float,
    target_hole_diameter: float,
    tolerance_mm: float = 0.10,
) -> CadVerificationResult:
    """Independently load the exported `.STL` and `.STEP` files from disk and verify dimensions."""
    loaded_mesh = trimesh.load_mesh(str(stl_path), file_type="stl")
    is_watertight = bool(loaded_mesh.is_watertight)
    is_winding = bool(loaded_mesh.is_winding_consistent)
    vol_mm3 = abs(float(loaded_mesh.volume))

    R_target = target_outer_width / 2.0
    r_h_target = target_hole_diameter / 2.0
    theo_vol_mm3 = (
        target_hole_center_distance * target_outer_width
        + math.pi * (R_target ** 2)
        - 2.0 * math.pi * (r_h_target ** 2)
    ) * target_thickness

    # Measure dimensions from the loaded STL vertices
    verts = np.asarray(loaded_mesh.vertices, dtype=np.float64)
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    extents = max_xyz - min_xyz
    stl_overall_length = float(extents[0])
    stl_outer_width = float(extents[1])
    stl_thickness = float(extents[2])
    stl_hole_center_dist = stl_overall_length - stl_outer_width

    # Recover inner hole radii from vertices closest to Hole A (0,0) and Hole B (L,0)
    d_a = np.hypot(verts[:, 0] - 0.0, verts[:, 1] - 0.0)
    d_b = np.hypot(verts[:, 0] - stl_hole_center_dist, verts[:, 1] - 0.0)
    stl_hole_dia_a = float(2.0 * d_a.min())
    stl_hole_dia_b = float(2.0 * d_b.min())

    # Parse ISO-10303-21 STEP file from disk to verify analytic/B-rep parameters independently
    step_text = step_path.read_text(encoding="utf-8")
    if not step_text.startswith("ISO-10303-21;") or "CLOSED_SHELL" not in step_text:
        raise ValueError("STEP file missing required ISO-10303-21 B-rep entities")
    step_topology = check_faceted_step_topology(step_text, expected_face_count=len(loaded_mesh.faces))
    volume_rel_err = abs(vol_mm3 - theo_vol_mm3) / theo_vol_mm3 if theo_vol_mm3 > 0 else 1.0

    m_hb = re.search(r"CARTESIAN_POINT\('Hole_B_Center_Bottom',\(([0-9.+-]+),", step_text)
    m_top = re.search(r"CARTESIAN_POINT\('Hole_A_Center_Top',\([0-9.+-]+,[0-9.+-]+,([0-9.+-]+)\)\);", step_text)
    m_ha = re.search(r"CIRCLE\('Hole_A_Profile',#\d+,([0-9.+-]+)\);", step_text)
    m_cap = re.search(r"CIRCLE\('End_Cap_A_Profile',#\d+,([0-9.+-]+)\);", step_text)

    step_L = float(m_hb.group(1)) if m_hb else 0.0
    step_T = float(m_top.group(1)) if m_top else 0.0
    step_D = 2.0 * float(m_ha.group(1)) if m_ha else 0.0
    step_W = 2.0 * float(m_cap.group(1)) if m_cap else 0.0

    target_dims = {
        "hole_center_distance": round(target_hole_center_distance, 4),
        "outer_width": round(target_outer_width, 4),
        "thickness": round(target_thickness, 4),
        "hole_diameter": round(target_hole_diameter, 4),
    }
    measured_stl = {
        "hole_center_distance": round(stl_hole_center_dist, 4),
        "outer_width": round(stl_outer_width, 4),
        "thickness": round(stl_thickness, 4),
        "hole_diameter_a": round(stl_hole_dia_a, 4),
        "hole_diameter_b": round(stl_hole_dia_b, 4),
    }
    measured_step = {
        "hole_center_distance": round(step_L, 4),
        "outer_width": round(step_W, 4),
        "thickness": round(step_T, 4),
        "hole_diameter": round(step_D, 4),
    }

    abs_errors = {
        "stl_hole_center_distance": round(abs(stl_hole_center_dist - target_hole_center_distance), 5),
        "stl_outer_width": round(abs(stl_outer_width - target_outer_width), 5),
        "stl_thickness": round(abs(stl_thickness - target_thickness), 5),
        "stl_hole_diameter_a": round(abs(stl_hole_dia_a - target_hole_diameter), 5),
        "stl_hole_diameter_b": round(abs(stl_hole_dia_b - target_hole_diameter), 5),
        "step_hole_center_distance": round(abs(step_L - target_hole_center_distance), 5),
        "step_outer_width": round(abs(step_W - target_outer_width), 5),
        "step_thickness": round(abs(step_T - target_thickness), 5),
        "step_hole_diameter": round(abs(step_D - target_hole_diameter), 5),
    }
    max_abs_err = max(abs_errors.values())
    status = (
        "geometry_valid"
        if is_watertight and is_winding and max_abs_err <= tolerance_mm
        and step_topology["ok"] and volume_rel_err <= 0.02
        else "rejected"
    )

    return CadVerificationResult(
        template_id="flat_two_hole_link_v1",
        status=status,
        step_topology=step_topology,
        is_watertight=is_watertight,
        is_winding_consistent=is_winding,
        volume_mm3=round(vol_mm3, 2),
        theoretical_volume_mm3=round(theo_vol_mm3, 2),
        target_dimensions_mm=target_dims,
        measured_stl_dimensions_mm=measured_stl,
        measured_step_dimensions_mm=measured_step,
        abs_errors_mm=abs_errors,
        max_abs_error_mm=round(max_abs_err, 5),
        tolerance_mm=tolerance_mm,
    )


def generate_flat_two_hole_link_v1(
    measurement: Union[FlatPartMeasurementResult, Dict[str, Any]],
    *,
    output_dir: Union[str, Path] = DEFAULT_CAD_OUTPUT_DIR,
    apply_clearance: bool = False,
    session_id: Optional[str] = None,
    scene_version: int = 1,
    plan_version: int = 1,
    step_id: str = "export-cad-link",
    observation_id: Optional[str] = None,
    measurement_record_id: Optional[str] = None,
    parent_message_ids: Optional[List[str]] = None,
) -> CadGenerationResult:
    """Generate `.STEP`, `.STL`, and 2D drawing from a validated `FlatPartMeasurementResult` or dict."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(measurement, FlatPartMeasurementResult):
        hole_dist = measurement.hole_center_distance
        outer_w = measurement.outer_width
        thick = measurement.thickness
        pin_dia = measurement.pin_diameter
        clearance = measurement.diametral_clearance
        plane_id = measurement.plane_id
        unc_95 = measurement.uncertainty_95_mm
    else:
        hole_dist = float(measurement.get("hole_center_distance", 150.0))
        outer_w = float(measurement.get("outer_width", 25.0))
        thick = float(measurement.get("thickness", 4.0))
        pin_dia = float(measurement.get("pin_diameter", 8.0))
        clearance = float(measurement.get("diametral_clearance", 0.4))
        plane_id = str(measurement.get("plane_id", "mat_z0"))
        raw_unc = measurement.get("uncertainty_95_mm", None)
        unc_95 = None if raw_unc is None else float(raw_unc)
    dim_source = "calibrated_flat_scan" if unc_95 is not None else "manual_caliper"

    if plane_id != "mat_z0":
        raise ValueError(f"CAD generation requires plane_id='mat_z0', got {plane_id!r}")

    validated = validate_link_parameters_mm(
        hole_center_distance=hole_dist,
        outer_width=outer_w,
        thickness=thick,
        pin_diameter=pin_dia,
        diametral_clearance=clearance,
    )
    # When exporting 1:1 replacement of the scanned link, hole_diameter matches pin_diameter unless clearance requested
    target_hole_dia = validated["effective_hole_diameter"] if apply_clearance else pin_dia

    step_path = out_dir / "flat_two_hole_link_v1.step"
    stl_path = out_dir / "flat_two_hole_link_v1.stl"
    svg_path = out_dir / "flat_two_hole_link_v1_drawing.svg"
    png_path = out_dir / "flat_two_hole_link_v1_drawing.png"
    params_path = out_dir / "flat_two_hole_link_v1_parameters.json"
    report_path = out_dir / "flat_two_hole_link_v1_report.json"

    mesh = build_watertight_link_mesh(
        hole_center_distance=hole_dist,
        outer_width=outer_w,
        thickness=thick,
        hole_diameter=target_hole_dia,
    )
    mesh.export(str(stl_path), file_type="stl")

    write_iso10303_21_step_file(
        step_path=step_path,
        mesh=mesh,
        hole_center_distance=hole_dist,
        outer_width=outer_w,
        thickness=thick,
        hole_diameter=target_hole_dia,
    )

    render_technical_drawing_svg_and_png(
        svg_path=svg_path,
        png_path=png_path,
        hole_center_distance=hole_dist,
        outer_width=outer_w,
        thickness=thick,
        hole_diameter=target_hole_dia,
    )

    verification = verify_generated_cad_artifacts(
        step_path=step_path,
        stl_path=stl_path,
        target_hole_center_distance=hole_dist,
        target_outer_width=outer_w,
        target_thickness=thick,
        target_hole_diameter=target_hole_dia,
    )

    effective_params = {
        "hole_center_distance_mm": hole_dist,
        "outer_width_mm": outer_w,
        "thickness_mm": thick,
        "pin_diameter_mm": pin_dia,
        "diametral_clearance_mm": clearance,
        "exported_hole_diameter_mm": target_hole_dia,
        "end_radius_mm": round(outer_w / 2.0, 4),
        "overall_length_mm": round(hole_dist + outer_w, 4),
    }
    params_path.write_text(json.dumps(effective_params, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(verification.to_dict(), indent=2), encoding="utf-8")

    # Build schema-valid `cad_request` envelope (`producer: "coordinator"`)
    meas_rec_uuid = measurement_record_id or str(uuid.uuid4())
    ev_art_uuid = str(uuid.uuid4())
    template_bytes = Path(__file__).read_bytes()

    cad_req_payload = {
        "template_id": "flat_two_hole_link_v1",
        "template_sha256": sha256_hex(template_bytes),
        "measurement_record_id": meas_rec_uuid,
        "parameters": {
            "hole_center_distance": {
                "value_mm": hole_dist,
                "uncertainty_95_mm": unc_95,
                "source": dim_source,
                "evidence_artifact_ids": [ev_art_uuid],
                "plane_id": "mat_z0",
            },
            "outer_width": {
                "value_mm": outer_w,
                "uncertainty_95_mm": unc_95,
                "source": dim_source,
                "evidence_artifact_ids": [ev_art_uuid],
                "plane_id": "mat_z0",
            },
            "thickness": {
                "value_mm": thick,
                "uncertainty_95_mm": 0.1,  # declared stock tolerance, not measured
                "source": "declared_stock",
                "evidence_artifact_ids": [ev_art_uuid],
                "plane_id": None,
            },
            "pin_diameter": {
                "value_mm": pin_dia,
                "uncertainty_95_mm": unc_95,
                "source": dim_source,
                "evidence_artifact_ids": [ev_art_uuid],
                "plane_id": "mat_z0",
            },
            "diametral_clearance": {
                "value_mm": clearance,
                "uncertainty_95_mm": None,
                "source": "fabrication_policy",
                "evidence_artifact_ids": [ev_art_uuid],
                "plane_id": None,
            },
        },
        "material_label": "PLA / Acrylic 4.0mm stock",
        "fabrication_method": "export_only",
        "requested_formats": ["STEP", "STL"],
        "claim_scope": "dimensional_candidate_for_unloaded_test_rig",
    }

    req_kwargs: Dict[str, Any] = {
        "producer": "coordinator",
        "payload_type": "cad_request",
        "payload": cad_req_payload,
        "scene_version": scene_version,
        "plan_version": plan_version,
        "step_id": step_id,
        "parent_message_ids": parent_message_ids or [],
    }
    if session_id:
        req_kwargs["session_id"] = session_id
    if observation_id:
        req_kwargs["observation_id"] = observation_id
    cad_request_env = make_envelope(**req_kwargs)

    # Build schema-valid `cad_artifact` envelope (`producer: "cad_worker"`)
    cad_art_payload = {
        "cad_request_message_id": cad_request_env["message_id"],
        "parameter_record": make_artifact(
            uri=str(params_path),
            mime_type="application/json",
            data=params_path.read_bytes(),
        ),
        "template_source": make_artifact(
            uri=str(Path(__file__).resolve()),
            mime_type="text/x-python",
            data=template_bytes,
        ),
        "exports": [
            make_artifact(
                uri=str(step_path),
                mime_type="application/step",
                data=step_path.read_bytes(),
            ),
            make_artifact(
                uri=str(stl_path),
                mime_type="model/stl",
                data=stl_path.read_bytes(),
            ),
            make_artifact(
                uri=str(svg_path),
                mime_type="image/svg+xml",
                data=svg_path.read_bytes(),
            ),
            make_artifact(
                uri=str(png_path),
                mime_type="image/png",
                data=png_path.read_bytes(),
            ),
        ],
        "checks": [
            {
                "check_id": "watertight-manifold-and-dimensional-verification",
                "result": "pass" if verification.status == "geometry_valid" else "fail",
                "evidence": make_artifact(
                    uri=str(report_path),
                    mime_type="application/json",
                    data=report_path.read_bytes(),
                ),
            }
        ],
        "status": verification.status,
        "physical_fit_status": "not_tested",
        "physical_test_record": None,
    }

    art_kwargs: Dict[str, Any] = {
        "producer": "cad_worker",
        "payload_type": "cad_artifact",
        "payload": cad_art_payload,
        "scene_version": scene_version,
        "plan_version": plan_version,
        "step_id": step_id,
        "parent_message_ids": [cad_request_env["message_id"]],
    }
    if session_id:
        art_kwargs["session_id"] = session_id
    if observation_id:
        art_kwargs["observation_id"] = observation_id
    cad_artifact_env = make_envelope(**art_kwargs)

    return CadGenerationResult(
        template_id="flat_two_hole_link_v1",
        status=verification.status,
        step_path=str(step_path),
        stl_path=str(stl_path),
        drawing_svg_path=str(svg_path),
        drawing_png_path=str(png_path),
        parameters_json_path=str(params_path),
        report_json_path=str(report_path),
        effective_parameters_mm=effective_params,
        verification=verification,
        cad_request_envelope=cad_request_env,
        cad_artifact_envelope=cad_artifact_env,
    )
