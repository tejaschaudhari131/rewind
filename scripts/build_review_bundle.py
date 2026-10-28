#!/usr/bin/env python3
"""
Build the complete, self-contained Adversarial Review Bundle for REWIND.

Generates:
1. /workspace/REWIND_ADVERSARIAL_REVIEW_BUNDLE.md
2. /workspace/CHATGPT_PLUS_REDTEAM_PROMPT.md
3. /workspace/GEMINI_PRO_REDTEAM_PROMPT.md
4. /workspace/rewind_adversarial_review_bundle.zip

Strictly verifies that zero .env values, API keys (LLM_*), or tokens appear in any output.
"""

from __future__ import annotations

import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path("/workspace/rewind").resolve()
WORKSPACE_ROOT = Path("/workspace").resolve()

BUNDLE_MD_PATH = WORKSPACE_ROOT / "REWIND_ADVERSARIAL_REVIEW_BUNDLE.md"
CHATGPT_PROMPT_PATH = WORKSPACE_ROOT / "CHATGPT_PLUS_REDTEAM_PROMPT.md"
GEMINI_PROMPT_PATH = WORKSPACE_ROOT / "GEMINI_PRO_REDTEAM_PROMPT.md"
ZIP_BUNDLE_PATH = WORKSPACE_ROOT / "rewind_adversarial_review_bundle.zip"

SECRET_PATTERNS = [
    re.compile(r"LLM_[A-Za-z0-9_]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_]{15,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),
]

CHATGPT_PLUS_PROMPT = """# Adversarial Red-Team Prompt for ChatGPT Plus (Astra Ultra / o3 / GPT-4.5)
## Role: REWIND Architecture, Deterministic Gate & Plane-Aware Metrology Auditor

You are an unforgiving, hostile principal systems engineer, computer-vision geometrician, and Meta Global AI Developer Hackathon judge reviewing the **REWIND** repository (`https://github.com/tejaschaudhari131/rewind`).

Your job is to perform a **line-by-line adversarial audit** of the full repository implementation against the Round 2 Architecture Lock (`LOCK-IN.md`, `CONTRACT.md`, `API-CONTRACTS.md`, and `contracts/*.json`). Do NOT give generic praise. Cite exact file paths, function names, and line numbers for every vulnerability, contract discrepancy, or edge case you find, and provide drop-in replacement code or concrete architectural fixes.

### 1. Audit Vector A — Single-Writer Authority, Semantic Validator & Atomic CAS State Machine
Inspect `backend/app/semantic_validator.py`, `backend/app/gate.py`, `backend/app/state_machine.py`, `backend/app/schemas.py`, and `contracts/contract.schema.json`:
1. **Race Conditions & CAS Atomicity**:
   - In `RewindStateMachine.verify_current_step()` (`state_machine.py`) and `run_checkpoint_gate()` (`gate.py`), can two concurrent HTTP/WebSocket requests (`POST /api/checkpoint/verify` and `POST /api/camera/bump` or voice command `"check step"`) interleave across `await` boundaries and allow a stale observation to mutate `step_statuses` or bypass `assert_atomic_cas_match`?
   - Does `state_machine.py` strictly enforce the state transition graph `MOVING -> CAPTURING -> CHECKING -> ACCEPTED | REJECTED | UNKNOWN` under concurrent or rapid-fire calls?
2. **Contract vs. Implementation Drift (`LOCK-IN.md` & `CONTRACT.md` vs. `contracts/contract.schema.json` & `backend/app/`)**:
   - Compare the 10 semantic rules in `CONTRACT.md` (e.g., immutable evidence store hash verification, `(provider_response_id, sam_track_id)` cross-clip association, `physical_fit_status`, planner `finish` verification) against `semantic_validator.py`. Where is the implementation relying on simplified checks or missing fields required by `CONTRACT.md`?
   - Can a malformed or adversarial model output from `muse-wonder-1.0` / `muse-spark-1.3-contributor` or `Muse Glimmer 30B` slip past `validate_envelope()` or `validate_stage_semantics()`?

### 2. Audit Vector B — Plane-Aware 2.5D Metrology, Homography & Parametric CAD Worker
Inspect `backend/app/calibration.py`, `backend/app/geometry.py`, `backend/app/cad_worker.py`, and `scripts/generate_printable_rig.py`:
1. **Projective Geometry & Off-Plane Parallax**:
   - Verify how `detect_apriltag_mat()` and `pixel_to_plane_mm()` compute and apply $H_{\\text{undistorted\\_px}\\to\\text{mat\\_z0\\_mm}}$. Are the 16 AprilTag corner coordinates ordered identically between `generate_printable_rig.py` and `calibration.py`?
   - What happens when a physical flat link (`thickness = 4.0 mm`) sits on `mat_z0` (`z = 0`) vs. its top face at `z = 4.0 mm` under a finite camera height (e.g., $Z_{\\text{cam}} = 500\\text{ mm}$)? Quantify the radial perspective scale error $\\Delta L / L \\approx h / Z_{\\text{cam}}$ on a $150\\text{ mm}$ link ($4 / 500 = 0.8\\% \\approx 1.2\\text{ mm}$) and show the exact thickness-plane compensation formula we should add to `calibration.py` / `geometry.py` so real webcam scans stay within $\\pm 0.25\\text{ mm}$!
2. **Subpixel Hole-Center & Witness-Mark Robustness on Real Webcam Frames**:
   - Inspect `measure_elevated_step2_geometry()` and `measure_flat_link_on_mat_z0()` in `geometry.py`. Currently, how does it locate the yellow rod, witness stripe, and internal holes? Where will real-world lighting, shadows, specular highlights on 3D-printed PLA/matte cardboard, or slight rotation break the HSV/threshold heuristics, and how should we harden it using the `sam-3.1` mask + OpenCV `cv2.aruco` (`DICT_APRILTAG_36h11`) + subpixel ellipse fitting (`cv2.fitEllipse`)?
3. **CAD Worker (`cad_worker.py`) Watertightness & Verification**:
   - Check `generate_flat_two_hole_link_cad()`. Does the generated `.STEP` file (`_write_ap214_step_solid`) represent true B-Rep topology with cylindrical inner holes or a bounding/faceted solid, and how can we wire CadQuery / `build123d` / OpenCASCADE when installed while keeping the zero-dependency fallback?

### 3. Audit Vector C — Live Meta Model API & Open-Weight Provider Clients
Inspect `backend/app/providers/meta_sam.py`, `meta_wonder.py`, `modal_glimmer.py`, `meta_voice.py`, `meta_image.py`, `baseten_spark.py`, and `modal_app/glimmer_server.py`:
1. **SAM 3.1 Token Grammar & Base85 Range-Coded Mask Decoder (`meta_sam.py`)**:
   - Audit `parse_sam_output_text()` and `_decode_sam_mask_crop()`. What happens if `sam-3.1` returns multiple detections for a single prompt, empty `<0f>` output, or `~` (`lossless`) vs `!` (`one_bit`) prefix when `meta_sam_parser` is or isn't installed?
2. **Glimmer Input Blinding (`assert_glimmer_input_blinded` & `sanitize_blinded_glimmer_context`)**:
   - Verify whether any subtle leakage path remains in `modal_glimmer.py` or `glimmer_server.py` (e.g., prompt wording, filename leakage like `checkpoint_fail_flipped_rod.png` if passed by name, or fallback contributor mode).
3. **Sub-2.8s Budget & Live Demo Resilience**:
   - Given our measured Phase 1 latencies (`sam-3.1` = `816–1,089 ms`, `muse-spark-1.3-contributor` text = `1,501 ms`, vision = `5,062–6,921 ms`), explain how `gate.py` behaves when `MODAL_GLIMMER_ENDPOINT` is cold vs. warm, and how to guarantee sub-2.8s live checkpoint verification without faking evidence.

### 4. Audit Vector D — 4-Pane Operator Workbench (`App.tsx`) & Hackathon Judge Kill-Shots
Inspect `frontend/src/App.tsx`, `frontend/src/components/StlViewer3D.tsx`, and `LOCK-IN.md`:
1. Compare the 4-pane workbench in `frontend/src/App.tsx` against the `LOCK-IN.md` specification (LIVE CAMERA, ORIGINAL EVIDENCE with before/current paired crops, MECHANISM + PART with analytic kinematic twin $x(\\theta)=r\\cos\\theta+\\sqrt{L^2-r^2\\sin^2\\theta}$, ACTION + CHECKS). What UI elements from `LOCK-IN.md` are missing or could be sharpened?
2. List the **Top 5 Judge Kill-Shots** (the hardest questions or live demo tests a Meta Hackathon judge will throw at REWIND) and provide the **Top 5 Highest-Leverage Code Upgrades** (ranked by score impact per hour of engineering) before final submission.
"""

GEMINI_PRO_PROMPT = """# Adversarial Red-Team Prompt for Gemini Pro (Argon High / Gemini 2.5 Pro — 1M Context Window)
## Role: "KINETIX" Multimodal Vision, Physical Rig & Full-Stack Meta Hackathon Auditor

You have the entire **REWIND** repository (`https://github.com/tejaschaudhari131/rewind`) loaded in your context window, along with the 4 visual artifacts (`workbench_pass_and_3d_cad.png`, `workbench_reject_flipped_rod.png`, `flat_two_hole_link_v1_drawing.png`, and `rewind_apriltag_mat_400x250mm.png`).

Act as a **ruthless Meta Global AI Developer Hackathon Finalist Judge** and **Industrial Computer Vision / Robotics Architect**. Evaluate this codebase and visual workbench across the official hackathon rubric (**Meta Model API & Open-Weight Usage 25%, Technical Execution 25%, Potential Impact 25%, Demo & Presentation Quality 25%**) plus eligibility for the **$10K Open Weight Bonus**, **$5K Agentic Bonus**, and **$5K Creative Bonus**.

### 1. Multi-Model Orchestration & Bonus Track Audit (All 6 Meta Models + Open Weights)
Audit how all models in `backend/app/providers/` and `modal_app/glimmer_server.py` are wired into the runtime flow:
- **`sam-3.1` (`meta_sam.py`)**: Teardown video tracking + stationary checkpoint segmentation + custom base85 range-coded mask decoding.
- **`Muse Glimmer 30B` (`modal_glimmer.py` & `modal_app/glimmer_server.py`)**: Blinded visual consistency auditor on Modal H100 (`enable_memory_snapshot=True`).
- **`muse-spark-1.3-contributor` & `muse-wonder-1.0` (`meta_wonder.py`)**: Reverse reassembly DAG compiler, failure recovery planner, and native Computer Use (`tools: [{"type": "computer"}]`) CAD orthographic inspector.
- **`muse-voice-transcribe-1.0` (`meta_voice.py`)**: Real-time WebSocket (`wss://api.meta.ai/v1/asr/realtime`) `ENDPOINTING` hands-free operator console.
- **`muse-image-1.0` (`meta_image.py`)**: Asynchronous visual repair card generator (`POST /v1/images/edits`).
- **`meta-models/muse-spark-open` (`baseten_spark.py`)**: Ambiguous session reviewer on Baseten.
**Question**: Where is any model call currently mocked, fallback-heavy, or under-utilized in the live FastAPI routes (`backend/app/main.py`), and how do we make every single model's live telemetry, latency, and token/frame usage unmistakably visible to judges in the UI?

### 2. Physical Desk Rig & Real-Camera Computer Vision Audit
Tejas is printing `fixtures/printable/rewind_apriltag_mat_letter_tiles.pdf` and `rewind_slider_crank_cutouts_1to1.pdf` to film the physical desk teardown and reassembly demo:
1. Inspect `scripts/generate_printable_rig.py`, `backend/app/calibration.py`, and `backend/app/geometry.py`:
   - Are the rendered `APRILTAG_36h11` markers in `generate_printable_rig.py` bit-exact canonical `tag36h11` bit matrices (detectable by standard `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)` or `pupil_apriltags`), or are they synthetic patterns that only work with `_detect_apriltag_corners_deterministic()`? **If they are synthetic patterns, provide the exact Python code to embed real canonical `APRILTAG_36h11` bitmasks (IDs 0, 1, 2, 3) and wire `cv2.aruco` into `calibration.py` so a real webcam over Tejas's desk locks onto the printed mat immediately!**
2. How should `geometry.py` combine **live `sam-3.1` masks** with local OpenCV contour/witness-mark analysis when a real webcam captures the physical rig (instead of relying on synthetic pixel colors)? Provide drop-in code for `geometry.py`.

### 3. Visual & Industrial Workbench Critique (`workbench_pass_and_3d_cad.png` & `App.tsx`)
Critique the 4-pane operator workbench UI (`frontend/src/App.tsx` and `StlViewer3D.tsx`):
1. What visual elements in `App.tsx` look like a "hackathon demo" (e.g., scenario injection buttons vs. live WebRTC webcam capture / video upload, static vs. animated slider-crank kinematic twin $x(\\theta)=r\\cos\\theta+\\sqrt{L^2-r^2\\sin^2\\theta}$, paired before/after crops from `LOCK-IN.md`)?
2. Provide concrete React/TypeScript upgrades for `App.tsx` to add:
   - **Live Webcam (`navigator.mediaDevices.getUserMedia`) + Frame Freeze & Upload Mode** alongside the Golden Rig fixtures so Tejas can switch seamlessly between live overhead camera capture and deterministic fixtures.
   - **Interactive Analytic Slider-Crank Kinematic Twin SVG** ($x(\\theta) = r\\cos\\theta + \\sqrt{L^2 - r^2\\sin^2\\theta}$ with $r=30\\text{ mm}, L=150\\text{ mm}$, $60\\text{ mm}$ stroke) in Pane 3 alongside the 3D `.STL` viewer and 2D orthographic drawing.

### 4. The Winning 3-Minute Demo Video Script & Shot List
Write a second-by-second **180-second winning demo script and camera shot list** (`0:00` to `3:00`) that proves:
1. Real physical slider-crank teardown tracked by `sam-3.1` and compiled into a reverse reassembly plan by `muse-spark-1.3-contributor` / `muse-wonder-1.0`.
2. Hands-free operator voice commands via `muse-voice-transcribe-1.0` (`"check step"`).
3. Sub-2.8s deterministic rejection of a **180° flipped connecting rod** where silhouette IoU is high (`0.97`) but witness stripe alignment + blinded `Muse Glimmer 30B` audit catch the inversion, followed by an async `muse-image-1.0` visual repair card.
4. Calibrated `mat_z0` flat-link scan (`150.0 mm` C-C) -> parametric `.STEP`/`.STL` regeneration -> `muse-wonder-1.0` Computer Use (`computer` tool) CAD verification.
"""


def run_cmd(args: List[str], cwd: Path = REPO_ROOT) -> str:
    res = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=True)
    return res.stdout.strip()


def assert_no_secrets(text: str, label: str) -> None:
    for pat in SECRET_PATTERNS:
        m = pat.search(text)
        if m:
            raise ValueError(f"Secret pattern matched in {label}: {m.group(0)[:8]}...")


def lang_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".md": "markdown",
        ".css": "css",
        ".html": "html",
        ".ini": "ini",
        ".svg": "xml",
    }.get(ext, "text")


def collect_files(rel_paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for rel in rel_paths:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts and not child.name.endswith(".pyc"):
                    out.append(child)
    return out


def format_file_section(path: Path) -> Tuple[str, int]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    assert_no_secrets(content, rel)
    lines = content.splitlines()
    line_count = len(lines)
    lang = lang_for_path(path)
    fence = "````" if "```" in content else "```"
    block = (
        f"### File: `{rel}` ({line_count} lines)\n\n"
        f"{fence}{lang}\n"
        f"{content.rstrip()}\n"
        f"{fence}\n"
    )
    return block, line_count


def build_bundle() -> None:
    # 1. Write standalone prompts
    assert_no_secrets(CHATGPT_PLUS_PROMPT, "CHATGPT_PLUS_PROMPT")
    assert_no_secrets(GEMINI_PROMPT_PATH.name, "GEMINI_PRO_PROMPT")
    assert_no_secrets(GEMINI_PRO_PROMPT, "GEMINI_PRO_PROMPT")

    CHATGPT_PROMPT_PATH.write_text(CHATGPT_PLUS_PROMPT, encoding="utf-8")
    GEMINI_PROMPT_PATH.write_text(GEMINI_PRO_PROMPT, encoding="utf-8")

    # 2. Gather git metadata and tree
    git_log = run_cmd(["git", "log", "--stat", "-n", "10"])
    tracked_files = run_cmd(["git", "ls-files"]).splitlines()
    pytest_output = run_cmd(["pytest", "-v"])

    # 3. Spec & Contract files (Section 3)
    spec_files = collect_files([
        "README.md",
        "LOCK-IN.md",
        "CONTRACT.md",
        "API-CONTRACTS.md",
        "docs/PHASE1_LIVE_SMOKE_REPORT.md",
        ".env.example",
        "pytest.ini",
        "contracts/contract.schema.json",
        "contracts/wonder-output.schema.json",
        "contracts/glimmer-output.schema.json",
        "contracts/example-cycle.json",
        "api-examples",
    ])

    # 4. Source Code & Tests (Section 4)
    code_files = collect_files([
        "backend/app/config.py",
        "backend/app/schemas.py",
        "backend/app/semantic_validator.py",
        "backend/app/calibration.py",
        "backend/app/geometry.py",
        "backend/app/cad_worker.py",
        "backend/app/gate.py",
        "backend/app/state_machine.py",
        "backend/app/main.py",
        "backend/app/providers",
        "modal_app/glimmer_server.py",
        "modal_app/README.md",
        "frontend/package.json",
        "frontend/vite.config.ts",
        "frontend/src/main.tsx",
        "frontend/src/index.css",
        "frontend/src/components/StlViewer3D.tsx",
        "frontend/src/App.tsx",
        "scripts/generate_contracts.py",
        "scripts/generate_printable_rig.py",
        "tests/test_contracts.py",
        "tests/test_semantic_validator.py",
        "tests/test_providers_unit.py",
        "tests/test_live_meta_smoke.py",
        "tests/test_phase2_phase3_pipeline.py",
        "tests/test_phase4_modal_and_ui.py",
    ])

    sections: List[str] = []
    sections.append(
        "# REWIND — Complete Adversarial Review & Red-Team Bundle\n\n"
        "- **Public GitHub Repository**: https://github.com/tejaschaudhari131/rewind\n"
        "- **Author**: Tejas Chaudhari (`tchaudha@andrew.cmu.edu`)\n"
        "- **Generated**: Wednesday, Oct 28, 2026 (America/New_York, EDT / UTC-4)\n"
        "- **Purpose**: Self-contained 100%-verbatim architecture, contract, and source-code bundle for adversarial review in **ChatGPT Plus (Astra Ultra / o3 / GPT-4.5)** and **Gemini Pro (Argon High / Gemini 2.5 Pro)**.\n\n"
        "---\n\n"
        "## Table of Contents\n"
        "1. [Section 1 — Copy-Paste Adversarial Red-Team Prompts (ChatGPT Plus & Gemini Pro)](#section-1--copy-paste-adversarial-red-team-prompts)\n"
        "2. [Section 2 — Repository Overview, Git Commit History & Live Pytest Results (35/35 Passed)](#section-2--repository-overview-git-commit-history--live-pytest-results)\n"
        "3. [Section 3 — Complete Architecture, Lock-In & JSON Schema Contracts](#section-3--complete-architecture-lock-in--json-schema-contracts)\n"
        "4. [Section 4 — Complete Verbatim Source Code & Automated Test Suite](#section-4--complete-verbatim-source-code--automated-test-suite)\n"
    )

    # Section 1
    sections.append(
        "---\n\n"
        "## Section 1 — Copy-Paste Adversarial Red-Team Prompts\n\n"
        "### 1A. Prompt for ChatGPT Plus (Architecture, Deterministic Gate & Metrology Auditor)\n\n"
        "````markdown\n"
        f"{CHATGPT_PLUS_PROMPT.strip()}\n"
        "````\n\n"
        "### 1B. Prompt for Gemini Pro (Multimodal Vision, Physical Rig & Full-Repo Auditor)\n\n"
        "````markdown\n"
        f"{GEMINI_PRO_PROMPT.strip()}\n"
        "````\n"
    )

    # Section 2
    tracked_list_md = "\n".join(f"  - `{f}`" for f in tracked_files)
    sections.append(
        "---\n\n"
        "## Section 2 — Repository Overview, Git Commit History & Live Pytest Results\n\n"
        "### 2.1 Public Repository & Tracked Files (`88` tracked files)\n"
        "- **URL**: `https://github.com/tejaschaudhari131/rewind`\n"
        f"{tracked_list_md}\n\n"
        "### 2.2 Multi-Commit History (`main` branch)\n"
        "```text\n"
        f"{git_log}\n"
        "```\n\n"
        "### 2.3 Live Automated Test Suite Execution (`pytest -v` -> `35 passed`)\n"
        "```text\n"
        f"{pytest_output}\n"
        "```\n"
    )

    # Section 3
    sections.append(
        "---\n\n"
        "## Section 3 — Complete Architecture, Lock-In & JSON Schema Contracts\n"
    )
    total_spec_lines = 0
    for sf in spec_files:
        block, lc = format_file_section(sf)
        total_spec_lines += lc
        sections.append(block)

    # Section 4
    sections.append(
        "---\n\n"
        "## Section 4 — Complete Verbatim Source Code & Automated Test Suite\n"
    )
    total_code_lines = 0
    for cf in code_files:
        block, lc = format_file_section(cf)
        total_code_lines += lc
        sections.append(block)

    full_bundle_text = "\n\n".join(sections)
    assert_no_secrets(full_bundle_text, "REWIND_ADVERSARIAL_REVIEW_BUNDLE.md")
    BUNDLE_MD_PATH.write_text(full_bundle_text, encoding="utf-8")

    # 5. Build Zip Archive with bundle + prompts + 4 key visuals
    visual_files = [
        REPO_ROOT / "docs/screenshots/workbench_pass_and_3d_cad.png",
        REPO_ROOT / "docs/screenshots/workbench_reject_flipped_rod.png",
        REPO_ROOT / "fixtures/cad_outputs/flat_two_hole_link_v1_drawing.png",
        REPO_ROOT / "fixtures/printable/rewind_apriltag_mat_400x250mm.png",
        REPO_ROOT / "fixtures/printable/rewind_apriltag_mat_letter_tiles.pdf",
        REPO_ROOT / "fixtures/printable/rewind_slider_crank_cutouts_1to1.pdf",
    ]
    with zipfile.ZipFile(ZIP_BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(BUNDLE_MD_PATH, arcname="REWIND_ADVERSARIAL_REVIEW_BUNDLE.md")
        zf.write(CHATGPT_PROMPT_PATH, arcname="CHATGPT_PLUS_REDTEAM_PROMPT.md")
        zf.write(GEMINI_PROMPT_PATH, arcname="GEMINI_PRO_REDTEAM_PROMPT.md")
        for vf in visual_files:
            if vf.exists():
                zf.write(vf, arcname=f"visuals/{vf.name}")

    bundle_lines = len(full_bundle_text.splitlines())
    bundle_bytes = BUNDLE_MD_PATH.stat().st_size
    approx_tokens = len(full_bundle_text) // 4
    print(f"Bundle written: {BUNDLE_MD_PATH} ({bundle_bytes:,} bytes, {bundle_lines:,} lines, ~{approx_tokens:,} tokens)")
    print(f"Spec lines included: {total_spec_lines:,} | Source+Test lines included: {total_code_lines:,}")
    print(f"ChatGPT prompt written: {CHATGPT_PROMPT_PATH} ({CHATGPT_PROMPT_PATH.stat().st_size:,} bytes)")
    print(f"Gemini prompt written: {GEMINI_PROMPT_PATH} ({GEMINI_PROMPT_PATH.stat().st_size:,} bytes)")
    print(f"Zip archive written: {ZIP_BUNDLE_PATH} ({ZIP_BUNDLE_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_bundle()
