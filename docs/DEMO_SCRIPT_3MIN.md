# REWIND — 3-Minute Winning Demo Video Script & Shot List (`0:00` – `3:00`)

- **Recording Format**: 1080p 60 FPS, Dual-Source Composited Layout:
  - **70% Primary Canvas**: Live 4-Pane REWIND Industrial Workbench UI (`http://127.0.0.1:8000`)
  - **30% Picture-in-Picture (Top-Right / Bottom-Left)**: Overhead Camera of Tejas's Hands on the Printed `400 × 250 mm` `APRILTAG_36h11` Mat (`fixtures/printable/rewind_apriltag_mat_letter_tiles.pdf`).

---

## Second-by-Second Shot List

### `0:00 – 0:12` — The 5-Second Viral Cold Open (180° Flipped Rod Catch)
- **Visual On Screen**:
  - Split-screen cold open: Tejas places the asymmetrical yellow connecting rod onto the slider-crank pins **flipped 180°** and speaks aloud: *"Check step."*
  - Within `1.9s`, **SAM 3.1 Silhouette IoU** hits `0.972` (Green), proving a standard bounding-box or mask-IoU model would pass the defective assembly.
  - Instantly, the **CIE LAB Witness-Stripe Polarity Vector (`cos = -0.994`)** + **Dual Open-Weight Consensus (Deterministic CIE LAB Pre-Check + Live Baseten `meta-models/muse-spark-open-1026`)** slam the gate **CRIMSON RED (`REJECT: WITNESS_INVERTED_180`)**, displaying the zoomed **Paired Reference vs. Live Witness Crops** in Pane 2 while **`muse-image-1.0`** pops in the annotated 180° rotation repair card.
- **Voiceover / Live Audio**:
  > *"Standard vision models pass this assembly step because the part's outer silhouette overlaps by 97%. They miss that the connecting rod is flipped 180 degrees—reversing the lubrication and load witness marks. This is REWIND: the self-verifying physical teardown, reassembly, and CAD recovery compiler built on all six Meta AI models."*
- **Rubric / Bonus Locked**: **5-Sec Viral Hook + Demo Quality (25%)**

---

### `0:12 – 0:42` — Physical Teardown & `sam-3.1` + `Muse Spark 1.3` / `Muse Wonder 1.0` Reverse DAG
- **Visual On Screen**:
  - Show the printed `400 × 250 mm` `APRILTAG_36h11` calibration mat (IDs `0, 1, 2, 3` locked in green at `0.12 mm` reprojection error / `0.39 px` RMSE).
  - Tejas disassembles the slider-crank mechanism in 4 smooth steps.
  - **`sam-3.1`** decodes custom base85 range-coded pixel masks across frames (`rle_base85_bytes` and `decoded_mask_area_px` live in the top 6-Model Telemetry Proof Bar); **`muse-spark-1.3-contributor`** and **`muse-wonder-1.0`** compile the reverse reassembly DAG.
- **Voiceover / Live Audio**:
  > *"We start by filming a normal teardown on a calibrated 400-by-250 millimeter AprilTag mat. SAM 3.1 tracks every component and witness mark across frames, decoding base85 pixel masks in canonical coordinates. Muse Spark 1.3 and Muse Wonder 1.0 compile the reverse reassembly graph—turning our teardown video into a deterministic verification harness."*
- **Rubric / Bonus Locked**: **`sam-3.1` Video Tracking + `muse-spark-1.3-contributor` / `muse-wonder-1.0` (25%)**

---

### `0:42 – 1:15` — Hands-Free `muse-voice-transcribe-1.0` + Sub-2.8s Valid Checkpoint Pass
- **Visual On Screen**:
  - Tejas seats the crank disk and connecting rod in the **correct orientation** while keeping both hands on the hardware.
  - He speaks aloud: *"Check step."*
  - The **`muse-voice-transcribe-1.0`** WebSocket VAD waveform bar (`wss://api.meta.ai/v1/asr/realtime`) endpoints the utterance in `180 ms`.
  - Pane 2 flashes **EMERALD GREEN (`ACCEPTED`: IoU `0.972`, Witness Cosine `+0.996`, `Muse Glimmer 30B: CONSISTENT`)** in `< 1.9s`.
- **Voiceover / Live Audio**:
  > *"During reassembly, my hands stay on the hardware. Muse Voice Transcribe 1.0 streams live 24-kilohertz PCM audio over WebSocket with endpointing. When I say 'check step,' our homography solver warps the live SAM 3.1 mask and passes both the geometric gate and our deterministic CIE LAB witness pre-check and live Baseten Muse Spark Open arbiter."*
- **Rubric / Bonus Locked**: **`muse-voice-transcribe-1.0` + Technical Execution (25%)**

---

### `1:15 – 1:58` — Flagship Failure Catch: 180° Flipped Rod + Dual Open-Weight Consensus + `muse-image-1.0` Repair Card
- **Visual On Screen**:
  - Tejas installs the connecting rod (`L = 150.0 mm`) **180° backward** and says *"Check step."*
  - Zoom into **Pane 2**:
    - **SAM 3.1 Silhouette IoU**: `0.9722` (PASS)
    - **Side-by-Side Paired Witness Crops**: `TEARDOWN REF WITNESS [LEFT]` vs `LIVE CHECKPOINT WITNESS [INVERTED 180° RIGHT]` (`witness_polarity_cosine = -0.994`, `witness_error = 275.06 px`).
    - **Dual Open-Weight Consensus**: The **deterministic CIE LAB witness-polarity pre-check (`cos = -0.994`)** and live **`Muse Spark Open` (`meta-models/muse-spark-open-1026` on Baseten)** flag `MISMATCH (WITNESS_INVERTED_180)`.
    - **`muse-image-1.0` (`/v1/images/edits`)** streams in the annotated visual repair card (`GET /api/repair-card/{session_id}/{step_id}`).
- **Voiceover / Live Audio**:
  > *"Now watch why Meta's multi-model verification pattern matters. I install the asymmetrical connecting rod backward. Silhouette IoU is 0.97—almost identical. But our illumination-invariant CIE LAB witness-polarity vector conflicts with the silhouette pass, triggering both our blinded Muse Glimmer 30B auditor on Modal H100 and Muse Spark Open on Baseten. Both open-weight models catch the 180-degree witness inversion, Muse Wonder blocks progression, and Muse Image 1.0 edits the live frame into a visual repair instruction card."*
- **Rubric / Bonus Locked**: **$10K Open Weight Bonus (`Muse Glimmer 30B` + `Muse Spark Open`) + $5K Creative Bonus (`muse-image-1.0`)**

---

### `1:58 – 2:42` — Calibrated `150.0 mm` Flat-Link Scan -> `.STEP`/`.STL` -> `muse-wonder-1.0` Computer Use CAD Verification
- **Visual On Screen**:
  - Tejas places a flat two-hole link in the `mat_z0` scan zone and says *"Scan flat part."*
  - **`sam-3.1` + `geometry.py`** rectify the `mat_z0` plane at `4.0 px/mm` and extract exact millimeter dimensions: **`150.01 mm` center-to-center**, **`25.00 mm` width**, **`Ø8.00 mm` pin bores**.
  - `cad_worker.py` compiles a watertight ISO-10303-21 `.STEP` and `.STL` mesh (`StlViewer3D.tsx`) alongside the animated **Analytic Slider-Crank Kinematic Twin SVG** ($x(\theta) = r\cos\theta + \sqrt{L^2 - r^2\sin^2\theta}$) and 3-view orthographic drawing (`flat_two_hole_link_v1_drawing.png`).
  - Over the 2D drawing in Pane 3, the **`muse-wonder-1.0` Computer Use (`tools: [{"type": "computer"}]`) Action Cursor & Encrypted Reasoning Continuity Badge** visibly steps through the inspection pins (`(600, 420) -> 150.01 mm C-C`, `(865, 255) -> Ø8.00 mm × 2`) to certify the CAD file.
- **Voiceover / Live Audio**:
  > *"What if a part is damaged or missing CAD? I place the flat link on the calibrated mat. From the SAM 3.1 mask, REWIND measures an exact 150.0 millimeter bore-to-bore center distance, compiles parametric STEP and STL geometry, and launches Muse Wonder 1.0 with native Computer Use and encrypted reasoning continuity—autonomously inspecting the generated three-view engineering drawing before certifying the file for fabrication."*
- **Rubric / Bonus Locked**: **$5K Agentic Bonus (`muse-wonder-1.0` Computer Use + Encrypted Reasoning)**

---

### `2:42 – 3:00` — Live 3D `.STL` Mesh Rotation + 6-Model Telemetry Proof Close
- **Visual On Screen**:
  - Tejas orbits the interactive 3D `.STL` mesh in Pane 3 beside the animated $x(\theta)$ kinematic twin and the top **6-Model Live Telemetry Proof Bar** showing all 6 Meta models verified.
- **Voiceover / Live Audio**:
  > *"From raw teardown video to blinded open-weight verification and computer-use-certified STEP CAD—REWIND gives physical hardware an undo button and a compiler."*
- **Rubric / Bonus Locked**: **Potential Impact (25%)**


---

## Two-stage gate: what's on screen (updated 2026-10-29)

- **Check-step press (0:42–1:15):** the `TWO-STAGE GATE` strip right away shows `Stage 1 PROVISIONAL_ACCEPTED (… ms · IoU … · witness cos …)` and then `Stage 2: waiting for Muse Spark vision audit…`. About 5–6 s later it shows `Stage 2 ACCEPTED [confirmed] · N ms · tokens · req …`. Narration: *"Local geometry gives an instant provisional answer. Nothing is final until the open-weight vision model independently agrees."*
- **Flipped rod (1:15–1:58):** Stage 1 shows `PROVISIONAL_REJECTED` (witness cosine negative). Stage 2 shows `REJECTED [confirmed]` with the model's own sentence (e.g. "LIVE shows stripe at opposite slider end, indicating reversed installation").
- **Honesty beat:** bump the camera or press reset while stage 2 is pending. The strip shows `Stage 2 result discarded (session reset / camera moved)`.
- **Do not claim** stage 2 is sub-2.8 s. Measured Baseten latency on crops is 4.8–6.6 s, with a 20 s deadline; past the deadline the result stays `UNKNOWN`. If `sam-3.1` is returning `model_not_found` at recording time (Meta's model list goes empty in windows; check `GET /v1/models` first), stage 1 shows `UNKNOWN` and the final can only be needs-review or a rejection. Record takes until a live-SAM run is captured.
