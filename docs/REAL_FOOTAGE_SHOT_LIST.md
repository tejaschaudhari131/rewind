# Real-Footage Production Shot List (Printed Mat + Cardboard Slider-Crank Rig)

Judges score **Demo Quality at 25%**. Synthetic frames alone will not win $150K; film these exact physical shots and composite them with the live workbench at `http://127.0.0.1:8000`.

## 1. Print & Cut Checklist (15 minutes, 100% scale — DO NOT "Fit to Page")
1. **Calibration Mat (`z = 0`)**: Print `fixtures/printable/rewind_apriltag_mat_letter_tiles.pdf` on 2 sheets of matte US Letter paper at **Actual Size (100%)**. Verify with a ruler that each corner `APRILTAG_36h11` tag (IDs `0, 1, 2, 3`) is exactly **40.0 × 40.0 mm** and the outer tag-corner span is **400.0 × 250.0 mm**. Splice along the `X = 200.0 mm` crosshairs and tape flat to a dark desk.
2. **Cardboard / Foamboard Cutouts**: Print `fixtures/printable/rewind_slider_crank_cutouts_1to1.pdf` at **100% scale**, glue onto 4 mm corrugated cardboard or foamcore, and cut out:
   - **Red Crank Disk**: radius `30.0 mm`, pivot pin at `(160, 125) mm`.
   - **Yellow Connecting Rod (`flat_two_hole_link_v1`)**: hole-to-hole `150.0 mm`, outer width `25.0 mm`, two `Ø8.0 mm` pin holes. **Critical**: draw a high-contrast **black + white witness stripe** across **ONLY ONE END** (the left/crank end) so the part is visually asymmetric.
   - **Blue Slider Block**: `60 × 40 mm`.
   - Use two M6 bolts, wooden dowels, or brass brads pushed up through the base as the crank and slider pins.

## 2. Camera & Lighting Setup
- **Mount**: Overhead tripod / desk arm **45–55 cm** straight above the mat center (`~90°` top-down) so all 4 AprilTags stay inside the frame with a 20 px margin and no hand occlusion when you pause.
- **Phone / Webcam Settings**: **1080p (`1920×1080`) at 60 fps**, **lock exposure (AE/AF Lock)** and **lock white balance** on the white paper mat so yellow/black/white LAB values don't drift when your hand enters. Use diffuse room light (avoid direct glare on the AprilTags).
- **Connect to Workbench**: Click **Enable Live Desk Cam (WebRTC)** in Pane 1 (or record MP4 on phone and AirDrop/upload for compositing).

## 3. Exact 5 Takes to Film (Total shoot time ~20 minutes)
| Take | Timecode | Physical Action on Mat | Workbench Action & Expected Result |
| :--- | :--- | :--- | :--- |
| **Take 1 — Cold Open & Flagship 180° Flip Reject** | `0:00–0:12` & `1:15–1:58` | Seat the yellow connecting rod onto the two pins **BACKWARDS (180° flipped)** so the black/white witness stripe is on the **RIGHT (slider) end** instead of the `[WITNESS REF]` crank end. Clear hands, say *"Check step."* | Silhouette IoU reads **`>= 0.95` (green)** while deterministic CIE LAB `witness_polarity_cosine` goes **negative (`~ -0.99`)** and Baseten `muse-spark-open-1026` returns **`MISMATCH`**. Gate slams **REJECTED**. |
| **Take 2 — 4-Step Teardown** | `0:12–0:42` | Start fully assembled; remove (1) retainer paddle, (2) yellow connecting rod, (3) blue slider + red crank, leaving bare mat with 4 visible AprilTags. | SAM 3.1 masks each removed part; reverse reassembly DAG populates in Pane 3. |
| **Take 3 — Correct Reassembly Pass** | `0:42–1:15` | Lift rod, rotate **180° in-plane** so the black/white witness stripe sits next to **`[WITNESS REF]`** on the left crank pin, seat flush, clear hands, say *"Check step."* | `witness_polarity_cosine > +0.90`, gate flashes **ACCEPTED**, state machine advances step. |
| **Take 4 — Camera Bump Quarantine** | `1:58–2:08` | Nudge the mat or tripod by ~1 cm during a check. | `registration_rmse_px > 1.5 px` triggers `CALIBRATION_INVALID` without corrupting session state. |
| **Take 5 — Flat `mat_z0` Scan -> `.STEP`/`.STL` CAD** | `2:08–2:45` | Place the bare yellow connecting rod **flat on the central `mat_z0` zone** (`z = 0`, all 4 AprilTags visible). Click **Scan Flat Rod (`mat_z0`)**. | Homography measures **`150.0 mm` C-C**, **`25.0 mm` width**, **`Ø8.0 mm` bores**; CadQuery exports `.STEP` + `.STL` and rotates live in the 3D viewer. |
