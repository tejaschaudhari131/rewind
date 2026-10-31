# REWIND × Flat-Pack Furniture: Reframe Plan (Recommendation A)

> "Every year, people build furniture wrong and find out ten steps later. REWIND watches your hands and says stop."

Status: plan only. Nothing here is implemented yet unless it is marked **exists**. Owner: Tejas (solo). Footage target: **Nov 2**. Submission: **Nov 4, 9 PM ET**.

## 1. Hero mistake (one, legible in 2 seconds)

**A side panel mounted with its cam-lock face turned outward.**

- In outline the panel is the same rectangle either way, so SAM mask IoU is about 0.95–0.98 and a silhouette check passes.
- It is wrong: the cam-lock holes and the half-drilled shelf-pin holes face the outside of the cabinet. The shelf can't lock, and the error only shows up several steps later. That is the most common flat-pack mistake people complain about.
- Anyone can see it once it's pointed out: "the holes are on the wrong side." The visual pays off instantly.

Backup (if the panel is awkward to film): a **drawer front mounted upside down**. The handle holes end up low and off-center, and the outline is identical.

## 2. What to get

Use any flat-pack piece where one side panel has **pre-drilled cam-lock or shelf-pin holes on one face only** and a **finished front edge** (an edge-banded strip on one long edge).

- **Option 1: use what you own.** Any IKEA-style bookshelf, nightstand, or small cabinet. Partially disassemble one side panel. This costs $0 and is the fastest.
- **Option 2: buy a small piece.** A small flat-pack shelf or nightstand from IKEA Pittsburgh (Robinson Twp.), Target, or Walmart. Budget about $20–$50, and check prices on the day. Choose one with a side panel of 60 cm or less so it fits in an overhead phone shot.

Mark the panel for the witness check (cheap, and it doesn't need to be visible to a viewer). The real holes are the witness, but adding a small strip of **colored painter's tape on the finished front edge** gives a robust second feature that matches today's stripe-based witness code.

## 3. How the existing pipeline maps

| REWIND today | IKEA reframe |
|---|---|
| Slider-crank on AprilTag mat | Panel lying on or next to the mat tiles (use the mat for scale and pose only) |
| Reference photo per checkpoint (**exists**) | One reference photo per manual step ("Step 3: left side panel, holes facing in") |
| SAM 3.1 concept prompt `connecting rod` | `wooden panel` / `white furniture panel`; optionally `cam lock hole` as a second concept |
| Witness = colored stripe polarity (CIE LAB cosine) (**exists**) | Witness = (a) tape-strip polarity on the finished edge (no code change), then (b) a hole-pattern side check: dark circular blobs present vs. absent on the visible face |
| Blinded visual audit, Baseten `muse-spark-open-1026` (**exists**) | Same crops, with the prompt reworded to "Does the visible face show the pre-drilled holes? Is the finished edge at the front?" |
| Two-stage gate, fail-closed (**exists**) | Unchanged. Stage 1 gives a provisional verdict in about 1.3 s; Baseten stage 2 confirms in about 5–7 s |
| Voice "check step" (muse-voice-transcribe) (**exists**) | Unchanged: "Check step three." |
| muse-image repair card (**exists**) | A "Flip the panel: holes face in" card |
| CAD link regeneration | Cut, or one 10 s beat (see the demo script) |

## 4. Code changes (only after the README/backend honesty pass lands, to avoid conflicts)

| # | Change | Files | Effort |
|---|---|---|---|
| 1 | Checkpoint config for the furniture: step ids, SAM concept prompt, and reference image paths, instead of hard-coded rod names | new `fixtures/ikea/checkpoints.json`; small loader in `backend/app/gate.py` / `config.py` | 2–3 h |
| 2 | Make the SAM concept prompt configurable per checkpoint | `backend/app/providers/meta_sam.py` | 1 h |
| 3 | Hole-face witness: count dark circular blobs in the panel crop (OpenCV HoughCircles or blob detector) and compare with the reference count. Report this as a deterministic pre-check, not a model | `backend/app/geometry.py` (new function) and tests in `tests/` | 3–4 h |
| 4 | Reword the blinded audit prompt for furniture; keep the blinding assertion | `backend/app/providers/baseten_vision.py` | 1 h |
| 5 | Muse Wonder manual check-off (section 5) | `backend/app/providers/meta_wonder.py`, `backend/app/main.py` route, new `frontend` manual pane | 4–6 h |
| 6 | Relabel the UI: "Step 3 of 8 · Left side panel", hide CAD by default | `frontend/src/App.tsx`, `WorkbenchUpgrades.tsx` | 2 h |
| 7 | Run the real footage through the gate and record measured results in an honest table | `scripts/` (new eval script), `docs/` | 2–3 h |

Total: about 2 working days. Do 1–4 first, so the catch works on real footage, then 6, then 5 if time allows.

## 5. Muse Wonder: a real manual check-off

Goal: a genuine computer-use action, not an overlay.

1. Serve the assembly manual locally. Either use the real PDF from the retailer's product page, saved to `fixtures/ikea/manual.pdf`, or a one-page HTML checklist generated from it (`fixtures/ikea/manual.html`, with one checkbox per step).
2. Open the page in a headless Chromium (Playwright) and take a screenshot.
3. Send the screenshot and the task ("Step 3 was verified PASS. Find step 3 and check its box. If REJECT, do nothing and highlight step 3.") to `muse-wonder` via the existing `run_cua_step` in `meta_wonder.py`.
4. Execute the returned click or scroll action in Playwright, then re-screenshot and verify the checkbox state in the DOM. Log the real request id and tokens in the telemetry ledger.
5. Fail closed: if Wonder returns no action, or the action does not change the DOM, the UI shows "Manual not updated", never a fake tick.

Use HTML for the checkbox (a PDF can't be clicked reliably) and show the PDF page beside it for authenticity. Be honest in the video: "Muse Wonder ticks the step in the manual."

## 6. Risks

- **Meta API reachability** depends on Tejas's computer being online (egress tunnel). Run `python scripts/preflight_live.py` before every take.
- **Stage-2 latency** is about 5–7 s. Show it on screen and don't hide it.
- **White-on-white holes** may have low contrast. Use side lighting and the tape strip as a fallback witness.
- **Calibration** stays "unvalidated" unless measured on real footage. Don't claim mm accuracy.
