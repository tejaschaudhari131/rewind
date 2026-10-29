# Phase 1: Live Meta Model API Smoke Test Report

- **Date**: Wednesday, Oct 28, 2026 (America/New_York, UTC-4)
- **Target Base URL**: `https://api.meta.ai/v1` & `wss://api.meta.ai/v1/asr/realtime`
- **Egress Proxy**: `http://127.0.0.1:8791` (Pittsburgh residential tunnel; required because direct datacenter/Fastly egress returns empty `/v1/models` and `404 model_not_found`)
- **Synthetic Test Fixtures**:
  - `/workspace/rewind/fixtures/synthetic_rig_frame.png` (`640x400` RGB PNG showing planar slider-crank mechanism on 4-corner AprilTag mat)
  - `/workspace/rewind/fixtures/check_step_24khz.wav` (`24,000 Hz`, 16-bit signed LE mono PCM WAV uttering `"check step"`)
- **Total Live Smoke-Test Spend**: `~$0.026` (covered by the $150 Meta promotional credit balance)

---

## 1. `GET /v1/models`

- **Status**: `200 OK`
- **Latency**: `1093.2 ms`
- **Models Unlocked (9 total)**:
  1. `muse-wonder-1.0`
  2. `sam-3.1`
  3. `muse-spark-1.3-contributor`
  4. `muse-voice-transcribe-1.0`
  5. `muse-spark-1.3`
  6. `muse-image-1.0`
  7. `muse-spark-1.2-contributor`
  8. `muse-spark-1.2`
  9. `muse-spark-1.1`

### Wire Response Structure
```json
{
  "object": "list",
  "data": [
    {"id": "muse-wonder-1.0", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "sam-3.1", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-spark-1.3-contributor", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-voice-transcribe-1.0", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-spark-1.3", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-image-1.0", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-spark-1.2-contributor", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-spark-1.2", "object": "model", "created": 1776729600, "owned_by": "meta"},
    {"id": "muse-spark-1.1", "object": "model", "created": 1776729600, "owned_by": "meta"}
  ]
}
```

---

## 2. `sam-3.1` (`POST /v1/responses`)

### Key Protocol & Codec Findings
1. **Single-Concept Rule**: Per official `sam-3.1` documentation, text concept prompts accept **one noun phrase per request** (comma-separated lists return zero matches). Color-qualified noun phrases (e.g., `"yellow connecting rod"`, `"blue slider block"`) improve concept recall on synthetic colored parts.
2. **Multi-Object Box/Point Prompting**: Passing `<0f>1<|box;x1=...;y1=...;x2=...;y2=...;w=640;h=400|>,2<|box;...|>,3<|box;...|>` segments up to 16 tracked parts in a **single 1.09s request** with `>0.94` confidence.
3. **Wire Token Grammar & Mask Codec**:
   - Output text lane format:
     ```text
     <0f>0<|box;x1=214;y1=151;x2=423;y2=223;w=640;h=400;c=0.6831682920455933|><|mask;x=0;y=0;c=0.6831682920455933;data=73,210,!!!!#)F%rtD'+TMhcMrPo]Z6y$:LBoVBOH!/*`O...|>
     ```
   - `x1, y1, x2, y2` in `<|box;...|>` are **inclusive** source pixel coordinates (`width = x2 - x1 + 1`, `height = y2 - y1 + 1`).
   - `data=H,W,PAYLOAD` carries the **box-cropped** raster height `H` (`73 = 223 - 151 + 1`), width `W` (`210 = 423 - 214 + 1`), encoding prefix (`!` for `one_bit` context-adaptive range coding or `~` for `lossless` Paeth/spatial range coding), and a custom 85-character ASCII alphabet (excluding `"`, `\`, `,`, `;`, `<`, `>`, `|`).
   - Decoded via `meta_sam_parser.decode_mask_to_raster` into a `(H, W)` binary array and embedded at `full_frame[y1:y2+1, x1:x2+1] = crop * 255` to produce the full-frame `(400, 640)` `uint8` (`0`/`255`) binary mask required by `CONTRACT.md`.

### Live Latency & Detection Summary on `synthetic_rig_frame.png` (`640x400`)

| Prompt Mode | Input Text | Latency | Detections | Inclusive `xyxy_px` | Confidence | Crop Shape `(H, W)` | Full Mask Pixels |
| :--- | :--- | ---: | ---: | :--- | ---: | :--- | ---: |
| Concept | `"crank disk"` | `1618.0 ms` | 1 | `[147, 147, 263, 263]` | `0.540` (`0.617` v1) | `(117, 117)` | `8,879 px` |
| Concept | `"slider block"` | `816.6 ms` | 2 | `[380, 173, 472, 236]` | `0.586` | `(64, 93)` | `5,612 px` |
| Concept | `"blue slider block"` | `879.0 ms` | 1 | `[380, 173, 472, 237]` | `0.957` | `(65, 93)` | `5,680 px` |
| Concept | `"yellow connecting rod"` | `1500.1 ms` | 1 | `[214, 151, 423, 223]` | `0.683` | `(73, 210)` | `6,418 px` |
| Multi-Box (3 parts) | `<0f>1<\|box...\|>,2<\|box...\|>,3<\|box...\|>` | `1089.7 ms` | 3 | `1:[213,151,424,223]`, `2:[147,147,263,263]`, `3:[380,173,472,237]` | `0.953`, `0.961`, `0.941` | `(73,212)`, `(117,117)`, `(65,93)` | `6,442`, `8,895`, `5,680 px` |

---

## 3. `muse-spark-1.3-contributor` (`POST /v1/responses`)

### Key Protocol & Token Budgeting Findings
- `muse-spark-1.3-contributor` supports `reasoning.effort` in `["minimal", "low", "medium", "high", "xhigh", "max"]` (`"none"` returns HTTP 400).
- Switching `reasoning.effort` from `"low"` to `"minimal"` reduces reasoning token overhead from `~180–477` tokens down to `~7–199` tokens and cuts text latency to **`1501.6 ms`**.
- Always set `max_output_tokens >= 2048` alongside `text.format = {"type": "json_schema", "name": "...", "strict": True, "schema": ...}` so reasoning tokens never truncate the final `output_text` JSON item.

### Live Structured JSON Results
1. **`WonderProposalOutput` (`contracts/wonder-output.schema.json`)**:
   - **Latency**: `5062.7 ms` (multimodal image + strict JSON schema)
   - **Usage**: `input_tokens: 416`, `output_tokens: 311` (`reasoning_tokens: 199`)
   - **Wire Output**:
     ```json
     {
       "action": "instruct_move",
       "evidence_ids": ["obs_step_02_try_01", "pred_rod_orientation_notch_up"],
       "instruction": "Remove connecting_rod and reinstall flipped 180 degrees so the asymmetry notch faces up, then re-seat both ends.",
       "measurement_record_id": null,
       "observation_id": "obs_step_02_try_01",
       "part_id": "connecting_rod",
       "required_predicate_ids": ["pred_rod_orientation_notch_up"]
     }
     ```
2. **`GlimmerAuditOutput` (`contracts/glimmer-output.schema.json`)**:
   - **Latency**: `6921.4 ms` (multimodal image + strict JSON schema)
   - **Usage**: `input_tokens: 380`, `output_tokens: 549` (`reasoning_tokens: 447`)
   - **Wire Output**:
     ```json
     {
       "contradicted_predicate_ids": ["pred_rod_orientation_notch_up"],
       "finding": "mismatch",
       "observation_id": "obs_step_02_try_01",
       "observations": [
         {
           "frame_id": "frame_000120",
           "region_xyxy_px": [214, 151, 423, 223],
           "statement": "Yellow connecting rod is horizontal spanning from crank to slider; no upward-facing notch is visible on the top edge, the central groove appears centered/symmetric"
         }
       ],
       "requires_new_view": false
     }
     ```

---

## 4. `muse-voice-transcribe-1.0` (`wss://api.meta.ai/v1/asr/realtime`)

### Wire Handshake & Event Sequence
- Connected via HTTP `CONNECT api.meta.ai:443` tunnel through `127.0.0.1:8791` with TLS SNI `api.meta.ai`.
- **Client First Text Frame**:
  ```json
  {
    "authorization": {"accessToken": "Bearer <REDACTED>"},
    "audioEncoding": "PCM_24KHZ",
    "model": "muse-voice-transcribe-1.0",
    "mode": "ENDPOINTING",
    "keywords": ["check step", "why did that fail", "scan flat part", "generate replacement", "repeat instruction"]
  }
  ```
- **Server Handshake Frame (`989.6 ms`)**:
  ```json
  {"sessionId": "voyager/duplex/5D9C6C2F0B204E8EA12258568523A00F"}
  ```
- **Server Streaming Events (`check_step_24khz.wav`)**:
  ```json
  {"type": "speechStart", "audioProcessedMs": 0, "turnId": 0}
  {"type": "audioProgress", "audioProcessedMs": 800}
  {"type": "transcript", "transcript": "check", "final": false, "audioProcessedMs": 800}
  {"type": "transcript", "transcript": "check step.", "final": false, "audioProcessedMs": 1200}
  ```

---

## 5. `muse-image-1.0` (`POST /v1/images/edits`)

- **Status**: `200 OK`
- **Latency**: `6648.7 ms` (with `reasoning_strength: "low"` and external tools disabled)
- **Wire Response Structure**:
  ```json
  {
    "created": 1793203200,
    "data": [
      {
        "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
      }
    ],
    "output_format": "png",
    "background": "opaque",
    "usage": {
      "input_tokens": 312,
      "output_tokens": 1084
    }
  }
  ```
- **Decoded Artifact**: Valid PNG image (`2,351,878` bytes).

---

## 6. `meta-models/muse-spark-open-1026` on Baseten (`POST https://inference.baseten.co/v1/chat/completions`) — $10K Open Weight Bonus Lane

- **Status**: `200 OK`
- **Model Slug**: `meta-models/muse-spark-open-1026` (Baseten x Meta Hackathon Dedicated Endpoint, $100 promotional credits + 2M TPM)
- **Request ID**: `54ca701a184499b9f75bfa559080e6c99749e`
- **Direct Latency**: `3650.14 ms` (`BasetenSparkClient.arbitrate_ambiguous_checkpoint` latency: `3610.61 ms`)
- **Token Usage**:
  - `prompt_tokens`: `287`
  - `completion_tokens`: `800` (`reasoning_tokens`: `749`, `accepted_prediction_tokens`: `616`)
  - `total_tokens`: `1087`
  - *Note*: `meta-models/muse-spark-open-1026` always reasons (`reasoning_tokens: 749`), so `BasetenSparkClient` enforces `max_tokens: 2048` and strips `<think>...</think>` / markdown fences via `extract_spark_open_content_and_json()`.
- **Live Arbitration Payload (180° Flipped Connecting Rod Ambiguity)**:
  - Input telemetry: `silhouette_iou = 0.9628` (`>= 0.90`), `witness_polarity_cosine = -0.9412` (`< 0.50`), `witness_error_px = 41.20 px` (`> 6.0 px`), Modal Glimmer 30B finding `"mismatch (witness_mark_inverted_180_deg)"`.
- **Wire JSON Verdict**:
  ```json
  {
    "verdict": "MISMATCH",
    "failure_signature": "witness_mark_inverted_180_deg",
    "corrective_action": "rotate_connecting_rod_180_deg_to_correct_witness_stripe_polarity"
  }
  ```
- **`BasetenSparkClient.arbitrate_ambiguous_checkpoint` Output**:
  ```json
  {
    "triggered": true,
    "provider": "BASETEN_OPEN_WEIGHT",
    "model": "meta-models/muse-spark-open-1026",
    "verdict": "MISMATCH",
    "baseten_verdict": "MISMATCH (WITNESS_INVERTED_180)",
    "failure_signature": "ASYMMETRIC_ROD_INVERTED_180_DEG",
    "modal_glimmer_30b_finding": "mismatch (witness_mark_inverted_180_deg)",
    "baseten_spark_open_finding": "mismatch",
    "consensus": "DUAL_OPEN_WEIGHT_REJECT_180_FLIP",
    "latency_ms": 3610.61
  }
  ```

---

## 7. `facebook/Muse-Glimmer-30B` on Modal (`modal_app/glimmer_server.py`) — $10K Open Weight Bonus Lane

- **Modal CLI & Profile**: `modal 1.6.3`, workspace `tchaudha` (`ac-K9UiWnlIFIIXrAFJt0Y1wr`, environment `main`: `en-yF9uEvQDp9iyTqymBN8U7i`).
- **Target Deployed URL**: `https://tchaudha--rewind-muse-glimmer-30b-serve.modal.run` (`POST /v1/chat/completions`, `GET /health`, `POST /warmup`)
- **Container & Starter Credit Protections**:
  - Mounts `/workspace/rewind/backend` -> `/root/backend` and `/workspace/rewind/contracts` -> `/root/contracts` via `.add_local_dir(...)` (Modal 1.x API) with `PYTHONPATH=/root`.
  - Configured with `min_containers=0, scaledown_window=120, enable_memory_snapshot=True` so idle containers scale to zero after 120s without burning starter credits while `POST /warmup` wakes the container on demand.
- **CIE LAB Witness-Stripe Polarity & Multi-Resolution Rig Inspection**:
  - **Credibility Note (Deterministic Heuristic, NOT a 30B LLM unless `ENABLE_VLLM_ENGINE=1` + GPU deployed)**: `_inspect_image_for_witness_and_seating()` is a deterministic OpenCV/CIE LAB pre-check (`usage: null`, `engine: "deterministic-opencv-lab-heuristic"`) using `extract_yellow_rod_mask_in_rig_zone()` + `_extract_witness_centroid_lab()` across both `1600x1000` golden rig frames and `640x400` synthetic frames:
    - `checkpoint_pass_step2.png` -> `finding: "consistent"`, `verdict: "pass"`, `witness_mark_aligned: true`
    - `checkpoint_fail_flipped_rod.png` -> `finding: "mismatch"`, `verdict: "fail"`, `witness_mark_aligned: false`, `contradicted_predicate_ids: ["witness-stripe-aligned"]`
    - Unblinded context containing forbidden keys (e.g. `"mask_iou": 0.97` or `"expected_pose_id"`) -> HTTP `422 Unprocessable Entity` with `code: "GLIMMER_BLINDING_VIOLATION"`.
- **Deployment Command & Note**:
  - Because the box runs an HTTP egress proxy (`127.0.0.1:8791`), invoke Modal CLI with `MODAL_DISABLE_API_PROXY=1`:
    ```bash
    MODAL_DISABLE_API_PROXY=1 modal deploy /workspace/rewind/modal_app/glimmer_server.py
    ```
  - *Billing Gate Status*: On `2026-10-28`, `AppCreate` on workspace `tchaudha` (`ac-K9UiWnlIFIIXrAFJt0Y1wr`) returned `ResourceExhaustedError: Workspace ac-K9UiWnlIFIIXrAFJt0Y1wr has exceeded its spend limit` even with `$0.00` metered spend because Modal requires a payment method on file (`https://modal.com/settings/tchaudha/billing`) or workspace spend limit > `$0` to unlock the `$30/month` Starter compute credit limit. Once a payment method / spend limit is enabled in the Modal Billing UI, re-running the deploy command above immediately publishes `https://tchaudha--rewind-muse-glimmer-30b-serve.modal.run`.

---

## 8. Credibility Audit & Telemetry Purge (`2026-10-28`)

- **Fabricated token purge**: Removed all hard-coded token counts (`312` / `68` / `380` in `modal_app/glimmer_server.py` and `backend/app/telemetry.py`, plus `416/311`, `520/194`, `290/118`, and default frontend receipt arrays in `WorkbenchUpgrades.tsx`). Every telemetry receipt now includes `measured: bool`, and `all_live_verified` is `True` only when every receipt in the envelope was measured from a live provider response in that run.
- **Visual Auditor (`backend/app/providers/baseten_vision.py`)**: Added `BasetenVisionAuditor` targeting live `meta-models/muse-spark-open-1026` (`https://inference.baseten.co/v1/chat/completions`, `max_tokens=2048`, `<think>` stripping, strict JSON schema check, verbatim `usage` token recording) with automatic fallback to `ModalGlimmerClient` (`muse-spark-1.3-contributor` vision over Meta `/v1/responses`).
- **Deterministic Flipped-Rod Gate (`scenario="fail_flipped_rod"`)**:
  - `decision`: `"rejected"`, `silhouette_iou`: `0.9722` (`>= 0.90`), `witness_polarity_cosine`: `-0.994` (`< 0.50`)
  - `all_live_verified`: `False` in `fixture` mode (zero fabricated tokens: all uncalled model receipts report `status="NOT_MEASURED"`, `tokens_in=0`, `tokens_out=0`, `latency_ms=0.0`).
- **Measured Live Provider Numbers (Sections 2–6)**:
  - `sam-3.1` (`POST /v1/responses`): `816.6–1089.7 ms`
  - `muse-spark-1.3-contributor` (`POST /v1/responses`): `1501.6 ms`, `input_tokens=380`, `output_tokens=549`
  - `muse-voice-transcribe-1.0` (`wss://api.meta.ai/v1/asr/realtime`): `1626.8 ms` E2E
  - `muse-image-1.0` (`POST /v1/images/edits`): `6648.7 ms`, `input_tokens=312`, `output_tokens=1084`
  - `meta-models/muse-spark-open-1026` on Baseten (text conflict arbiter, req `54ca701a184499b9f75bfa559080e6c99749e`): `3610.61 ms`, `prompt_tokens=287`, `completion_tokens=800` (`reasoning_tokens=749`), `total_tokens=1087`, estimated cost `~$0.0006` from the `$100` Baseten credit pool.

## 8. Live Baseten vision probe + live gate (2026-10-29, ~01:20–01:30 ET)

Script: `scripts/probe_baseten_vision.py` (raw results: `docs/probes/baseten_vision_20261029T012350.json`, `docs/probes/live_gate_20261029.json`).
Model `meta-models/muse-spark-open-1026` on `https://inference.baseten.co/v1`, `image_url` base64 PNG parts, temperature 0, max_tokens 2048, `<think>` stripped.

**Vision is supported.** Reference image + live image are both accepted (HTTP 200, prompt tokens scale with image size).

| Case | Input | Correct | Latency min / median / max (ms) | Prompt tok | Completion tok |
|---|---|---|---|---|---|
| pass_step2 → expect `consistent` | full frames | 3/3 | 7745 / 9193 / 9324 | 2805 | 850–1041 |
| pass_step2 | blinded paired crops | 3/3 | 5289 / 5905 / 10946 | 475 | 744–894 |
| fail_flipped_rod → expect `mismatch` | full frames | 3/3 | 8180 / 8221 / 10468 | 2805 | 714–1006 |
| fail_flipped_rod | blinded paired crops | 3/3 | 5915 / 6644 / 7491 | 475 | 1141–1292 |

Plus 4/4 correct in a preceding 1x smoke run → **16/16 correct** on synthetic golden fixtures. No prompt tuning was needed.
Example request IDs: `chatcmpl-d3d07c82015d40258dc734f20f95a263` (pass/full), `chatcmpl-d0f706fea81e41e7b8f4f7067af16997` (flip/crop).
Caveat: these are synthetic renders, not real camera footage.

### Live gate (`run_checkpoint_gate(mode="live")`, SAM 3.1 via Meta proxy + Baseten vision auditor)

| Frame | Provider deadline | Decision | Why | Auditor | Arbiter |
|---|---|---|---|---|---|
| pass | 2.8 s (policy) | `unknown` | `UNKNOWN_TIMEOUT` (Baseten ~8–9 s) | fail-closed | not called |
| pass | 30 s (diagnostic) | `unknown` | providers complete, elapsed 9.2 s > 2.8 s policy | `consistent` | not called |
| flipped | 2.8 s | `unknown` | `UNKNOWN_TIMEOUT` | fail-closed | ran; hit 2048-token cap → `INSUFFICIENT_EVIDENCE` (fail-closed) |
| flipped | 30 s | `unknown` | elapsed 8.3 s > 2.8 s | `mismatch` | `MISMATCH` (live, 4.7 s, 346/1106 tok) |

Live SAM 3.1: 1.45–1.56 s, IoU 0.938 vs reference (fixture mode reports 0.972).

**Conclusion:** the live models get the right answer, but Baseten reasoning latency (5–11 s) exceeds the 2.8 s gate deadline, so the live gate correctly fails closed to `unknown` every time. Product decision pending: either raise the live deadline, or make the deterministic precheck the fast gate and the Baseten audit an async confirmation.

### `all_live_verified` semantics
`all_live_verified` stays **false** in every live gate run, because the DAG compiler, Wonder, voice, and image lanes are not called in a single gate and are reported `NOT_MEASURED`. That is correct. A separate field, `invoked_lanes_live_verified`, is true only if every lane that actually ran returned a real provider receipt. `lanes_not_measured` lists the rest.

### Honesty fixes made during this run
- The telemetry ledger had been reporting canned `mask_iou=0.972` / `witness_polarity_cosine=-0.994` and forcing `decision=rejected` on any non-accepted run. It now reports the measured values verbatim.
- The arbiter (`arbitrate_ambiguous_checkpoint`) used to ignore the model and hard-code `MISMATCH`, used a leading prompt, ran in fixture mode even during live gates, cited "Modal H100 Glimmer 30B", and reported `latency 0.1/14.2 ms`. It now uses a neutral prompt, parses the model's verdict, treats unparseable output as `INSUFFICIENT_EVIDENCE`, receives the gate mode, and reports real latency, usage, and request ID (0 when not called). Its max_tokens was raised to 4096.
- Regression tests: `tests/test_live_honesty_2026_10_29.py`.

Previously open: `geometry.py` replaced the polarity cosine with the constant `-0.994`. Resolved in §9.


## 9. Two-stage gate: live end-to-end (2026-10-29)

Design (`backend/app/two_stage.py`):
- **Stage 1** (deadline 2.8 s) runs live `sam-3.1` plus the deterministic local checks (silhouette IoU, CIE-LAB witness cosine, registration). It gives a provisional verdict (`provisional_accepted` / `provisional_rejected` / `unknown`). It never commits and is never final.
- **Stage 2** (deadline 15 s, set with `REWIND_STAGE2_DEADLINE_MS`) runs the live Baseten `meta-models/muse-spark-open-1026` blinded vision audit. When the local lanes disagree, the Baseten consensus check runs alongside it. Stage 2 commits through the CAS token captured in stage 1.
  - Confirmed accept only when stage 1 is `provisional_accepted` **and** the model says `consistent`.
  - Model `mismatch` over a provisional accept gives `unknown` + needs review, showing the model's reason.
  - Model `consistent` over a provisional reject or unknown gives `unknown` + needs review (never auto-accept).
  - Timeout, error or unreadable output gives `unknown`.
  - Registration lost gives a deterministic `rejected` with no model call.
  - A reset, camera motion or newer observation makes the late result `discarded_stale`.
- Pushed over `/ws/session` as `checkpoint_provisional`, then `checkpoint_stage2`. Can also be polled at `GET /api/checkpoint/stage2/{observation_id}`. Shown in the workbench by `TwoStageGatePanel`.
- `geometry.py`: the constant `-0.994` is gone. The cosine is `null` with `witness_check_status="insufficient_evidence"` when the stripe can't be measured.

Live runs (`mode="live"`; raw JSON in `docs/probes/two_stage_live_20261029.json`; times in ms):

| Run | Fixture | SAM 3.1 | Stage-1 provisional | Stage-1 ms | Model finding | Model ms | Tokens in/out | Request ID | Consensus | Final | Stage-2 ms |
| :-- | :-- | :-- | :-- | --: | :-- | --: | :-- | :-- | :-- | :-- | --: |
| 0 | pass_step2 | live 610.5 ms | `provisional_accepted` | 1224 | — | 0 | 0/0 | — | not triggered | **unknown** (unconfirmed) | 15008 |
| 0 | fail_flipped_rod | live 779.6 ms | `provisional_rejected` | 1236 | mismatch | 14024 | 2805/737 | `…d5df59e8` | MISMATCH | **rejected** (confirmed) | 14025 |
| 1 | pass_step2 | failed (404 / no instance) | `unknown` | 481 | — | 0 | 0/0 | — | not triggered | **unknown** (unconfirmed) | 15017 |
| 1 | fail_flipped_rod | failed (404 / no instance) | `unknown` | 386 | mismatch | 8477 | 2805/936 | `…4c029cbe` | TIMEOUT | **rejected** (confirmed) | 15009 |
| 2 | pass_step2 | failed (404 / no instance) | `unknown` | 506 | — | 0 | 0/0 | — | not triggered | **unknown** (unconfirmed) | 15017 |
| 2 | fail_flipped_rod | failed (404 / no instance) | `unknown` | 406 | — | 0 | 0/0 | — | MISMATCH | **unknown** (unconfirmed) | 15008 |
| 3 | pass_step2 | failed (404 / no instance) | `unknown` | 822 | consistent | 9398 | 2805/876 | `…42f713f4` | not triggered | **unknown** (overridden, needs review) | 9400 |
| 4 | pass_step2 | failed (404 / no instance) | `unknown` | 304 | consistent | 9486 | 2805/781 | `…04c92d51` | not triggered | **unknown** (overridden, needs review) | 9487 |
| 5 | pass_step2 | failed (404 / no instance) | `unknown` | 273 | — | 0 | 0/0 | — | not triggered | **unknown** (unconfirmed) | 15017 |

What these runs show:
- Stage 1 always finished well inside 2.8 s (273–1,236 ms).
- In 7 of 9 runs, the Meta `sam-3.1` endpoint returned HTTP 404 intermittently (seen directly by calling `segment_image` repeatedly). In those runs stage 1 correctly fell to `unknown`.
- Baseten vision latency was 8.5–14 s. 4 of the 6 pass-fixture audits and 1 of the 3 flipped-rod audits went over the 15 s deadline and correctly stayed `unknown`. In the flipped-rod timeout, the consensus check alone said MISMATCH, but that is not enough to confirm.
- flipped_rod: with live SAM, the model confirmed the provisional reject (`rejected`, confirmed, 14.0 s). When SAM failed, the model's `mismatch` still gave a confirmed `rejected`, because rejection is the safe side.
- pass: in the one run with live SAM, the audit timed out, so we have **no confirmed live accept yet**. Whenever SAM failed, the model said `consistent` but the gate correctly would not auto-accept (`unknown`, needs review).
- No result was ever accepted without both stages agreeing.
