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
