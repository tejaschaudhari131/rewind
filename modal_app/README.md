# REWIND — Modal `Muse Glimmer 30B` Blinded Visual Verifier (`modal_app/`)

This directory contains the production Modal GPU deployment script (`glimmer_server.py`) for the open-weight **`Muse Glimmer 30B`** (`meta-models/Muse-Glimmer-30B`) blinded visual verifier on workspace **`tchaudha`**.

## Key Architecture & Guarantees
1. **Warm H100 Container Pool**:
   - `@app.cls(gpu="H100", enable_memory_snapshot=True, min_containers=1, scaledown_window=300)`
   - Mounts persistent Hugging Face weight volume `rewind-glimmer-weights` at `/cache/huggingface` so cold starts never re-download weights.
2. **Strict Input Blinding (`assert_glimmer_input_blinded`)**:
   - Rejects any request whose context leaks upstream SAM or Geometry fields (`geometry`, `metrics`, `sam`, `segmentation`, `masks`, `iou`, `gate_decision`, `expected_pose_id`, `expected_step`) with HTTP `422 GLIMMER_BLINDING_VIOLATION`.
3. **Guided Decoding & Schema Validation**:
   - Validates every model output against `contracts/glimmer-output.schema.json` and returns both the OpenAI-compatible `choices[0].message.content` JSON string and `glimmer_verdict` (`verdict`, `confidence`, `observed_part_id`, `witness_mark_aligned`, `seating_flush`, `failure_mode`, `notes`).

## Deployment Commands
```bash
# 1. Dry-run syntax & schema check
python3 modal_app/glimmer_server.py --check-syntax

# 2. Deploy to Modal workspace `tchaudha`
MODAL_PROFILE=tchaudha modal deploy modal_app/glimmer_server.py
```
