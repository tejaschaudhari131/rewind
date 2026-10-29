# Response to external audit (2026-10-29, written against bc7cd2b)

Fixes landed on `main` on top of f251197. The regression tests are in `tests/test_external_audit_2026_10_29.py` (53 cases). Every harness probe is asserted to be rejected or fail closed. Full suite: 95 passed.

| ID | Finding | Status | Commit(s) | Notes |
|---|---|---|---|---|
| A1 | Movement does not invalidate pending authorization | **Fixed (partial)** | ca5a3ff | A movement or registration loss bumps the generation and invalidates in-flight tokens, so the commit is rejected as STALE_OBSERVATION. Stationarity is only measured when the caller supplies `motion_score`. |
| A2 | Reset publishes stale result | **Fixed** | ca5a3ff, 7178ef6 | Reset calls `retire()` first, and the random-UUID session plus the generation CAS reject late results. |
| A3 | Voice bypasses step requirements | **Fixed** | 7178ef6 | Negated utterances are unrecognized. Voice `check_step` returns `unknown/no_live_frame_for_voice_check` and never runs on the pass fixture. |
| A4 | Schema-valid ≠ authenticated evidence | **Fixed (scope-limited)** | 99c6962 | An accepted gate requires a consistent geometry, audit and segmentation chain in the same session. There is no cryptographic signing yet. |
| B1 | Fabricated calibration and holdout | **Fixed (honesty) / Deferred (real capture)** | 5d46d49 | The envelope status is `unvalidated`, and the placeholders are labelled. Real intrinsics and holdout capture are deferred because they need a physical rig. |
| B2 | Top-of-part parallax vs ±0.25 mm | **Fixed** | 99c6962 | Adds `projection.py` (height-plane projection with behind-camera rejection). The ±0.25 mm claim was removed from the README. |
| B3 | Letter target not 1:1 | **Fixed** | 5d46d49 | Four overlapping letter tiles at 1:1, each with a 100 mm ruler. Tests check the PDF ppi. |
| B4 | Fixture heuristics ≠ physical seating | **Partially fixed / Deferred** | 99c6962, 7178ef6 | SAM is now authoritative, an empty or ambiguous mask gives unknown, user-supplied dimensions are labelled `manual_caliper` with null uncertainty, and frame paths are confined. A full pipeline rework (hands_clear, a seating model) is deferred. |
| B5 | Invalid STEP topology, self-referential verification | **Fixed (honest faceted)** | c285ddf | FACETED_BREP with a PLANE per face, plus an independent topology checker. Tamper tests cover dropped faces, dangling refs and moved vertices. An analytic CadQuery kernel is deferred. |
| C1 | Modal does not run Glimmer | **Fixed (fail-closed) / Deferred (real model)** | 99c6962 | Missing, blank, small or undecodable input returns unknown, and the heuristic is labelled NOT_MEASURED. Real 30B inference is deferred because of the Modal billing gate. |
| C2 | Deadline rewritten, not enforced | **Fixed** | ca5a3ff | `deadline.py` enforces a real wall-clock deadline: a late provider gives UNKNOWN_TIMEOUT, and the simulated elapsed time is not clamped. |
| C3 | SAM semantic use fails open | **Fixed** | 99c6962 | Adds SamDecodeError. Empty, malformed or out-of-image output and None confidence all fail closed. The box prompt is validated. |
| C4 | Independence lost after blinding guard | **Fixed** | 99c6962 | A recursive allowlist sanitizer strips nested claim and verdict fields. |
| C5 | Voice and bonus lanes overstate execution | **Fixed** | 7178ef6 | The CUA status is `cua_actions_returned`/`no_cua_actions` with `measured: false`. Live voice uses the 24 kHz path. The README table was updated. |
| D1 | Teardown→reassembly flow not implemented | **Deferred** | — | Product and UI scope. Not attempted in this pass. |
| D2 | Workbench lacks four containers | **Deferred** | — | UI scope. Not attempted in this pass. |

Other deferrals: speechComplete gating inside the ASR loop, and multi-worker DB persistence.
