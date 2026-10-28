# REWIND Round 2: contract authority and runtime rules

These are design artifacts, not an implemented or hardware-tested application.
`contract.schema.json` is the complete local JSON Schema (Draft 2020-12).
`glimmer-output.schema.json` and `wonder-output.schema.json` are compact model-output contracts.
`example-cycle.json` is synthetic: its success, latency, hashes and media paths are not observations.

## Stage ownership

| Payload | Authorized producer | Purpose |
|---|---|---|
| calibration | calibration_service | Frozen camera parameters and calibrated planes |
| measurement_record | measurement_service | Provenanced link dimensions and eligibility |
| segmentation | sam_adapter | Decoded full-frame masks and response-scoped track IDs |
| geometry | geometry_service | Measured evidence and predicate results |
| visual_audit | glimmer_adapter | Blinded visual consistency assessment |
| planner_proposal | wonder_adapter | Proposed next action |
| gate_decision | gate_service | Sole authority to advance a checkpoint |
| cad_request | coordinator | Approved template request from a validated measurement record |
| cad_artifact | cad_worker | Geometry validation and exported artifacts |

Models return only their compact payloads. The trusted adapter supplies the envelope;
never copy authority, producer, versions, hashes or timestamps from model-generated text.
Producer strings and hashes are not authentication. Enforce ownership in application code,
read-only reference/policy mounts, restricted tools and an atomic state update.

## Coordinate contract

Archive raw camera images, then undistort using frozen camera parameters. Feed the same
undistorted image coordinate system to SAM, marker detection and the comparison UI.
Image coordinates have top-left origin, x right, y down, in pixels. Boxes are inclusive xyxy.
Masks are decoded to full-frame 0/255 binary PNG with exactly the source image dimensions.
Keep each original provider payload and raw image as evidence; the normalized contract is an adapter output.

`H_undistorted_px_to_plane_mm` maps homogeneous undistorted pixels to coordinates on the
named physical plane, followed by homogeneous normalization. Its origin and axes are
recorded explicitly. A mat-plane transform is invalid for elevated link faces or pivots.
The MVP reports assembly errors in pixels. Only flat scans on `mat_z0` receive millimeter dimensions.
Camera movement invalidates calibration until reacquired. Never independently align each part
to improve its IoU: use the fixed registration, otherwise a wrong placement can look correct.

## Required semantic checks beyond JSON Schema

1. All referenced IDs and hashes must resolve to the immutable evidence store. Verify file hashes;
   reject model-invented references, future timestamps, unexpected dimensions and out-of-bounds regions.
2. Every result must match observation ID/hash, scene version, plan version, step ID, reference,
   calibration and policy currently in force. A new movement, tracking loss, camera shift or user
   correction invalidates pending checks and clears green UI state. Do not increment on every video frame.
3. A SAM track is identified by `(provider_response_id, sam_track_id)`. Cross-clip identity is a
   separate association decision; do not inherit it from coincident numeric IDs.
4. A millimeter metric requires a non-null plane ID and an applicable validated plane, with
   demonstrated coplanarity or implemented calibrated 3D projection. Pixel metrics do not establish
   millimeter accuracy. IoU and visibility are bounded [0,1]; model confidence is not physical uncertainty.
5. The gate may accept only when fresh segmentation, geometry and required audit are complete,
   all mandatory predicates pass, Glimmer says consistent without contradiction, and the current
   scene is still stable. Unknown uncertainty or hidden engagement cannot become a passed predicate.
6. Glimmer sees original reference/current frames and the neutral check objective. It does not
   see Wonder's success claim or the geometry pass/fail label. It may challenge evidence; it cannot
   change thresholds. Incomplete output, malformed JSON, wrong references or timeout means unknown.
7. Atomically compare `(scene_version, plan_version, step_id, observation_id)` before committing.
   Late or timed-out jobs remain diagnostic records and cannot authorize a newer observation.
8. Planner `finish` is a proposal. The controller independently verifies that every required
   step and supported motion test completed. `generate_supported_part` requires an eligible
   measurement record ID; the controller fetches dimensions itself, not from prose.
9. CAD requests must match the frozen measurement record and allowlisted template hash.
   Hole diameter = pin diameter + diametral clearance; end radius = outer width / 2.
   Check positive wall thickness, valid solid, hole centers, thickness, extents, volume and re-import.
   Parameters must fit the declared supported rig envelope. Numerical bounds are chosen and tested
   with the hardware; they are not universal material or strength limits.
10. `physical_fit_status=tested_on_this_rig` requires a matching physical test record for the
    exact artifact/parameter revision. An earlier print cannot substantiate a newly changed design.

## Timing and state

`MOVING -> CAPTURING -> CHECKING -> ACCEPTED | REJECTED | UNKNOWN`.
Immediate UI acknowledgment is a target under 100 ms; local measurements target under 0.9 s.
At 2.8 s from the check request, missing required evidence yields UNKNOWN_TIMEOUT; preserve
the current step, cancel acceptance for that job, and offer a new capture. An amber local match
is never called audited or verified. A fully audited sub-three-second p95 is an unmeasured target.
No official provider guarantee was found. Benchmark complete warm runs before making that claim.

## Validation performed on these artifacts

The generator validates the schema, seven synthetic stage messages, compact model outputs,
and rejects extra fields, an unauthorized producer label, an IoU with millimeter units,
and an audit payload attempting to set `accepted`. These are structural checks only;
the semantic checks above still require application implementation and meaningful tests.
