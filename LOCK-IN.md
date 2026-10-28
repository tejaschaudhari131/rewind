# REWIND: final architecture lock

Decision: constrained fusion, not the full Candidate C proposal.

Hook: **Watch a mechanism come apart. REWIND checks the rebuild and recreates one missing flat link from measured evidence.**

This package specifies a proposed build. It does not establish hardware availability,
segmentation accuracy, measured latency, model access, physical fit or award eligibility.

## Hardware and supported claim

Use an openly disclosed hand-cranked slider-crank inspection rig on a 400 x 250 mm matte
black base. Seven tracked modules: base, green crank support (about 60 x 60 mm), red crank
disk (100 mm diameter, nominal radius 30 mm), yellow connecting rod (150 mm hole-center
distance, 25 mm width, known 4 mm stock thickness, nominal 6.5 mm holes), blue slider
(about 60 x 40 mm), silver/gray guide assembly (about 220 x 70 mm), purple output paddle
(about 50 x 40 mm). Fasteners are not claimed as separately segmented components.

Use actual mechanical hardware and matte contrasting surfaces, captive hand-removable
fasteners and repeatable locating features. Dimensions above are proposed design dimensions,
not a verified catalog product. Fabrication access and a working rig on day one are prerequisites.
Not every module is visible in every pose. Hidden components remain unknown until observed.

The camera learns the observed teardown sequence, measures flat geometry and monitors
visible reconstruction. Mechanism family and part template are preset and disclosed.
Nominal zero-offset motion is x(theta)=r*cos(theta)+sqrt(L^2-r^2*sin(theta)^2), giving 60 mm
stroke for r=30 mm. Scan the removed rod flush on the calibrated plane. Thickness, pin diameter
and fabrication clearance are independently supplied; no inferred strength/material properties.

## Scope lock

Required: capture and source ledger, SAM tracking, Wonder step compiler, local metrology,
Glimmer visual audit, versioned gate, Voice control, deterministic STEP/STL export of one flat
two-hole capsule link, four-pane workbench, held-out trials.

Optional asynchronous lanes: Wonder computer use for CAD inspection; Muse Image explanatory
repair card; Spark Open review of ambiguous sessions; Spark Contributor short summaries.
None may authorize a checkpoint or enter its latency-critical path.

Cut: arbitrary 3D reconstruction, inferred stress/strength, unqualified millimeter overlays on
elevated parts, universal IoU threshold, mandatory LoRA, a second required open-model pass,
URDF/MuJoCo as a core deliverable, arbitrary model-written CAD execution, live fabrication claims.

Ghost targets are original reference pixels/masks or deterministic CAD projections. Red/cyan
is measured disagreement, not stress. STEP/STL plus parameter JSON and CadQuery template source
are exported; STEP/STL do not preserve the source feature history.

## Request lanes

1. Record 10-15 seconds of slow articulation and a complete disassembly. Submit short H.264
   clips to SAM. Decode masks into the local contract. Compile conservative reverse-order
   steps with Wonder, linking every instruction to source states. Freeze targets and policy.
2. Record the rod flat on the registered mat. Measure the profile and hole centers, enter
   independently measured thickness/pin/clearance, and create a validated measurement record.
3. At each stationary checkpoint, increment observation ID and freeze frame, scene/plan/step
   versions and reference/calibration/policy hashes. Run SAM image request and Glimmer audit
   concurrently; Glimmer receives original frames and a neutral objective, not planner verdicts.
4. Local registration/marker/witness checks can reject early. Fresh SAM masks permit full
   contour checks. The gate accepts only with all required complete, current evidence. It uses
   an atomic compare-and-set. A changed scene invalidates pending work and prior green state.
5. Wonder receives the decision and chooses recovery or next action. Already compiled next-step
   instructions display immediately after a pass. New long-form reasoning is not required per pass.
6. On missing-link request, coordinator resolves an eligible immutable measurement record.
   A deterministic template generates the solid; validation checks dimensions and re-imported
   export. A valid solid remains a candidate until a matching physical specimen is tested.

## Timing targets, not measurements

| Time from check request | Work |
|---|---|
| 0-0.05 s | UI acknowledgment |
| 0-0.40 s | Stability window and fresh capture |
| 0.40-0.65 s | Local measurements; early reject or amber provisional match |
| 0.40-2.50 s | SAM image and Glimmer audit in parallel, unproven provider budget |
| up to 0.10 s after SAM | Full mask comparison |
| by 2.70 s | Complete atomic decision and UI update if all evidence is available |
| 2.80 s hard deadline | UNKNOWN_TIMEOUT if incomplete; no green and no advancement |

Warm one vision-capable Glimmer replica before trials. Benchmark complete checks, not just
token generation. If p95/p99 do not meet the target, relax verified-check timing or explicitly
offer fast measurements-only status. Provider documentation cannot promise every pass under 3 s.

## Four-pane workbench

At a 1600 x 900 desktop viewport: 56 px header, 40 px footer, 16 px padding and 16 px gutter.
The 1552 px content width splits 60/40; available pane height splits 58/42.

- Top left: LIVE CAMERA. Actual feed with deterministic contours, source-grounded cyan target,
  red discrepancies, part labels and scene revision. Freeze the analyzed frame explicitly.
- Top right: ORIGINAL EVIDENCE. Before/current paired crops, source clip timestamp and step.
  A source thumbnail opens the actual recorded frame. Label live, recorded and generated media.
- Bottom left: MECHANISM + PART. Analytic kinematic twin, measured trajectory and replacement
  preview; shows measured dimensions/provenance and pixel residuals. Export STEP/STL and source.
- Bottom right: ACTION + CHECKS. One next action, named failed predicates, pending auditor,
  elapsed time, clear accepted/rejected/unknown state, Check, Show reference and mic controls.

Header: session, mode (Record/Rebuild/Replacement), camera registration, capture health.
Footer: evidence age, model job states, current observation ID, compute cost. No reasoning transcript.
Color plus text/icons; pending is amber, observed pass green, mismatch red, unknown neutral.
Generated image cards open after completion and never replace evidence panes.

## Planned application repository (not implemented by this artifact package)

```text
rewind/
  README.md
  LICENSE
  .env.example
  pyproject.toml
  contracts/
    contract.schema.json
    glimmer-output.schema.json
    wonder-output.schema.json
    example-cycle.json
  backend/rewind/
    app.py
    config.py
    store.py
    schemas.py
    coordinator.py
    gate.py
    planner.py
    auditor.py
    providers/
      meta_responses.py
      sam.py
      glimmer.py
      voice.py
      baseten.py
      image.py
    vision/
      calibration.py
      sam_decode.py
      association.py
      measurements.py
      kinematics.py
    cad/
      flat_two_hole_link.py
      validate_export.py
    optional/
      cua_inspector.py
  frontend/
    package.json
    src/
      App.tsx
      state.ts
      api.ts
      panes/
        LiveCamera.tsx
        OriginalEvidence.tsx
        MechanismAndPart.tsx
        ActionAndChecks.tsx
  deploy/
    modal_glimmer.py
    compose.yaml
  fixtures/
    raw/
    development/
    holdout/
  tests/
    test_contracts.py
    test_calibration_planes.py
    test_sam_identity.py
    test_gate_rejects_stale.py
    test_cad_roundtrip.py
  eval/
    checkpoint_latency.py
    false_accepts.py
    motion_holdout.py
    report.md
  docs/
    hardware.md
    claims-and-limits.md
    demo-runbook.md
```

Use one application service with durable session state and background jobs, plus a separate
CAD worker and GPU inference endpoint. Do not build a fleet of application microservices.

## Launch gates

- October 28: physical rig works, reversible rod removal, real SAM clips and vision Glimmer calls.
- By October 30: reject wrong rod, wrong position, partial engagement and obscured evidence.
- Before freeze: camera-plane error measured, held-out calibration/motion recordings separated
  from fitting, a second rod length changes the exported dimensions, and physical specimen claim
  is tied to its exact artifact revision. Measure warm latency and real false-accept/unknown rates.
- Freeze features before final recording. Deadline remains November 4, 9 PM Eastern.

Do not invent 100% accuracy, instant printing, a 20-minute training guarantee or a stress model.
The supported claim is an observed rebuild plus a measured replacement candidate for this rig.
