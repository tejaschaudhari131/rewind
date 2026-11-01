# REWIND UI v2: inspiration and takeaways (researched 2026-10-31)

Sources below are pages actually opened/read during research. Takeaway → what REWIND v2 does with it.

| Source | What they do well | REWIND v2 takeaway |
|---|---|---|
| BILT Intelligent Instructions ([App Store](https://apps.apple.com/us/app/bilt-intelligent-instructions/id879452214), [FAQ](https://bilt.ai/frequently-asked-questions/)) | One clear step at a time; step count, replay any step; voice + text + visual | Left **step rail** (8 steps, current step armed); one step's title in large type; check history you can scrub. |
| Duolingo feedback design ([DESIGN.md breakdown](https://github.com/Meliwat/awesome-ios-design-md/blob/main/design-md/misc/duolingo/DESIGN.md), [Gamification guide](https://blakecrosley.com/guides/design/duolingo)) | Green pop on correct, red shake on wrong, always show the *correct solution*, sound + haptic paired, chunky 3D "ledge" buttons | **STOP shakes, OK pops**, color flash, optional WebAudio tone + `navigator.vibrate`; **"Manual expects / Camera sees"** side-by-side is our "correct solution" reveal; ledge-style Check button. |
| LandingLens predictions ([View Predictions](https://landinglens.docs.landing.ai/view-predictions)) | Overlays on the image; green check / red X on match vs label; toggle predictions | **Detected-hole markers** drawn on the frame from the witness's real pixel coordinates; overlay toggle. |
| Cognex In-Sight Web HMI ([Web HMI](https://www.cognex.com/products/machine-vision/vision-sensors/in-sight-2000-vision-sensors/web-hmi)) | Operator pass/fail at a glance; filmstrip of recent inspections | Giant one-word verdict readable at 1080p; **"Rewind" filmstrip** of this session's real checks. |
| Project Astra ([DeepMind](https://deepmind.google/models/project-astra/)) & Meta Ray-Ban Live AI ([Meta help](https://www.meta.com/help/ai-glasses/894093646030348/)) | Camera is continuously "watching"; assistant speaks up at the right moment | Hero is the camera; CHECKING scan line while models run; REWIND only interrupts to say STOP. |

## Unique ideas REWIND owns
1. **"Caught at step 3 — steps 4–8 would be built on top of it."** The STOP card states the cost of the mistake (labelled DECLARED, derived from step count) — inspection UIs say *fail*, we say *what it would have cost you*.
2. **An honest model trace as a UI element.** Each Meta lane (SAM 3.1, Muse Spark Open, Muse Wonder) shows a time only if it was MEASURED this run; otherwise NOT_MEASURED. Trust is the feature.
3. **The manual ticks itself — only when it's true.** Muse Wonder's real click on the checklist, verified in the DOM, is the reward moment; in fixture mode it visibly does not tick.
