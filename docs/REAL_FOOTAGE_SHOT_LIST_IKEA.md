# Real Footage Shot List: Flat-Pack Panel (solo, target Nov 2)

## Kit
- The furniture piece with a side panel, partly disassembled (holes on one face, finished front edge).
- Painter's tape (one colored strip on the finished edge, for a backup witness).
- AprilTag mat tiles printed at 100% (`fixtures/printable/rewind_apriltag_mat_letter_tiles.pdf`). Check each tile's 100 mm ruler.
- A phone on an overhead arm, or taped to a shelf or stool about 80–100 cm above the floor. Use a second phone or a screen recorder for the workbench.
- Two lamps. Overhead room light is not enough.

## Setup
1. Work on the floor or a table with a plain, non-white background (a dark sheet), so white panels contrast.
2. Tape the 4 mat tiles at the corners of the work area, flat and fully visible in the frame. The panel sits between or on them.
3. Mount the phone straight down, landscape, 1080p 30 or 60 fps. Frame the whole panel plus all tiles.
4. Lighting: one lamp at about 45° from the side, so the holes cast shadows and read as dark circles, and a second lamp diffused opposite. No direct glare on the panel.
5. Lock exposure and focus: on iPhone, long-press until "AE/AF LOCK" appears; on Android, use Pro mode or tap-and-hold to lock. Turn off HDR and auto-macro.
6. Take reference stills (one per step) with the panel correctly placed and holes facing in. These become the per-step reference photos.

## Pre-flight (before every session)
- [ ] Tejas's computer is **online and awake** (sleep disabled); the Meta egress tunnel depends on it.
- [ ] `python scripts/preflight_live.py` exits 0 (Meta model list includes sam-3.1 and muse-spark; Baseten Spark Open is listed).
- [ ] The REWIND backend and frontend are running; a single test "check step" returns.
- [ ] Phone battery is over 60%, storage is free, and Do Not Disturb is on.
- [ ] Exposure and focus locked; tiles are all in frame.
- [ ] Clap once at the start of each take (an audio sync mark).

## Takes (about 2 hours total)
| # | Take | Spoken | Expected |
|---|---|---|---|
| 1 | Reference stills: empty mat, then correct panel per step | none | Reference set |
| 2 | **Reversed panel** (holes facing out), placed slowly | "Check step three." | REJECT (the cold open) |
| 3 | Flip the panel to correct, holes facing in | "Check step three." | PASS, then the Wonder tick |
| 4 | Repeat takes 2 and 3 **five times each** with slight position changes | same | Results table (count every outcome honestly) |
| 5 | Hand covers the camera mid-check | "Check step three." | UNKNOWN / blocked (security beat) |
| 6 | Backup: drawer front upside down, then corrected | "Check step five." | REJECT, then PASS |
| 7 | B-roll: parts pile, manual page, hands close-up, a frustrated look | none | Editing cutaways |
| 8 | Optional: a flat bracket on the tiles for the CAD beat | none | STL preview |

## After filming
Send the raw clips (don't trim them) and the screen recordings. Note any take where the Meta API was down; count it as a failed run, and don't drop it.
