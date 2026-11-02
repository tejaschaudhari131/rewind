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

## How to film with REWIND on your phone

1. **Serve it.** On a machine with normal internet access, run `scripts/serve_phone.sh`. It starts the server, opens a Cloudflare quick tunnel (HTTPS, which phone cameras need) and prints the `https://….trycloudflare.com` URL. The access token is in `.phone_token` (chmod 600, gitignored). Open `URL/?t=<token>` once; a cookie keeps you signed in for 7 days. Keep the computer running the Meta egress proxy (`127.0.0.1:8791`) online. If it drops, a red "Meta API unreachable" banner appears.
2. **Pick the input.** Choose **Live camera** (back camera; the page shows LIVE CAMERA) or **Upload photo / video** (JPG/PNG/HEIC ≤ 8 MB, or an MP4/MOV you can scrub to a frame; the page shows UPLOADED PHOTO or UPLOADED VIDEO FRAME). For the clean filming view, open `URL/?t=<token>&look=b&record=1&source=camera`.
3. **Check.** Frame the panel the way the manual's reference does, then tap **Check this step** (or press space on a laptop). There is also an optional, labelled "auto-check when still ~1s" mode. Frames are deleted after the check unless you tick **save to eval set**; saved frames go to `fixtures/real_footage/` with a timestamp (gitignored, never commit them).

The laptop webcam works through the same URL, or through `http://localhost:8077` (browsers treat localhost as secure).
