# REWIND: 3-Minute Demo Script (Flat-Pack Reframe)

Layout: overhead phone feed of the hands (large), with the REWIND workbench as a side panel. Captions are always on. Every number shown on screen must come from a real recorded run; leave a placeholder (`[measured]`) until then.

**0:00–0:10 · Cold open on the catch**
Hands place the side panel on the frame, cam-lock holes facing out. "Check step three." The overlay shows the outline match (`IoU [measured]`, green), then the gate turns RED: "REJECT: panel reversed, holes facing out." The paired crops show the reference and live holes side by side.

**0:10–0:20 · Impact line**
VO: "Most furniture mistakes look 95% right, and you find out ten steps later. REWIND watches your hands and says stop."

**0:20–0:45 · The problem**
Quick shots: a manual, a pile of parts, a person squinting. VO: "An outline check would pass this panel; it's the same rectangle. What matters is which face the holes are on."

**0:45–1:30 · How it works (Meta model callouts, one line each)**
- "I just talk to it." **Muse Voice Transcribe** turns "check step three" into a gate call.
- **SAM 3.1** segments the "wooden panel" from a text prompt.
- A deterministic local check compares the hole pattern and finished-edge orientation. This is **stage 1: a provisional verdict in about 1.3 s**.
- **Muse Spark Open on Baseten** does a blinded visual audit of the crops. This is **stage 2: it confirms in about 5–7 s**, shown with a live timer. VO: "The fast check answers instantly; the model confirms a few seconds later."
- **Muse Image** draws the repair card: "Flip the panel, holes facing in."

**1:30–1:55 · The fix**
Hands flip the panel. "Check step three." Stage 1 is green and provisional; stage 2 confirms PASS. **Muse Wonder** opens the assembly manual page and ticks step 3 (real computer-use action, shown on screen).

**1:55–2:15 · Security line**
VO: "I come from security, so REWIND fails closed. If the camera's blocked, the model times out, or the evidence is stale, the step does not pass. The auditor model is blinded to what we expect, and an outside red-team audit is in the repo." On screen: a hand covers the camera, and the gate shows UNKNOWN, not PASS.

**2:15–2:30 · CAD beat (optional; cut if over time)**
"Lost a bracket? Lay the flat part on the mat and REWIND exports an STL to print a replacement." Show the 3D preview for about 5 s. Make no accuracy claim.

**2:30–2:50 · Honest results**
A card showing real takes: "Reversed panel caught [x/5]. Correct panel passed [y/5]. Stage-2 median [measured] s." Then a one-line table of live vs. deterministic components.

**2:50–3:00 · Close**
"REWIND: a second pair of eyes for anyone building or fixing things. Built on Meta's models." The repo URL is on screen.

Editing notes: keep the cold open under 10 s; don't speed up stage-2 waits without labelling the cut ("sped up 2×"); keep the voiceover to about 330 words.
