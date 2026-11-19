# Badminton Form Coach

Computer-vision badminton training. Pick a drill: **smash**, **clear** or **drop** form, or **6-point footwork**.
Works on uploaded video **and** live webcam, and saves every rep so you can track your best rep and best session.

```
camera / video -> MediaPipe pose -> swing detection -> joint-angle checks -> coaching feedback + score
```

## Features
- **Video analysis:** upload a clip, get an annotated video (skeleton, live elbow angle, PREP / CONTACT labels), a 0-100 form score, written fixes, key-frame stills, and wrist-speed / elbow-angle plots.
- **Live mode:** get ready (an **ideal-stroke shadow** loops in the corner so you know what to aim for) -> smash -> a result card with a **letter grade, per-metric bars and the top fix**, next to a **slow-motion replay of your swing with the target shadow following you** -> click **REDO** (or press `r`) to try again. Press `v` (or click the replay) to switch to the 8 snapshots, then the key moments. Tracks attempt count and best score.
- **Smash benchmark:** 8 metrics, each scored 0-100 against a benchmark (60 = pass line, 100 = benchmark), weighted into a score and an A-F grade.
- **Whole-swing assessment:** the swing is judged across the whole movement, not from one frame. 8 snapshots spread from the load to the finish (always including the 4 key moments) are each compared with a target path, and their average is the **Whole-swing match** metric. See [Whole-swing match](#whole-swing-match).
- **Tunable thresholds:** all form targets are sliders in the web UI (and fields in `analysis.Config`).
- **Racket arm:** auto-detected (or set manually).
- **Swing breakdown + shadow:** a translucent white **shadow** of the target form, made from your own skeleton (same limb lengths, same spot) moved to the benchmark, is drawn on every frame of the swing, so it follows you through the replay and the annotated video. The 4 key moments (**load, trophy, contact, finish**) also get yellow arrows from your joints to where they should be and the numbers for that moment (e.g. `elbow 106 -> 170 deg`). Shown in the live result card and under the video analysis.
- **Clear and drop:** the same live/video flow as the smash, graded against stroke-specific targets. A drop is scored on control: a swing that's too fast, or a big smash-style follow-through, loses points. See [Clear and drop](#clear-and-drop).
- **6-point footwork:** mark your training area once; the coach calls corners (on screen + spoken), scores every rep on reaction, movement, lunge and recovery, and shows a session card after 12 reps. See [6-point footwork](#6-point-footwork).
- **Progress tracking:** every rep is saved to `data/history.json`. Result cards show your session average vs your best and last session and flag a new best rep; the web UI has a **Progress** tab with charts.

## Quick start
Requires Python 3.10+ and a webcam for live mode.

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # web UI: "Analyze a video" + "Live webcam" tabs
python live.py                   # native live window (most reliable for demos); r = redo, q = quit
python live.py --lite            # lighter pose model: higher frame rate, slightly less precise
python live.py clip.mp4          # replay a clip in live mode
python live.py --drill clear                    # or --drill drop
python live.py --drill footwork                 # 6-point footwork, full singles half court (5.2 x 6.7 m)
python live.py --drill footwork --court 3x3     # ...in a smaller area (width x depth, metres)
# in the live window: d = next drill (smash -> clear -> drop -> footwork), r = redo / new session, v = result view, c = re-mark court, q = quit
python test_footwork.py          # footwork + history tests (no video needed)
python test_analysis.py          # logic tests on synthetic skeletons (no video needed)
```

The pose model (`models/pose_landmarker_full.task`, ~9 MB) is **not stored in git**; it downloads automatically on first run (needs internet once). For an offline demo, run the app once beforehand.

## Smash benchmark
**The benchmark numbers are demo placeholders, not measured from real players.** They live in `analysis.Config` (`min_*`/`max_*` = pass line, `bench_*` = full marks).

| Metric | Measure | Pass line | Benchmark | Weight |
|---|---|---|---|---|
| Racket-hand speed | peak wrist speed, torso-lengths / s | >= 7 | 14 | 25% |
| Arm extension at contact | elbow angle | >= 150 deg | 170 deg | 15% |
| Contact height | wrist above shoulder, torso lengths | >= 0.60 | 0.95 | 15% |
| Downward follow-through | wrist drop below shoulder within 0.5 s after contact | >= 0.0 | 0.6 | 10% |
| Preparation elbow height ("trophy") | elbow vs shoulder, torso lengths | >= -0.15 | 0.10 | 10% |
| Knee loading before swing | min knee angle | <= 165 deg | 140 deg | 10% |
| Swing tempo | prep -> contact time | <= 0.45 s | 0.20 s | 5% |
| Whole-swing match | average match of 8 snapshots to the target path | >= 70% | 92% | 10% |

Grade: A >= 90, B >= 80, C >= 70, D >= 60, else F. "Top fix" = the failed metric costing the most weighted points.

Swing = peak wrist speed. Contact = highest wrist point around that peak. Prep = most-flexed elbow in the 0.7 s before contact.

## Whole-swing match
The checks above each look at a moment in the swing. The **target path** covers every frame from the load to the finish:
1. At each key moment the benchmark sets a band for 4 body measures: knee angle, elbow height, hand height and elbow angle (e.g. contact: elbow >= 170 deg, hand >= 0.95 torso above the shoulder; trophy: elbow at shoulder height or above, elbow bent, hand up by the head).
2. Where you are outside a band, that is a correction. Corrections are blended between key moments and fade out over 0.2 s, so the target is **your own movement, shifted by what each part of the swing needed**. That is the shadow you see following you.
3. 8 snapshots spread evenly across the swing (snapped so the 4 key moments are among them) are each scored 0-100% by how far you are from the target. The average is the Whole-swing match. Each snapshot also names the measure that's furthest off (e.g. `elbow 106->170`).

Same caveat as the benchmarks: this is a placeholder target built from coaching heuristics (`analysis.path_bands`), not a path recorded from real players.

## Clear and drop
Same swing detection as the smash; only the targets change (`analysis.stroke_metrics`). Placeholder numbers, as for the smash. Contact position needs a **side-on** camera: it uses the nose to tell which way you face. If that can't be read (e.g. you face the camera), it isn't scored.

| Metric | Clear target | Drop target |
|---|---|---|
| Racket-hand speed (torso/s) | pass 6, full marks 11 (20%) | **sweet spot 6-8.5**, pass 4.5-10; too fast is a FIX (20%) |
| Arm extension at contact | as smash (15%) | as smash (10%) |
| Contact height (torso) | pass 0.65, full 1.0 (15%) | pass 0.60, full 0.95 (10%) |
| Contact position, + = in front (torso) | sweet spot -0.05 to 0.20, i.e. above the head (10%) | in front: pass 0.10, full 0.35 (15%) |
| Follow-through (wrist drop below shoulder) | full: pass 0.0, full 0.4 (10%) | **controlled**: pass <= 0.5, full 0.15 (10%) |
| Prep elbow ("trophy") | as smash (10%) | as smash; same prep = disguise (10%) |
| Knee loading | as smash (5%) | as smash (5%) |
| Swing tempo, prep -> contact | <= 0.5 s, full 0.25 s (5%) | "Disguise": <= 0.5 s, full 0.25 s (10%) |
| Whole-swing match | as smash (10%) | as smash (10%) |

It grades whether your **form** matches the stroke you chose. With no shuttle tracking, it can't see where the shot actually went.

## 6-point footwork
**Setup:** put the camera behind the baseline (or in a high back corner) so it sees **the whole area and your feet**. A phone on a tripod, raised up, works best for a full half court; a laptop webcam can work for a small area. On the first run, click the 4 corners of your area in the window (NET-LEFT, NET-RIGHT, BACK-RIGHT, BACK-LEFT, with left/right as you face the net). The corners are saved to `data/court.json` and reused; press `c` to redo them.

**Drill:** stand still in the blue base circle. After a random pause a corner is called. Go there, lunge, and push back to base. Leaving before the call, or going to the wrong corner, doesn't count. After 12 reps (`--reps N`) you get a session card: grade, average, best/worst rep, comparison with your best and last session, the score for each corner, and your weakest metric. `--quiet` turns off the spoken calls (they use macOS `say`).

**Smaller areas:** any rectangle works. Targets sit just inside the edges, and the move/recovery time benchmarks are set from the distance to each corner (distance / speed), so a 2.5 m room is judged on how fast you move, not on how far you go. Below about 2.5 m, the score depends more on reaction, split step and lunge form.

| Metric | Measure | Pass line | Benchmark | Weight |
|---|---|---|---|---|
| Move time to corner | first step -> foot in the target zone | dist / 1.6 m/s + 0.2 s | dist / 3.0 m/s | 25% |
| Recovery to base | arrival -> back in base | dist / 1.3 m/s + 0.6 s | dist / 2.4 m/s + 0.25 s | 20% |
| Reaction time | call -> feet start moving | <= 0.50 s | 0.28 s | 15% |
| Lunge depth | most-bent knee at the corner | <= 135 deg | 105 deg | 15% |
| Direct path | straight distance / distance travelled | >= 0.75 | 0.93 | 10% |
| Square to the net (side targets) | shoulder width vs standing | >= 0.60 | 0.85 | 10% |
| Staying low | how much lower than standing while moving | >= 3% | 12% | 8% |
| Split step | hip bounce just before moving, torso lengths | >= 0.03 | 0.10 | 7% |

Side targets also report crossover steps (0 = pure chasse) as info. **These numbers are placeholders**, like the smash ones (`footwork.FootworkConfig`).

Recorded clips (upload in the web UI with the drill set to footwork, or `python live.py --drill footwork clip.mp4`) run in **free mode**: there are no call-outs, any trip from base to a corner and back is a rep, and reaction time isn't scored. In the web UI you type the corner pixel positions and check them against a preview. If the camera hasn't moved, the corners you clicked in live mode are filled in for you.

## Recording tips
- Phone about 3-4 m away, side-on or 45 degrees, whole body and racket arm in frame, plain background, good light.
- One overhead stroke per clip, about 2-4 s. Record a good take and a deliberately bad one (low contact, bent arm) for a clear before/after.

## Project layout
```
app.py            Streamlit UI (drill selector, video analysis, live, progress)
analysis.py       pose extraction, swing detection, smash/clear/drop metrics, shared scoring, drawing, video export
footwork.py       court calibration, footwork drill state machine, per-rep scoring, overlays
history.py        session history (best rep / best session / last session)
live.py           real-time stroke + footwork modes in a standalone OpenCV window
test_analysis.py  stroke tests on synthetic skeletons
test_footwork.py  footwork + history tests on a scripted synthetic player
data/             (git-ignored) history.json and court.json
requirements.txt
```

## Limitations
- 2D pose from a single camera: angles are projections, so camera placement affects readings. Trunk rotation and depth are not measured.
- Default thresholds are coaching heuristics, **not validated biomechanical norms**. Next step: collect coach-rated clips and fit / validate them.
- Single player; no shuttlecock tracking yet. Court position comes from the corners you mark, not from detecting court lines.
- Footwork: the camera must see your feet the whole time. Knee and shoulder readings are rougher when you're far from the camera, and side steps vs crossovers can be noisy from one 2D view.

## Roadmap
1. Validate thresholds against coach labels; add per-skill-level presets.
2. Stroke-type classification from pose + shuttle trajectory (BST transformer: ~83% accuracy over 25 classes on ShuttleSet).
3. Shuttle tracking (TrackNetV3) and court detection, then rally and game-level analytics.
4. Stronger pose models (e.g. RTMPose) for broadcast / far-field footage.

## References
- BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports, arXiv:2502.21085
- ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis, arXiv:2306.04948
- MediaPipe Pose Landmarker (Google)
