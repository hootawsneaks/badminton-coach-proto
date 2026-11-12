# Badminton Form Coach

Computer-vision badminton training. Pick a drill: **smash form** or **6-point footwork**.
Works on uploaded video **and** live webcam, and saves every rep so you can track your best rep and best session.

```
camera / video -> MediaPipe pose -> swing detection -> joint-angle checks -> coaching feedback + score
```

## Features
- **Video analysis:** upload a clip, get an annotated video (skeleton, live elbow angle, PREP / CONTACT labels), a 0-100 form score, written fixes, key-frame stills, and wrist-speed / elbow-angle plots.
- **Live mode:** get ready -> smash -> the screen **freezes on your contact frame with a letter grade, per-metric bars and the top fix** -> click **REDO** (or press `r`) to try again. Tracks attempt count and best score.
- **Smash benchmark:** 7 metrics, each scored 0-100 against a benchmark (60 = pass line, 100 = benchmark), weighted into a score and an A-F grade.
- **Tunable thresholds:** all form targets are sliders in the web UI (and fields in `analysis.Config`).
- **Racket arm:** auto-detected (or set manually).
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
python live.py --drill footwork                 # 6-point footwork, full singles half court (5.2 x 6.7 m)
python live.py --drill footwork --court 3x3     # ...in a smaller area (width x depth, metres)
# in the live window: d = switch drill (smash <-> footwork), r = redo / new session, c = re-mark court, q = quit
python test_footwork.py          # footwork + history tests (no video needed)
python test_analysis.py          # logic tests on synthetic skeletons (no video needed)
```

The pose model (`models/pose_landmarker_full.task`, ~9 MB) is **not stored in git**; it downloads automatically on first run (needs internet once). For an offline demo, run the app once beforehand.

## Smash benchmark
**The benchmark numbers are demo placeholders, not measured from real players.** They live in `analysis.Config` (`min_*`/`max_*` = pass line, `bench_*` = full marks).

| Metric | Measure | Pass line | Benchmark | Weight |
|---|---|---|---|---|
| Racket-hand speed | peak wrist speed, torso-lengths / s | >= 7 | 14 | 25% |
| Arm extension at contact | elbow angle | >= 150 deg | 170 deg | 20% |
| Contact height | wrist above shoulder, torso lengths | >= 0.60 | 0.95 | 20% |
| Downward follow-through | wrist drop below shoulder within 0.5 s after contact | >= 0.0 | 0.6 | 10% |
| Preparation elbow height ("trophy") | elbow vs shoulder, torso lengths | >= -0.15 | 0.10 | 10% |
| Knee loading before swing | min knee angle | <= 165 deg | 140 deg | 10% |
| Swing tempo | prep -> contact time | <= 0.45 s | 0.20 s | 5% |

Grade: A >= 90, B >= 80, C >= 70, D >= 60, else F. "Top fix" = the failed metric costing the most weighted points.

Swing = peak wrist speed. Contact = highest wrist point around that peak. Prep = most-flexed elbow in the 0.7 s before contact.

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
analysis.py       pose extraction, swing detection, smash checks, shared scoring, drawing, video export
footwork.py       court calibration, footwork drill state machine, per-rep scoring, overlays
history.py        session history (best rep / best session / last session)
live.py           real-time smash + footwork modes in a standalone OpenCV window
test_analysis.py  smash tests on synthetic skeletons
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
