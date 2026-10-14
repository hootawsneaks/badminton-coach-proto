# Badminton Form Coach

Computer-vision form correction for badminton overhead strokes (clear / smash), with a path toward full game analysis.
Works on uploaded video **and** live webcam.

```
camera / video -> MediaPipe pose -> swing detection -> joint-angle checks -> coaching feedback + score
```

## Features
- **Video analysis:** upload a clip, get an annotated video (skeleton, live elbow angle, PREP / CONTACT labels), a 0-100 form score, written fixes, key-frame stills, and wrist-speed / elbow-angle plots.
- **Live mode:** rolling buffer with automatic swing detection; feedback appears on screen after each stroke.
- **Tunable thresholds:** all form targets are sliders in the web UI (and fields in `analysis.Config`).
- **Racket arm:** auto-detected (or set manually).

## Quick start
Requires Python 3.10+ and a webcam for live mode.

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # web UI: "Analyze a video" + "Live webcam" tabs
python live.py                   # native live window (most reliable for demos); press q to quit
python live.py clip.mp4          # replay a clip in live mode
python test_analysis.py          # logic tests on synthetic skeletons (no video needed)
```

The pose model (`models/pose_landmarker_full.task`, ~9 MB) is **not stored in git**; it downloads automatically on first run (needs internet once). For an offline demo, run the app once beforehand.

## What it checks
| Check | Measure | Default target |
|---|---|---|
| Arm extension at contact | elbow angle | >= 150 deg |
| Contact height | wrist above shoulder, in torso lengths | >= 0.60 |
| Preparation elbow height ("trophy") | elbow vs shoulder, in torso lengths | >= -0.15 |
| Knee loading before swing | min knee angle | <= 165 deg |
| Peak racket-hand speed | torso-lengths / second | informational only |

Swing = peak wrist speed. Contact = highest wrist point around that peak. Prep = most-flexed elbow in the 0.7 s before contact.

## Recording tips
- Phone about 3-4 m away, side-on or 45 degrees, whole body and racket arm in frame, plain background, good light.
- One overhead stroke per clip, about 2-4 s. Record a good take and a deliberately bad one (low contact, bent arm) for a clear before/after.

## Project layout
```
app.py            Streamlit UI
analysis.py       pose extraction, swing detection, checks, drawing, video export
live.py           real-time analyzer + standalone OpenCV window
test_analysis.py  synthetic-skeleton tests
requirements.txt
```

## Limitations
- 2D pose from a single camera: angles are projections, so camera placement affects readings. Trunk rotation and depth are not measured.
- Default thresholds are coaching heuristics, **not validated biomechanical norms**. Next step: collect coach-rated clips and fit / validate them.
- Single player; no shuttlecock or court tracking yet.

## Roadmap
1. Validate thresholds against coach labels; add per-skill-level presets.
2. Stroke-type classification from pose + shuttle trajectory (BST transformer: ~83% accuracy over 25 classes on ShuttleSet).
3. Shuttle tracking (TrackNetV3) and court detection, then rally and game-level analytics.
4. Stronger pose models (e.g. RTMPose) for broadcast / far-field footage.

## References
- BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports, arXiv:2502.21085
- ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis, arXiv:2306.04948
- MediaPipe Pose Landmarker (Google)
