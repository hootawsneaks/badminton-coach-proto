# Badminton Form Coach (demo)

Pose-based form correction for overhead clears/smashes.
Pipeline: MediaPipe PoseLandmarker -> swing detection (peak wrist speed) -> joint-angle checks -> coaching feedback.

## Run
    pip install -r requirements.txt
    streamlit run app.py        # web UI: upload clip tab + live webcam tab
    python live.py              # native OpenCV live window (most reliable for a live demo); q to quit
    python live.py clip.mp4     # replay a clip in "live" mode
    python test_analysis.py     # logic tests on synthetic skeletons

The pose model (models/pose_landmarker_full.task) auto-downloads on first run if missing.

## What it checks (all thresholds adjustable in the sidebar)
1. Arm extension at contact (elbow angle)
2. Contact height (wrist above shoulder, in torso lengths)
3. Preparation elbow height ("trophy" position)
4. Knee loading before the swing
5. Peak racket-hand speed (informational; compare attempts)

## Filming tips for the demo
- Phone propped ~3-4 m away, side-on or 45 degrees, whole body + racket arm in frame, even light, plain background.
- One overhead stroke per clip, trim to ~2-4 s. Shoot a deliberately bad take (low contact, bent arm) and a good one.
- Do a dry run on the actual laptop beforehand. First launch loads the model (a few seconds).

## Honest limitations (say these before they ask)
- 2D pose from one camera: angles are projections, so camera angle matters. Depth/rotation (trunk rotation) is not measured.
- Default thresholds are coaching heuristics, not validated norms. Next step: label coach-rated clips and fit/validate them.
- Single player, no shuttle tracking yet.

## Roadmap toward actual games (cite these)
- Stroke-type classification from pose + shuttle trajectory: BST (Badminton Stroke-type Transformer), arXiv 2502.21085 (~83% acc, 25 classes on ShuttleSet).
- Data: ShuttleSet (arXiv 2306.04948), BadmintonDB.
- Shuttle tracking: TrackNetV3; court detection for player positions; RTMPose for stronger pose in broadcast footage.
