"""Badminton overhead-stroke form analysis (clear / smash) from 2D pose.

Pipeline: MediaPipe PoseLandmarker -> smoothed keypoints -> swing detection
(peak wrist speed) -> joint-angle checks at key phases -> coaching feedback.

NOTE: the default thresholds are coaching heuristics, NOT validated
biomechanical norms. They are exposed as `Config` so they can be tuned
(ideally against coach-labelled clips) before being presented as ground truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

# MediaPipe BlazePose landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


@dataclass
class Config:
    # Smash benchmark. Each metric has a pass line (below it = "FIX") and a benchmark (= full marks).
    # These are DEMO placeholder numbers, not measured from real players - tune them against coach-rated clips.
    min_elbow_extension_deg: float = 150.0   # hitting-arm elbow angle at contact
    bench_elbow_extension_deg: float = 170.0
    min_contact_height: float = 0.60         # wrist above shoulder, in torso lengths
    bench_contact_height: float = 0.95
    min_trophy_elbow_height: float = -0.15   # elbow vs shoulder at prep (torso lengths; >= means ~level/high)
    bench_trophy_elbow_height: float = 0.10
    max_knee_angle_prep: float = 165.0       # some knee flexion (loading) before the swing
    bench_knee_angle_prep: float = 140.0
    min_smash_speed: float = 7.0             # peak racket-hand speed, torso-lengths / second
    bench_smash_speed: float = 14.0
    min_follow_through: float = 0.0          # how far the wrist drops below the shoulder after contact (torso lengths)
    bench_follow_through: float = 0.6
    max_swing_time_s: float = 0.45           # prep -> contact time (shorter = more explosive)
    bench_swing_time_s: float = 0.20
    follow_window_s: float = 0.5             # look-ahead window after contact for the follow-through
    # --- swing detection ---
    swing_speed_thresh: float = 4.0          # wrist speed, torso-lengths / second
    prep_window_s: float = 0.7               # look-back window from contact for the prep phase
    smooth_win: int = 5


@dataclass
class Check:
    name: str
    value: float | None
    target: str
    status: str            # "good" | "warn" | "info" | "na"
    message: str
    points: float | None = None   # 0-100 vs benchmark (None = not scored)
    weight: float = 0.0
    bench: str = ""


@dataclass
class Report:
    ok: bool
    reason: str = ""
    arm: str = ""
    contact_idx: int = -1
    prep_idx: int = -1
    checks: list = field(default_factory=list)
    score: float | None = None
    grade: str = ""
    top_fix: str = ""
    speed: np.ndarray | None = None
    elbow: np.ndarray | None = None
    fps: float = 30.0

    def summary_lines(self):
        return [f"{c.name}: {c.message}" for c in self.checks if c.status == "warn"]


# ---------------------------------------------------------------- geometry
def angle(a, b, c) -> float:
    """Angle ABC in degrees (vertex b). Inputs are (x, y)."""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along axis 0 (works on [N, ...])."""
    out = arr.copy()
    flat = out.reshape(len(out), -1)
    idx = np.arange(len(out))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.sum() == 0:
            continue
        if good.sum() < len(col):
            col[~good] = np.interp(idx[~good], idx[good], col[good])
    return flat.reshape(out.shape)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    pad = w // 2
    shape = x.shape
    flat = x.reshape(len(x), -1)
    xp = np.pad(flat, [(pad, w - 1 - pad), (0, 0)], mode="edge")
    out = np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(flat.shape[1])], axis=1)
    return out.reshape(shape)


# ---------------------------------------------------------------- pose
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def make_landmarker(video_mode: bool = True):
    from mediapipe.tasks.python import vision, BaseOptions
    ensure_model()
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def detect(landmarker, bgr: np.ndarray, ts_ms: int | None = None):
    """Run pose on a BGR frame. Returns (33,2) pixel coords + (33,) visibility, or NaNs."""
    import mediapipe as mp
    h, w = bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = landmarker.detect_for_video(img, ts_ms) if ts_ms is not None else landmarker.detect(img)
    if not res.pose_landmarks:
        return np.full((33, 2), np.nan), np.zeros(33)
    lm = res.pose_landmarks[0]
    xy = np.array([[p.x * w, p.y * h] for p in lm])
    vis = np.array([getattr(p, "visibility", 1.0) or 0.0 for p in lm])
    # The model extrapolates joints that are off-screen; treat those (and near-zero visibility) as missing.
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h) & (vis >= 0.15)
    xy[~inside] = np.nan
    return xy, vis


def extract_video(path: str, max_width: int = 960, progress_cb=None):
    """Run pose over a video file. Returns dict with kps [N,33,2], vis, fps, size, frames (resized BGR)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    lm = make_landmarker(True)
    kps, viss, frames = [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > max_width:
            fr = cv2.resize(fr, (max_width, int(h * max_width / w)))
        xy, vis = detect(lm, fr, int(i * 1000 / fps))
        kps.append(xy); viss.append(vis); frames.append(fr)
        i += 1
        if progress_cb and total:
            progress_cb(min(i / total, 1.0))
    cap.release(); lm.close()
    if not kps:
        raise ValueError("Video has no frames")
    return dict(kps=np.array(kps), vis=np.array(viss), fps=fps, frames=frames)


# ---------------------------------------------------------------- framing check
FRAME_HINT = "Body not fully in frame - step back 2-3 m so you are visible from head to feet"


def _ok(p) -> bool:
    return not np.isnan(p).any()


def body_in_frame(xy) -> bool:
    """One frame: torso, at least one full leg and at least one full arm are visible on screen."""
    torso = all(_ok(xy[i]) for i in (L_SH, R_SH, L_HIP, R_HIP))
    leg = any(all(_ok(xy[i]) for i in leg_) for leg_ in ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)))
    arm = any(all(_ok(xy[i]) for i in arm_) for arm_ in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)))
    return torso and leg and arm


def frame_coverage(kps: np.ndarray) -> float:
    return float(np.mean([body_in_frame(k) for k in kps])) if len(kps) else 0.0


# ---------------------------------------------------------------- grading
# Weights of each smash metric in the overall score (sum to 1).
WEIGHTS = {"speed": 0.25, "extension": 0.20, "height": 0.20, "follow": 0.10,
           "prep": 0.10, "knee": 0.10, "tempo": 0.05}


def points(value: float, pass_line: float, bench: float) -> float:
    """0-100 credit: 60 at the pass line, 100 at the benchmark, linear (works for lower-is-better too)."""
    if value is None or np.isnan(value) or bench == pass_line:
        return 0.0
    return float(np.clip(60 + 40 * (value - pass_line) / (bench - pass_line), 0, 100))


def letter(score: float) -> str:
    for cut, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cut:
            return g
    return "F"


# ---------------------------------------------------------------- analysis
def _arm_idx(arm: str):
    return (R_SH, R_EL, R_WR) if arm == "right" else (L_SH, L_EL, L_WR)


def analyze(kps: np.ndarray, fps: float, cfg: Config | None = None, arm: str | None = None) -> Report:
    """Analyse one overhead stroke inside a [N,33,2] keypoint sequence (pixel coords, y down)."""
    cfg = cfg or Config()
    N = len(kps)
    if N < 8:
        return Report(False, "Not enough frames")
    det = ~np.isnan(kps[:, NOSE, 0])
    if det.mean() < 0.5:
        return Report(False, "Player not detected in enough frames - frame the whole body, side-on, good light")
    if frame_coverage(kps) < 0.6:
        return Report(False, FRAME_HINT)
    k = _smooth(_interp_nan(kps), cfg.smooth_win)

    mid_sh = (k[:, L_SH] + k[:, R_SH]) / 2
    mid_hip = (k[:, L_HIP] + k[:, R_HIP]) / 2
    torso = float(np.nanmedian(np.linalg.norm(mid_sh - mid_hip, axis=1)))
    if torso < 5:
        return Report(False, "Torso too small in frame - move closer")

    def speed_of(wr):
        v = np.zeros(N); v[1:] = np.linalg.norm(np.diff(k[:, wr], axis=0), axis=1) * fps / torso
        return _smooth(v, 3)

    sp = {"right": speed_of(R_WR), "left": speed_of(L_WR)}
    if arm is None:
        arm = "right" if sp["right"].max() >= sp["left"].max() else "left"
    speed = sp[arm]
    sh_i, el_i, wr_i = _arm_idx(arm)
    knee_i, hip_i, ank_i = (R_KN, R_HIP, R_AN) if arm == "right" else (L_KN, L_HIP, L_AN)
    other = (L_KN, L_HIP, L_AN) if arm == "right" else (R_KN, R_HIP, R_AN)

    c = int(np.argmax(speed))
    if speed[c] < cfg.swing_speed_thresh:
        return Report(False, f"No clear swing detected (peak wrist speed {speed[c]:.1f} < {cfg.swing_speed_thresh:.1f} torso/s)",
                      speed=speed, fps=fps, arm=arm)

    elbow = np.array([angle(k[t, sh_i], k[t, el_i], k[t, wr_i]) for t in range(N)])
    # contact: highest wrist point shortly around the speed peak
    lo, hi = max(0, c - int(0.12 * fps)), min(N, c + int(0.12 * fps) + 1)
    contact = lo + int(np.argmin(k[lo:hi, wr_i, 1]))   # smallest y = highest
    win = max(3, int(cfg.prep_window_s * fps))
    plo = max(0, contact - win)
    prep = plo + int(np.argmin(elbow[plo:contact])) if contact > plo else plo

    checks: list[Check] = []

    def add(key, name, value, pass_line, bench, unit, good_msg, fix_msg, fmt="{:.0f}"):
        if value is None or np.isnan(value):
            checks.append(Check(name, None, "", "na", "Not measurable in this clip.", None, WEIGHTS[key]))
            return
        higher = bench > pass_line
        ok = value >= pass_line if higher else value <= pass_line
        op = ">=" if higher else "<="
        v = fmt.format(value)
        checks.append(Check(name, float(value), f"{op} {fmt.format(pass_line)}{unit}",
                            "good" if ok else "warn",
                            f"{v}{unit} (benchmark {fmt.format(bench)}{unit}) - " + (good_msg if ok else fix_msg),
                            points(value, pass_line, bench), WEIGHTS[key], f"{fmt.format(bench)}{unit}"))

    # 1. racket-hand speed (the "smash power" proxy)
    add("speed", "Racket-hand speed", float(speed[c]), cfg.min_smash_speed, cfg.bench_smash_speed, " torso/s",
        "explosive swing.", "swing faster - snap the forearm (pronation) and whip through the shuttle.", "{:.1f}")

    # 2. elbow extension at contact
    add("extension", "Arm extension at contact", float(elbow[contact]), cfg.min_elbow_extension_deg,
        cfg.bench_elbow_extension_deg, " deg", "arm well extended.",
        "reach up and straighten the arm at contact to hit at the highest point.")

    # 3. contact height
    add("height", "Contact height", float((k[contact, sh_i, 1] - k[contact, wr_i, 1]) / torso),
        cfg.min_contact_height, cfg.bench_contact_height, " torso", "high contact point.",
        "hitting too low; get under and slightly behind the shuttle and hit at full reach.", "{:.2f}")

    # 4. follow-through: a smash finishes down and across the body
    fend = min(N, contact + 1 + int(cfg.follow_window_s * fps))
    ft = float(np.max(k[contact + 1:fend, wr_i, 1] - k[contact + 1:fend, sh_i, 1]) / torso) \
        if fend - contact > 3 else float("nan")
    add("follow", "Downward follow-through", ft, cfg.min_follow_through, cfg.bench_follow_through, " torso",
        "racket follows through down past the shoulder.",
        "finish the swing down and across your body - don't stop the racket after contact.", "{:.2f}")

    # 5. trophy-position elbow height
    add("prep", "Preparation elbow height", float((k[prep, sh_i, 1] - k[prep, el_i, 1]) / torso),
        cfg.min_trophy_elbow_height, cfg.bench_trophy_elbow_height, " torso",
        "elbow up in a good 'trophy' position.",
        "elbow drops below the shoulder in preparation - lift it to build the 'trophy' position.", "{:.2f}")

    # 6. knee loading (use the more flexed knee in the prep window)
    kn = []
    for (kk, hh, aa) in ((knee_i, hip_i, ank_i), other):
        kn.append(np.nanmin([angle(k[t, hh], k[t, kk], k[t, aa]) for t in range(plo, contact + 1)]))
    add("knee", "Knee loading", float(np.nanmin(kn)), cfg.max_knee_angle_prep, cfg.bench_knee_angle_prep, " deg",
        "good leg load before the swing.", "legs are stiff; bend the knees and jump/push up into the smash.")

    # 7. swing tempo: prep -> contact
    add("tempo", "Swing tempo", (contact - prep) / fps, cfg.max_swing_time_s, cfg.bench_swing_time_s, " s",
        "quick, explosive acceleration.", "the swing is slow from prep to contact; accelerate late and hard.", "{:.2f}")

    scored = [x for x in checks if x.points is not None]
    wsum = sum(x.weight for x in scored)
    score = sum(x.points * x.weight for x in scored) / wsum if wsum else 0.0
    fixes = sorted((x for x in scored if x.status == "warn"), key=lambda x: -(100 - x.points) * x.weight)
    top_fix = f"{fixes[0].name}: {fixes[0].message.split(' - ', 1)[-1]}" if fixes else ""
    return Report(True, arm=arm, contact_idx=contact, prep_idx=prep, checks=checks, score=score,
                  grade=letter(score), top_fix=top_fix, speed=speed, elbow=elbow, fps=fps)


# ---------------------------------------------------------------- drawing
def draw_pose(img, xy, color=(80, 220, 120), highlight_arm: str | None = None, thick=3):
    if np.isnan(xy[NOSE, 0]):
        return img
    for a, b in SKELETON:
        if np.isnan(xy[a]).any() or np.isnan(xy[b]).any():
            continue
        col = color
        if highlight_arm and {a, b} & set(_arm_idx(highlight_arm)) and not ({a, b} & {L_HIP, R_HIP}):
            col = (60, 160, 255)
        cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), col, thick, cv2.LINE_AA)
    for i in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN):
        if not np.isnan(xy[i]).any():
            cv2.circle(img, tuple(int(v) for v in xy[i]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), bg=(0, 0, 0), thick=2):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + b + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def annotate_video(frames, kps, rep: Report, fps: float, out_path: str):
    """Write annotated video with skeleton, live elbow angle, phase labels and result banner."""
    h, w = frames[0].shape[:2]
    k = _smooth(_interp_nan(kps), 5)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    hold = int(0.8 * fps)
    for t, fr in enumerate(frames):
        im = fr.copy()
        draw_pose(im, k[t], highlight_arm=rep.arm if rep.ok else None)
        if rep.ok and rep.elbow is not None:
            sh_i, el_i, wr_i = _arm_idx(rep.arm)
            put_text(im, f"{rep.elbow[t]:.0f} deg", (int(k[t, el_i, 0]) + 10, int(k[t, el_i, 1])), 0.6, (255, 255, 255), (30, 30, 30))
            lab = None
            if abs(t - rep.prep_idx) <= hold // 2: lab = "PREP"
            if rep.contact_idx - hold // 2 <= t <= rep.contact_idx + hold: lab = "CONTACT"
            if lab:
                put_text(im, lab, (20, 40), 1.0, (255, 255, 255), (0, 120, 255) if lab == "CONTACT" else (160, 90, 0), 2)
            if t >= rep.contact_idx:
                put_text(im, f"Smash grade {rep.grade}  ({rep.score:.0f}/100)", (20, h - 20), 0.9, (255, 255, 255),
                         (0, 130, 0) if rep.score >= 75 else (0, 110, 200) if rep.score >= 50 else (0, 0, 170), 2)
        wr.write(im)
    wr.release()


def to_h264(src: str, dst: str) -> bool:
    """Re-encode to H.264 so browsers can play it. Returns False if no ffmpeg available."""
    import subprocess
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
                       check=True, timeout=300)
        return os.path.exists(dst)
    except Exception:
        return False
