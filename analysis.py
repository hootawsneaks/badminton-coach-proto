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
    # --- checks (heuristic defaults) ---
    min_elbow_extension_deg: float = 150.0   # hitting-arm elbow angle at contact
    min_contact_height: float = 0.60         # wrist above shoulder, in torso lengths
    min_trophy_elbow_height: float = -0.15   # elbow vs shoulder at prep (torso lengths; >= means ~level/high)
    max_knee_angle_prep: float = 165.0       # some knee flexion (loading) before the swing
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


@dataclass
class Report:
    ok: bool
    reason: str = ""
    arm: str = ""
    contact_idx: int = -1
    prep_idx: int = -1
    checks: list = field(default_factory=list)
    score: float | None = None
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

    # 1. elbow extension at contact
    e_c = float(elbow[contact])
    checks.append(Check("Arm extension at contact", e_c, f">= {cfg.min_elbow_extension_deg:.0f} deg",
                        "good" if e_c >= cfg.min_elbow_extension_deg else "warn",
                        f"Elbow {e_c:.0f} deg - " + ("arm well extended." if e_c >= cfg.min_elbow_extension_deg
                        else "reach up and straighten the arm at contact to hit at the highest point.")))

    # 2. contact height
    ch = float((k[contact, sh_i, 1] - k[contact, wr_i, 1]) / torso)
    checks.append(Check("Contact height", ch, f">= {cfg.min_contact_height:.2f} torso above shoulder",
                        "good" if ch >= cfg.min_contact_height else "warn",
                        f"Contact {ch:.2f} torso-lengths above shoulder - " + ("high contact point." if ch >= cfg.min_contact_height
                        else "you are hitting too low; get under the shuttle and hit at full reach.")))

    # 3. trophy-position elbow height
    eh = float((k[prep, sh_i, 1] - k[prep, el_i, 1]) / torso)
    checks.append(Check("Preparation elbow height", eh, f">= {cfg.min_trophy_elbow_height:.2f}",
                        "good" if eh >= cfg.min_trophy_elbow_height else "warn",
                        ("Elbow raised to about shoulder level in the preparation phase." if eh >= cfg.min_trophy_elbow_height
                         else "Elbow drops below the shoulder in preparation - lift the elbow to build a proper 'trophy' position.")))

    # 4. knee loading (use the more flexed knee in the prep window)
    kn = []
    for (kk, hh, aa) in (( knee_i, hip_i, ank_i), other):
        kn.append(np.nanmin([angle(k[t, hh], k[t, kk], k[t, aa]) for t in range(plo, contact + 1)]))
    knee = float(np.nanmin(kn))
    checks.append(Check("Knee loading", knee, f"<= {cfg.max_knee_angle_prep:.0f} deg",
                        "good" if knee <= cfg.max_knee_angle_prep else "warn",
                        f"Min knee angle {knee:.0f} deg - " + ("good leg load before the swing." if knee <= cfg.max_knee_angle_prep
                        else "legs are stiff; bend the knees and push up through the stroke for power.")))

    # 5. swing speed (informational: compare attempts, not an absolute pass/fail)
    checks.append(Check("Peak racket-hand speed", float(speed[c]), "higher = more power (compare attempts)",
                        "info", f"{speed[c]:.1f} torso-lengths/s"))

    scored = [x for x in checks if x.status in ("good", "warn")]
    score = 100.0 * sum(x.status == "good" for x in scored) / max(1, len(scored))
    return Report(True, arm=arm, contact_idx=contact, prep_idx=prep, checks=checks, score=score,
                  speed=speed, elbow=elbow, fps=fps)


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
                put_text(im, f"Form score {rep.score:.0f}/100", (20, h - 20), 0.9, (255, 255, 255),
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
