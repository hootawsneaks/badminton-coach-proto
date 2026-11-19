"""Badminton overhead-stroke form analysis (smash / clear / drop) from 2D pose.

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
# Lighter, faster model for live mode (higher frame rate, a bit less precise on fast wrists).
LITE_MODEL_PATH = MODEL_PATH.replace("_full", "_lite")
LITE_MODEL_URL = MODEL_URL.replace("_full", "_lite")

# MediaPipe BlazePose landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


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
    smooth_s: float = 0.1                    # keypoint smoothing window in seconds (scaled to the frame rate)
    contact_lookback_s: float = 0.3          # wrist speed peaks on the downswing, so search this far back for contact


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
    stroke: str = "smash"
    phases: list = field(default_factory=list)   # [Phase] key moments of the swing, in time order
    snapshots: list = field(default_factory=list)  # [Phase] evenly spaced moments across the whole swing
    track: "Track | None" = None                 # per-frame target path ("shadow") for the whole swing

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


def win_frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


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
def ensure_model(path: str = MODEL_PATH, url: str = MODEL_URL):
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def make_landmarker(video_mode: bool = True, lite: bool = False):
    from mediapipe.tasks.python import vision, BaseOptions
    path, url = (LITE_MODEL_PATH, LITE_MODEL_URL) if lite else (MODEL_PATH, MODEL_URL)
    ensure_model(path, url)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path),
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
WEIGHTS = {"speed": 0.25, "extension": 0.15, "height": 0.15, "follow": 0.10,
           "prep": 0.10, "knee": 0.10, "tempo": 0.05, "path": 0.10}


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


def make_check(weight, name, value, pass_line, bench, unit, good_msg, fix_msg, fmt="{:.0f}") -> Check:
    """Score one metric against its pass line / benchmark (either direction). value None/NaN -> not scored."""
    if value is None or np.isnan(value):
        return Check(name, None, "", "na", "Not measurable here.", None, weight)
    higher = bench > pass_line
    ok = value >= pass_line if higher else value <= pass_line
    op = ">=" if higher else "<="
    return Check(name, float(value), f"{op} {fmt.format(pass_line)}{unit}", "good" if ok else "warn",
                 f"{fmt.format(value)}{unit} (benchmark {fmt.format(bench)}{unit}) - " + (good_msg if ok else fix_msg),
                 points(value, pass_line, bench), weight, f"{fmt.format(bench)}{unit}")


def score_checks(checks) -> tuple[float, str, str]:
    """Weighted score over the scored checks -> (score, letter grade, top fix)."""
    scored = [x for x in checks if x.points is not None]
    wsum = sum(x.weight for x in scored)
    score = sum(x.points * x.weight for x in scored) / wsum if wsum else 0.0
    fixes = sorted((x for x in scored if x.status == "warn"), key=lambda x: -(100 - x.points) * x.weight)
    top_fix = f"{fixes[0].name}: {fixes[0].message.split(' - ', 1)[-1]}" if fixes else ""
    return score, letter(score), top_fix


def make_band_check(weight, name, value, lo_pass, lo_bench, hi_bench, hi_pass, unit, good_msg, low_msg, high_msg,
                    fmt="{:.0f}") -> Check:
    """For metrics with a sweet spot (e.g. drop-shot speed): 100 inside [lo_bench, hi_bench], 60 at either pass
    line, falling off beyond. Too far either way is a FIX."""
    if value is None or np.isnan(value):
        return Check(name, None, "", "na", "Not measurable here.", None, weight)
    if value < lo_bench:
        pts, msg = points(value, lo_pass, lo_bench), low_msg
    elif value > hi_bench:
        pts, msg = points(value, hi_pass, hi_bench), high_msg
    else:
        pts, msg = 100.0, good_msg
    ok = lo_pass <= value <= hi_pass
    f = fmt.format
    return Check(name, float(value), f"{f(lo_pass)}-{f(hi_pass)}{unit}", "good" if ok else "warn",
                 f"{f(value)}{unit} (target {f(lo_bench)}-{f(hi_bench)}{unit}) - " + (good_msg if ok else msg),
                 pts, weight, f"{f(lo_bench)}-{f(hi_bench)}{unit}")


# ---------------------------------------------------------------- strokes
STROKES = ("smash", "clear", "drop")


@dataclass
class Metric:
    key: str            # which measured value (see analyze)
    name: str
    pass_line: float
    bench: float
    weight: float
    unit: str
    fmt: str
    good: str
    fix: str
    band: tuple | None = None   # (hi_bench, hi_pass, too_high_msg): scored as a sweet spot, `fix` = too low


def stroke_metrics(stroke: str, cfg: Config) -> list[Metric]:
    """Benchmarks per overhead stroke. Smash reads its numbers from `cfg` (tunable in the web UI);
    clear and drop are fixed DEMO placeholders - same caveat: tune them against coach-rated clips."""
    W = WEIGHTS
    ext = Metric("extension", "Arm extension at contact", cfg.min_elbow_extension_deg, cfg.bench_elbow_extension_deg,
                 0, " deg", "{:.0f}", "arm well extended.",
                 "reach up and straighten the arm at contact to hit at the highest point.")
    prep = Metric("prep", "Preparation elbow height", cfg.min_trophy_elbow_height, cfg.bench_trophy_elbow_height, 0,
                  " torso", "{:.2f}", "elbow up in a good 'trophy' position.",
                  "elbow drops below the shoulder in preparation - lift it to build the 'trophy' position.")
    knee = Metric("knee", "Knee loading", cfg.max_knee_angle_prep, cfg.bench_knee_angle_prep, 0, " deg", "{:.0f}",
                  "good leg load before the swing.", f"legs are stiff; bend the knees and push up into the {stroke}.")
    path = Metric("path", "Whole-swing match", 70.0, 92.0, 0, "%", "{:.0f}",
                  "your movement follows the target path through the whole swing.",
                  "parts of the swing drift off the target path - watch the replay and follow the shadow.")
    w = lambda m, weight, **kw: Metric(**{**m.__dict__, "weight": weight, **kw})
    if stroke == "smash":
        return [
            Metric("speed", "Racket-hand speed", cfg.min_smash_speed, cfg.bench_smash_speed, W["speed"], " torso/s",
                   "{:.1f}", "explosive swing.",
                   "swing faster - snap the forearm (pronation) and whip through the shuttle."),
            w(ext, W["extension"]),
            Metric("height", "Contact height", cfg.min_contact_height, cfg.bench_contact_height, W["height"], " torso",
                   "{:.2f}", "high contact point.",
                   "hitting too low; get under and slightly behind the shuttle and hit at full reach."),
            Metric("follow", "Downward follow-through", cfg.min_follow_through, cfg.bench_follow_through, W["follow"],
                   " torso", "{:.2f}", "racket follows through down past the shoulder.",
                   "finish the swing down and across your body - don't stop the racket after contact."),
            w(prep, W["prep"]), w(knee, W["knee"]),
            Metric("tempo", "Swing tempo", cfg.max_swing_time_s, cfg.bench_swing_time_s, W["tempo"], " s", "{:.2f}",
                   "quick, explosive acceleration.",
                   "the swing is slow from prep to contact; accelerate late and hard."),
            w(path, W["path"]),
        ]
    if stroke == "clear":
        return [
            Metric("speed", "Racket-hand speed", 6.0, 11.0, 0.20, " torso/s", "{:.1f}",
                   "powerful swing - enough to reach the back line.",
                   "swing faster - a clear needs a full, fast throwing action to go deep."),
            w(ext, 0.15),
            Metric("height", "Contact height", 0.65, 1.00, 0.15, " torso", "{:.2f}", "hit at full reach.",
                   "contact is low - get behind the shuttle early and hit at the top of your reach."),
            Metric("front", "Contact position", -0.15, -0.05, 0.10, " torso", "{:.2f}",
                   "contact above / just in front of the head - right for a clear.",
                   "contact is behind you - move back faster so you get behind the shuttle.",
                   band=(0.20, 0.35, "contact is far in front - for a clear let it come above your head "
                                     "(hitting it out in front sends it down like a smash/drop).")),
            Metric("follow", "Full follow-through", 0.0, 0.4, 0.10, " torso", "{:.2f}", "full follow-through.",
                   "don't stop the racket at contact - follow right through so the clear carries."),
            w(prep, 0.10), w(knee, 0.05),
            Metric("tempo", "Swing tempo", 0.50, 0.25, 0.05, " s", "{:.2f}", "smooth, quick acceleration.",
                   "the swing is slow from prep to contact; accelerate through the shuttle."),
            w(path, 0.10),
        ]
    if stroke == "drop":
        return [
            Metric("speed", "Racket-hand speed", 4.5, 6.0, 0.20, " torso/s", "{:.1f}", "good touch - controlled pace.",
                   "too slow and obvious - swing like a smash and slow the racket only at contact.",
                   band=(8.5, 10.0, "too fast for a drop - ease off at contact so it falls just over the net.")),
            w(ext, 0.10),
            Metric("height", "Contact height", 0.60, 0.95, 0.10, " torso", "{:.2f}", "high contact point.",
                   "contact is low - a drop from low height floats up; hit it high."),
            Metric("front", "Contact in front", 0.10, 0.35, 0.15, " torso", "{:.2f}",
                   "taken well in front - the drop travels down steeply.",
                   "take the drop further in front of your body so it goes down, not up."),
            Metric("follow", "Controlled follow-through", 0.50, 0.15, 0.10, " torso", "{:.2f}",
                   "short, controlled finish.",
                   "the follow-through is big like a smash - check the racket after contact for a softer drop."),
            w(prep, 0.10, good="same 'trophy' prep as a smash - good disguise."),
            w(knee, 0.05),
            Metric("tempo", "Disguise (swing tempo)", 0.50, 0.25, 0.10, " s", "{:.2f}",
                   "quick prep-to-contact like a smash - hard to read.",
                   "slow, telegraphed swing - keep your smash preparation and slow down only at contact."),
            w(path, 0.10),
        ]
    raise ValueError(f"unknown stroke {stroke!r}, expected one of {STROKES}")


# ---------------------------------------------------------------- analysis
def _arm_idx(arm: str):
    return (R_SH, R_EL, R_WR) if arm == "right" else (L_SH, L_EL, L_WR)


def analyze(kps: np.ndarray, fps: float, cfg: Config | None = None, arm: str | None = None,
            stroke: str = "smash") -> Report:
    """Analyse one overhead stroke (smash / clear / drop) inside a [N,33,2] keypoint sequence (pixel coords, y down)."""
    cfg = cfg or Config()
    metrics = stroke_metrics(stroke, cfg)
    N = len(kps)
    if N < 8:
        return Report(False, "Not enough frames")
    det = ~np.isnan(kps[:, NOSE, 0])
    if det.mean() < 0.5:
        return Report(False, "Player not detected in enough frames - frame the whole body, side-on, good light")
    if frame_coverage(kps) < 0.6:
        return Report(False, FRAME_HINT)
    # Smoothing is time-based: a fixed 5-frame window at a ~10 fps live feed spans half a second
    # and flattens the top of the swing, which made contact look low and the follow-through huge.
    sw = win_frames(cfg.smooth_s, fps)
    k = _smooth(_interp_nan(kps), sw)

    mid_sh = (k[:, L_SH] + k[:, R_SH]) / 2
    mid_hip = (k[:, L_HIP] + k[:, R_HIP]) / 2
    torso = float(np.nanmedian(np.linalg.norm(mid_sh - mid_hip, axis=1)))
    if torso < 5:
        return Report(False, "Torso too small in frame - move closer")

    def speed_of(wr):
        v = np.zeros(N); v[1:] = np.linalg.norm(np.diff(k[:, wr], axis=0), axis=1) * fps / torso
        return _smooth(v, sw)

    sp = {"right": speed_of(R_WR), "left": speed_of(L_WR)}
    if arm is None:
        arm = "right" if sp["right"].max() >= sp["left"].max() else "left"
    speed = sp[arm]
    sh_i, el_i, wr_i = _arm_idx(arm)

    c = int(np.argmax(speed))
    if speed[c] < cfg.swing_speed_thresh:
        return Report(False, f"No clear swing detected (peak wrist speed {speed[c]:.1f} < {cfg.swing_speed_thresh:.1f} torso/s)",
                      speed=speed, fps=fps, arm=arm)

    elbow = np.array([angle(k[t, sh_i], k[t, el_i], k[t, wr_i]) for t in range(N)])
    # contact: highest wrist point around the speed peak. In a smash the hand is fastest just *after*
    # contact on the way down, so look mostly backwards (and always at least a couple of frames).
    back, fwd = max(2, int(round(cfg.contact_lookback_s * fps))), max(1, int(round(0.1 * fps)))
    lo, hi = max(0, c - back), min(N, c + fwd + 1)
    contact = lo + int(np.argmin(k[lo:hi, wr_i, 1]))   # smallest y = highest
    win = max(3, int(cfg.prep_window_s * fps))
    plo = max(0, contact - win)
    prep = plo + int(np.argmin(elbow[plo:contact])) if contact > plo else plo

    # ---- measured values
    v = {}
    v["speed"] = float(speed[c])                                          # racket-hand speed (power proxy)
    v["extension"] = float(elbow[contact])                                # elbow angle at contact
    v["height"] = float((k[contact, sh_i, 1] - k[contact, wr_i, 1]) / torso)
    fend = min(N, contact + 1 + int(cfg.follow_window_s * fps))           # wrist drop below shoulder after contact
    if fend - contact > 3:
        drop = k[contact + 1:fend, wr_i, 1] - k[contact + 1:fend, sh_i, 1]
        v["follow"], finish = float(np.max(drop) / torso), contact + 1 + int(np.argmax(drop))
    else:
        v["follow"], finish = float("nan"), min(N - 1, contact + win_frames(0.3, fps))
    v["prep"] = float((k[prep, sh_i, 1] - k[prep, el_i, 1]) / torso)    # trophy-position elbow height
    knees = np.array([min((angle(k[t, h], k[t, kk], k[t, a]) for h, kk, a in LEGS), key=lambda x: (np.isnan(x), x))
                      for t in range(plo, contact + 1)])                  # most-flexed knee in the prep window
    v["knee"] = float(np.nanmin(knees)) if not np.isnan(knees).all() else float("nan")
    load = plo + int(np.nanargmin(knees)) if not np.isnan(knees).all() else prep
    v["tempo"] = (contact - prep) / fps
    # contact in front (+) / behind (-) the hitting shoulder. "Forward" = the way the face points, from the
    # nose vs the shoulder centre - only readable from a side-on-ish camera, otherwise not scored.
    facing = float(np.nanmedian((k[plo:contact + 1, NOSE, 0] - mid_sh[plo:contact + 1, 0]) / torso))
    v["front"] = float((k[contact, wr_i, 0] - k[contact, sh_i, 0]) * np.sign(facing) / torso) \
        if abs(facing) >= 0.05 else float("nan")

    face = float(np.sign(facing)) if abs(facing) >= 0.05 else 0.0
    keys = {"Load": load, "Trophy": prep, "Contact": contact, "Finish": finish}
    track = build_track(k, keys, metrics, arm, torso, face, fps)
    path_m = next((m for m in metrics if m.key == "path"), None)
    snapshots = build_snapshots(track, contact, fps, path_m.pass_line if path_m else 70.0)
    v["path"] = float(np.mean([p.match for p in snapshots])) if snapshots else float("nan")

    checks: list[Check] = []
    for m in metrics:
        if m.band:
            hi_bench, hi_pass, high_msg = m.band
            checks.append(make_band_check(m.weight, m.name, v[m.key], m.pass_line, m.bench, hi_bench, hi_pass, m.unit,
                                          m.good, m.fix, high_msg, m.fmt))
        else:
            checks.append(make_check(m.weight, m.name, v[m.key], m.pass_line, m.bench, m.unit, m.good, m.fix, m.fmt))

    score, grade, top_fix = score_checks(checks)
    phases = build_phases(k, keys, metrics, checks, v, arm, torso, face)
    return Report(True, arm=arm, contact_idx=contact, prep_idx=prep, checks=checks, score=score,
                  grade=grade, top_fix=top_fix, speed=speed, elbow=elbow, fps=fps, stroke=stroke, phases=phases,
                  snapshots=snapshots, track=track)


# ---------------------------------------------------------------- phases + "shadow" (target pose)
# One swing is shown as several snapshots. Each snapshot gets a shadow: the player's own skeleton
# (same limb lengths, same spot) bent into the benchmark position for the metrics judged at that moment.
LEGS = ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN))
PHASE_METRICS = {"Load": ("knee",), "Trophy": ("prep",), "Contact": ("extension", "height", "front"),
                 "Finish": ("follow",)}
PHASE_HELP = {"Load": "legs bent, ready to push", "Trophy": "elbow up, racket back",
              "Contact": "hit at full reach", "Finish": "racket follows through"}
SHORT = {"knee": "knee", "prep": "elbow ht", "extension": "elbow", "height": "reach", "front": "in front",
         "follow": "follow"}


@dataclass
class Phase:
    name: str
    idx: int               # frame index in the analysed sequence
    status: str            # "good" | "warn" | "na"
    notes: list            # short lines, one per metric judged at this moment
    pose: np.ndarray       # (33,2) the player's (smoothed) pose at idx
    ghost: np.ndarray      # (33,2) the target pose ("shadow"), same pixel coords
    match: float | None = None   # 0-100: how close the pose is to the target path at this moment


def metric_target(m: Metric, value) -> float | None:
    """Benchmark value to aim for; the player's own value if they already beat it. None = not measurable."""
    if value is None or np.isnan(value):
        return None
    if m.band:
        return float(np.clip(value, m.bench, m.band[0]))
    return float(max(value, m.bench) if m.bench > m.pass_line else min(value, m.bench))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, -1.0])


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _two_link(root, target, l1, l2, hint):
    """Place a 2-segment limb root -> joint -> end reaching for target; the joint bends to the side nearest `hint`."""
    d = np.asarray(target, float) - root
    u = _unit(d)
    dist = float(np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
    a = np.arccos(np.clip((l1 ** 2 + dist ** 2 - l2 ** 2) / (2 * l1 * dist), -1, 1))
    joint = min((root + l1 * _rot(u, s * a) for s in (1, -1)), key=lambda j: np.linalg.norm(j - hint))
    return joint, root + u * dist


def _bend_knees(g, xy, knee_deg: float, torso: float, face: float):
    """In place on g: keep the feet planted and lower everything above them until the knees reach knee_deg."""
    legs = [leg for leg in LEGS if not np.isnan(xy[list(leg)]).any()]
    if not legs:
        return g
    K, deltas = np.radians(knee_deg), []
    for h, kn, an in legs:
        a, b = np.linalg.norm(xy[h] - xy[kn]), np.linalg.norm(xy[kn] - xy[an])
        d = np.sqrt(a * a + b * b - 2 * a * b * np.cos(K))
        deltas.append(xy[an] + _unit(xy[h] - xy[an]) * d - xy[h])
    feet = {L_KN, R_KN, L_AN, R_AN, L_HEEL, R_HEEL, L_TOE, R_TOE}
    g[[i for i in range(33) if i not in feet]] += np.mean(deltas, axis=0)
    for h, kn, an in legs:
        a, b = np.linalg.norm(xy[h] - xy[kn]), np.linalg.norm(xy[kn] - xy[an])
        hint = xy[kn] + np.array([(face or 1) * 0.3 * torso, 0])   # knees bend forward
        g[kn], _ = _two_link(g[h], xy[an], a, b, hint)
    return g


def ghost_pose(xy, phase: str, tgt: dict, arm: str, torso: float, face: float) -> np.ndarray:
    """Target pose for one phase. `tgt` holds only the metrics that need changing (key -> target value)."""
    g = xy.copy()
    if np.isnan(xy[[L_SH, R_SH, L_HIP, R_HIP]]).any():
        return g
    sh_i, el_i, wr_i = _arm_idx(arm)
    sh, el, wr = xy[sh_i], xy[el_i], xy[wr_i]
    l1, l2 = np.linalg.norm(el - sh), np.linalg.norm(wr - el)
    arm_ok = not np.isnan([l1, l2]).any() and l1 > 1 and l2 > 1
    if phase == "Load" and "knee" in tgt:
        _bend_knees(g, xy, tgt["knee"], torso, face)
    elif phase == "Trophy" and "prep" in tgt and arm_ok:
        dy = float(np.clip(-tgt["prep"] * torso, -0.98 * l1, 0.98 * l1))
        side = np.sign(el[0] - sh[0]) or -(face or 1)
        new_el = sh + np.array([side * np.sqrt(l1 * l1 - dy * dy), dy])
        u0, u1 = el - sh, new_el - sh                                    # swing the whole arm up, forearm rides along
        turn = np.arctan2(u0[0] * u1[1] - u0[1] * u1[0], u0 @ u1)
        g[el_i], g[wr_i] = new_el, new_el + _rot(wr - el, turn)
    elif phase == "Contact" and tgt and arm_ok:
        E = np.radians(tgt.get("extension", angle(sh, el, wr)))
        r = np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * np.cos(E))         # shoulder-to-wrist reach at that elbow angle
        dy = -tgt["height"] * torso if "height" in tgt else wr[1] - sh[1]
        dx = tgt["front"] * torso * face if "front" in tgt and face else wr[0] - sh[0]
        g[el_i], g[wr_i] = _two_link(sh, sh + r * _unit(np.array([dx, dy])), l1, l2, el)
    elif phase == "Finish" and "follow" in tgt and arm_ok:
        g[el_i], g[wr_i] = _two_link(sh, (wr[0], sh[1] + tgt["follow"] * torso), l1, l2, el)
    return g


def build_phases(k, idx: dict, metrics, checks, v: dict, arm, torso, face) -> list:
    by_key = {m.key: (m, c) for m, c in zip(metrics, checks)}
    out = []
    for name, keys in PHASE_METRICS.items():
        t, tgt, notes, stats = idx[name], {}, [], []
        for key in keys:
            if key not in by_key or by_key[key][1].points is None:
                continue
            m, c = by_key[key]
            target = metric_target(m, v[key])
            f = m.fmt.format
            unit = m.unit.replace(" torso", "")
            if target is not None and abs(target - v[key]) > 1e-9:
                tgt[key] = target
            stats.append(c.status)
            notes.append(f"{SHORT[key]} {f(v[key])} ok" if c.status == "good" and key not in tgt else
                         f"{SHORT[key]} {f(v[key])} -> {f(target)}{unit}")
        status = "warn" if "warn" in stats else "good" if stats else "na"
        out.append(Phase(name, t, status, notes or ["not scored here"], k[t].copy(),
                         ghost_pose(k[t], name, tgt, arm, torso, face)))
    return sorted(out, key=lambda p: p.idx)


# ---------------------------------------------------------------- whole-swing target path
# The key moments above are judged on single frames. The target *path* covers every frame from the
# first key moment to the finish: at each key moment the benchmark sets a band for a few body
# quantities, giving a correction (how far you are outside it). Corrections are blended between key
# moments and fade out over TRACK_RADIUS_S, so the shadow is your own movement shifted by what each
# part of the swing needed: it follows you through the swing and shows where you should be. `N_SNAPSHOTS` frames spread across the swing are
# scored against it ("Whole-swing match").
PATH_Q = ("knee", "el_ht", "wr_ht", "elbow")
PATH_SCALE = {"knee": 25.0, "el_ht": 0.4, "wr_ht": 0.4, "elbow": 35.0}   # this far off the band = 0% match
PATH_SHORT = {"knee": "knee", "el_ht": "elbow ht", "wr_ht": "hand ht", "elbow": "elbow"}
PATH_FMT = {"knee": "{:.0f}", "el_ht": "{:.2f}", "wr_ht": "{:.2f}", "elbow": "{:.0f}"}
LOOSE = {"knee": (0.0, 180.0), "el_ht": (-3.0, 3.0), "wr_ht": (-3.0, 3.0), "elbow": (0.0, 180.0)}
N_SNAPSHOTS = 8


@dataclass
class Track:
    start: int             # first frame with a target (earliest key moment)
    end: int               # last frame with a target (finish)
    keys: dict             # key moment name -> frame index
    pose: np.ndarray       # [N,33,2] smoothed player pose, every frame
    ghost: np.ndarray      # [N,33,2] target pose; NaN outside [start, end]
    match: np.ndarray      # [N] 0-100 match to the target path; NaN outside [start, end]
    actual: list           # per frame {quantity: value}
    target: list           # per frame {quantity: value} (None outside [start, end])


def path_bands(metrics) -> dict:
    """{key moment: {quantity: (lo, hi)}}. Benchmark-driven where a metric exists, plus two template
    points for the trophy (elbow bent, hand up by the head). Placeholder numbers like the benchmarks."""
    m = {x.key: x for x in metrics}
    b = {ph: dict(LOOSE) for ph in PHASE_METRICS}
    b["Trophy"]["elbow"] = (0.0, 130.0)
    b["Trophy"]["wr_ht"] = (-0.1, 3.0)
    if "knee" in m:
        b["Load"]["knee"] = (0.0, m["knee"].bench)
    if "prep" in m:
        b["Trophy"]["el_ht"] = (m["prep"].bench, 3.0)
    if "extension" in m:
        b["Contact"]["elbow"] = (m["extension"].bench, 180.0)
    if "height" in m:
        b["Contact"]["wr_ht"] = (m["height"].bench, 3.0)
    if "follow" in m:
        f = m["follow"]   # follow = how far the wrist drops below the shoulder, i.e. -wr_ht
        b["Finish"]["wr_ht"] = (-3.0, -f.bench) if f.bench > f.pass_line else (-f.bench, 3.0)
    return b


def pose_quantities(xy, arm: str, torso: float) -> dict:
    sh, el, wr = _arm_idx(arm)
    kn = [angle(xy[h], xy[k], xy[a]) for h, k, a in LEGS]
    kn = [x for x in kn if not np.isnan(x)]
    return {"knee": min(kn) if kn else float("nan"),
            "el_ht": float((xy[sh, 1] - xy[el, 1]) / torso),
            "wr_ht": float((xy[sh, 1] - xy[wr, 1]) / torso),
            "elbow": angle(xy[sh], xy[el], xy[wr])}


def fit_pose(xy, tgt: dict, arm: str, torso: float, face: float) -> np.ndarray:
    """The player's pose moved to the target quantities in `tgt` (only the ones that need changing),
    keeping their limb lengths and position."""
    g = xy.copy()
    if np.isnan(xy[[L_SH, R_SH, L_HIP, R_HIP]]).any():
        return g
    if "knee" in tgt:
        _bend_knees(g, xy, tgt["knee"], torso, face)
    sh_i, el_i, wr_i = _arm_idx(arm)
    sh, el, wr = g[sh_i].copy(), g[el_i].copy(), g[wr_i].copy()
    l1, l2 = np.linalg.norm(el - sh), np.linalg.norm(wr - el)
    if np.isnan([l1, l2]).any() or l1 <= 1 or l2 <= 1:
        return g
    if "el_ht" in tgt:                    # swing the whole arm about the shoulder, forearm rides along
        dy = float(np.clip(-tgt["el_ht"] * torso, -0.98 * l1, 0.98 * l1))
        side = np.sign(el[0] - sh[0]) or -(face or 1)
        new_el = sh + np.array([side * np.sqrt(l1 * l1 - dy * dy), dy])
        u0, u1 = el - sh, new_el - sh
        turn = np.arctan2(u0[0] * u1[1] - u0[1] * u1[0], u0 @ u1)
        el, wr = new_el, new_el + _rot(wr - el, turn)
        g[el_i], g[wr_i] = el, wr
    if "elbow" in tgt or "wr_ht" in tgt:  # place the wrist at the target height and reach, solve the elbow
        E = np.radians(tgt.get("elbow", angle(sh, el, wr)))
        r = np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * np.cos(E))
        dy = -tgt["wr_ht"] * torso if "wr_ht" in tgt else wr[1] - sh[1]
        dy = float(np.clip(dy, -0.999 * r, 0.999 * r))
        side = np.sign(wr[0] - sh[0]) or np.sign(el[0] - sh[0]) or (face or 1)
        dx = side * np.sqrt(max(r * r - dy * dy, 0.0))
        g[el_i], g[wr_i] = _two_link(sh, sh + np.array([dx, dy]), l1, l2, el)
    return g


def key_corrections(actual: list, keys: dict, bands: dict) -> dict:
    """{frame: {quantity: correction}} at each key-moment frame: how far that quantity has to move to get
    into the benchmark band (0 if it is already inside). Moments on the same frame share their bands."""
    out = {}
    for t in sorted(set(keys.values())):
        names = [n for n, tk in keys.items() if tk == t]
        c = {}
        for q in PATH_Q:
            lo = max(bands[n][q][0] for n in names)
            hi = min(bands[n][q][1] for n in names)
            a = actual[t][q]
            c[q] = 0.0 if np.isnan(a) else float(np.clip(a, lo, max(lo, hi)) - a)
        out[t] = c
    return out


def path_targets(actual: dict, t: int, corr: dict, radius: float) -> dict:
    """Target quantities at frame t: the player's own value plus the key-moment corrections, blended.
    Each correction is full strength on its own frame and fades out over `radius` frames, or by the
    next key moment if that is closer, so between two close moments it slides from one to the other."""
    ts = sorted(corr)
    tg = {}
    for q in PATH_Q:
        if np.isnan(actual[q]):
            continue
        d = 0.0
        for i, tk in enumerate(ts):
            gap = (tk - ts[i - 1]) if t < tk and i > 0 else (ts[i + 1] - tk) if t > tk and i + 1 < len(ts) else radius
            r = min(radius, gap)
            d += max(0.0, 1.0 - abs(t - tk) / r) * corr[tk][q]
        tg[q] = actual[q] + d
    return tg


def path_match(actual: dict, target: dict) -> float:
    """0-100. Root-mean-square of how far each quantity is outside its band, in PATH_SCALE units."""
    devs = [min(1.5, abs(target[q] - actual[q]) / PATH_SCALE[q]) for q in PATH_Q
            if not np.isnan(actual[q]) and target.get(q) is not None]
    if not devs:
        return float("nan")
    return float(100 * np.clip(1 - np.sqrt(np.mean(np.square(devs))), 0, 1))


TRACK_RADIUS_S = 0.2   # how far either side of a key moment its correction fades in / out


def build_track(k, keys: dict, metrics, arm, torso, face, fps: float = 30.0) -> Track:
    N = len(k)
    start, end = min(keys.values()), max(keys.values())
    radius = max(1.5, TRACK_RADIUS_S * fps)
    ghost = np.full_like(k, np.nan)
    match = np.full(N, np.nan)
    actual = [pose_quantities(k[t], arm, torso) for t in range(N)]
    corr = key_corrections(actual, keys, path_bands(metrics))
    target = [None] * N
    for t in range(start, end + 1):
        tg = path_targets(actual[t], t, corr, radius)
        target[t] = tg
        match[t] = path_match(actual[t], tg)
        ghost[t] = fit_pose(k[t], {q: v for q, v in tg.items() if abs(v - actual[t][q]) > 1e-6}, arm, torso, face)
    return Track(start, end, dict(keys), k.copy(), ghost, match, actual, target)


def snapshot_idx(start: int, end: int, keys: dict, n: int = N_SNAPSHOTS) -> list:
    """n frames spread evenly from start to end, nudged so every key moment is one of them."""
    if end - start + 1 <= n:
        return list(range(start, end + 1))
    slots = list(np.round(np.linspace(start, end, n)).astype(int))
    taken = set()
    for t in sorted(set(keys.values())):
        j = min((j for j in range(n) if j not in taken), key=lambda j: abs(slots[j] - t))
        slots[j] = t
        taken.add(j)
    return sorted(set(slots))


def build_snapshots(tr: Track, contact: int, fps: float, pass_line: float) -> list:
    names = {t: n for n, t in sorted(tr.keys.items(), key=lambda kv: list(PHASE_METRICS).index(kv[0]))[::-1]}
    out = []
    for t in snapshot_idx(tr.start, tr.end, tr.keys):
        m = float(tr.match[t])
        notes = [f"match {m:.0f}%"]
        a, g = tr.actual[t], tr.target[t] or {}
        off = [(abs(g[q] - a[q]) / PATH_SCALE[q], q) for q in g if abs(g[q] - a[q]) > 1e-6]
        if off:
            q = max(off)[1]
            f = PATH_FMT[q].format
            notes.append(f"{PATH_SHORT[q]} {f(a[q])}->{f(g[q])}")
        out.append(Phase(names.get(t, f"{(t - contact) / fps:+.2f}s"), t, "good" if m >= pass_line else "warn",
                         notes, tr.pose[t].copy(), tr.ghost[t].copy(), m))
    return out


def replay_span(rep: Report, before_s: float = 0.3, after_s: float = 0.4) -> tuple:
    """Frames to replay: the tracked swing plus a little lead-in and run-out."""
    tr = rep.track
    return (max(0, tr.start - win_frames(before_s, rep.fps)),
            min(len(tr.pose) - 1, tr.end + win_frames(after_s, rep.fps)))


# ---------------------------------------------------------------- the ideal stroke (demo "shadow")
# A side-on stick figure doing the target stroke, shown before you swing so you know what to aim for.
# Units: torso lengths, x = forward, y = up, origin at the shoulders. Arm angles in degrees (0 = forward,
# 90 = up), unwrapped so the interpolation turns the way a real swing does.
DEMO_KEYS = ("Ready", "Load", "Trophy", "Contact", "Finish")
DEMO_T = (0.0, 0.28, 0.48, 0.64, 0.80)          # fraction of the loop; it returns to Ready at 1.0
DEMO_RACKET = {"smash": ((-70, 0), (-170, 120), (-195, 100), (-275, 88), (-420, -100)),
               "clear": ((-70, 0), (-170, 120), (-195, 100), (-275, 92), (-390, -60)),
               "drop": ((-70, 0), (-170, 120), (-195, 100), (-280, 80), (-350, -20))}
DEMO_OFF = ((-100, -90), (60, 70), (70, 80), (-40, -70), (-80, -90))    # non-racket arm points at the shuttle
DEMO_KNEE = (165, 140, 150, 172, 165)
DEMO_HELP = {"Ready": "racket up, on your toes", **PHASE_HELP}
_UA, _FA, _TH, _SH = 0.58, 0.52, 0.87, 0.87     # upper arm, forearm, thigh, shin


def _demo_at(u: float, stroke: str):
    """Interpolated (racket arm, off arm, knee) at loop fraction u, plus the nearest key moment's name."""
    rk = list(DEMO_RACKET.get(stroke, DEMO_RACKET["smash"]))
    rk.append((rk[0][0] - 360, rk[0][1]))
    off, knee = list(DEMO_OFF) + [DEMO_OFF[0]], list(DEMO_KNEE) + [DEMO_KNEE[0]]
    ts = list(DEMO_T) + [1.0]
    u = u % 1.0
    i = max(j for j in range(len(ts) - 1) if ts[j] <= u)
    s = (u - ts[i]) / (ts[i + 1] - ts[i])
    s = s * s * (3 - 2 * s)                                   # ease in/out
    lerp = lambda a, b: tuple(x + (y - x) * s for x, y in zip(a, b))
    near = min(range(len(DEMO_KEYS)), key=lambda j: min(abs(u - DEMO_T[j]), abs(u - DEMO_T[j] - 1)))
    return lerp(rk[i], rk[i + 1]), lerp(off[i], off[i + 1]), knee[i] + (knee[i + 1] - knee[i]) * s, DEMO_KEYS[near]


def demo_pose(u: float, stroke: str = "smash"):
    """(33,2) ideal pose at loop fraction u (x forward, y up, torso units; unused joints NaN), the racket
    tip, and the nearest key moment's name. The racket arm is the right arm."""
    (ru, rf), (ou, of), knee, name = _demo_at(u, stroke)
    p = np.full((33, 2), np.nan)
    p[NOSE] = (0.15, 0.35)
    p[R_SH], p[L_SH] = (-0.06, 0.0), (0.06, 0.0)
    p[R_HIP], p[L_HIP] = (-0.05, -1.0), (0.05, -1.0)
    p[R_AN], p[L_AN] = (-0.3, -1.0 - _TH - _SH + 0.02), (0.3, -1.0 - _TH - _SH + 0.02)
    K = np.radians(knee)
    d = np.sqrt(_TH ** 2 + _SH ** 2 - 2 * _TH * _SH * np.cos(K))
    drop = np.mean([np.linalg.norm(p[h] - p[a]) - d for h, a in ((R_HIP, R_AN), (L_HIP, L_AN))])
    for i in (NOSE, R_SH, L_SH, R_HIP, L_HIP):
        p[i] = p[i] + (0, -drop)
    for h, kn, an in ((R_HIP, R_KN, R_AN), (L_HIP, L_KN, L_AN)):
        p[kn], _ = _two_link(p[h], p[an], _TH, _SH, p[an] + (1.0, 0.5))
    pol = lambda a: np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    p[R_EL] = p[R_SH] + _UA * pol(ru); p[R_WR] = p[R_EL] + _FA * pol(rf)
    p[L_EL] = p[L_SH] + _UA * pol(ou); p[L_WR] = p[L_EL] + _FA * pol(of)
    p[R_TOE], p[L_TOE] = p[R_AN] + (0.22, -0.05), p[L_AN] + (0.22, -0.05)
    return p, p[R_WR] + 0.75 * pol(rf + 15), name


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


GHOST_COL = (255, 255, 255)
ARROW_COL = (0, 230, 255)
STATUS_COL = {"good": (0, 140, 0), "warn": (0, 110, 230), "na": (90, 90, 90)}


def draw_ghost(img, ghost, pose=None, scale_px: float = 0.0, alpha=0.55, thick=9):
    """Translucent 'shadow' skeleton of the target pose, plus arrows from the player's joints to where they
    should be (only for joints that are noticeably off)."""
    if np.isnan(ghost[[L_SH, R_SH]]).any():
        return img
    ov = img.copy()
    for a, b in SKELETON:
        if not (np.isnan(ghost[a]).any() or np.isnan(ghost[b]).any()):
            cv2.line(ov, tuple(int(v) for v in ghost[a]), tuple(int(v) for v in ghost[b]), GHOST_COL, thick, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    if pose is not None:
        for i in (L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN):
            p, q = pose[i], ghost[i]
            if not (np.isnan(p).any() or np.isnan(q).any()) and np.linalg.norm(q - p) > max(6.0, scale_px):
                cv2.arrowedLine(img, tuple(int(v) for v in p), tuple(int(v) for v in q), ARROW_COL, 2, cv2.LINE_AA,
                                tipLength=0.3)
    return img


def _crop(frame_shape, pts, size, extra_h: float = 60):
    """Affine (2x3) that fits the points (the player + shadow) into a tile of `size`, plus its scale."""
    tw, th = size
    pts = pts[~np.isnan(pts).any(axis=1)]
    H, W = frame_shape[:2]
    if len(pts):
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * 1.3 + 20, (y1 - y0) * 1.25 + extra_h      # room for the header / notes
    s = min(tw / bw, th / bh)
    return np.float32([[s, 0, tw / 2 - cx * s], [0, s, th / 2 - cy * s + 8]]), s


def _overlay(frame, M, size, pose, ghost, arm, mirror, arrows=True):
    """Warp the frame into the tile and draw the shadow, then the player, on it."""
    tw, th = size
    im = cv2.warpAffine(frame, M, (tw, th), borderValue=(25, 25, 25))
    tf = lambda p: np.c_[p[:, 0] * M[0, 0] + M[0, 2], p[:, 1] * M[1, 1] + M[1, 2]]
    pose, ghost = tf(pose), tf(ghost)
    if mirror:
        im = cv2.flip(im, 1)
        pose[:, 0], ghost[:, 0] = tw - 1 - pose[:, 0], tw - 1 - ghost[:, 0]
    torso = np.linalg.norm((pose[L_SH] + pose[R_SH]) / 2 - (pose[L_HIP] + pose[R_HIP]) / 2)
    draw_ghost(im, ghost, pose if arrows else None, 0.08 * torso if not np.isnan(torso) else 0,
               thick=max(4, int(tw / 30)))
    draw_pose(im, pose, highlight_arm=arm, thick=2)
    return im


def phase_tile(frame, ph, n: int, arm: str, size: tuple, mirror: bool = False) -> np.ndarray:
    """One snapshot of the swing: the frame cropped around the player, their skeleton over the target shadow,
    a numbered header and the per-metric notes for that moment."""
    tw, th = size
    M, _ = _crop(frame.shape, np.vstack([ph.pose, ph.ghost]), size)
    im = _overlay(frame, M, size, ph.pose, ph.ghost, arm, mirror)
    cv2.rectangle(im, (0, 0), (tw, 24), STATUS_COL[ph.status], -1)
    title = ph.name.upper() if ph.name in PHASE_HELP else ph.name
    fs = 0.55 if tw >= 160 else 0.45
    cv2.putText(im, f"{n}. {title}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    hint = PHASE_HELP.get(ph.name, "")
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    if hw < tw - 110:
        cv2.putText(im, hint, (tw - hw - 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)
    ns = 0.42 if tw >= 160 else 0.36
    y = th - 8 - 18 * (len(ph.notes) - 1)
    for line in ph.notes:
        put_text(im, line, (6, y), ns, (255, 255, 255), (40, 40, 40), 1)
        y += 18
    cv2.rectangle(im, (0, 0), (tw - 1, th - 1), (60, 60, 60), 1)
    return im


def phase_strip(frames, rep: Report, size=(240, 270), mirror=False) -> list:
    """Tiles for every key moment of `rep`; `frames` is indexable by frame index."""
    return [phase_tile(frames[p.idx], p, i + 1, rep.arm, size, mirror) for i, p in enumerate(rep.phases)]


def snapshot_strip(frames, rep: Report, size=(160, 220), mirror=False) -> list:
    """Tiles for the snapshots spread across the whole swing, each with its match to the target path."""
    return [phase_tile(frames[p.idx], p, i + 1, rep.arm, size, mirror) for i, p in enumerate(rep.snapshots)]


def _match_col(m, pass_line=70.0):
    if m is None or np.isnan(m):
        return (90, 90, 90)
    return (0, 160, 0) if m >= pass_line else (0, 150, 230) if m >= pass_line - 25 else (0, 60, 200)


def replay_tile(frames, rep: Report, i: int, size: tuple, mirror: bool = False, span=None) -> np.ndarray:
    """Frame i of the swing replay: the player with the target shadow moving along with them, a header with
    the time (and key moment, if this is one) and a timeline coloured by how well each frame matched."""
    tw, th = size
    tr = rep.track
    a, b = span or replay_span(rep)
    ghost = tr.ghost[i] if not np.isnan(tr.ghost[i]).all() else np.full((33, 2), np.nan)
    M, _ = _crop(frames[i].shape, np.vstack([tr.pose[a:b + 1].reshape(-1, 2), tr.ghost[a:b + 1].reshape(-1, 2)]),
                 size, extra_h=90)
    im = _overlay(frames[i], M, size, tr.pose[i], ghost, rep.arm, mirror)
    m = tr.match[i]
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(im, (0, 0), (tw, 26), _match_col(m), -1)
    cv2.putText(im, f"REPLAY {(i - rep.contact_idx) / rep.fps:+.2f}s", (8, 18), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    key = [n for n, t in tr.keys.items() if t == i]
    if key:
        lab = f"{key[-1].upper()}: {PHASE_HELP[key[-1]]}"
        (lw, _), _ = cv2.getTextSize(lab, F, 0.45, 1)
        cv2.putText(im, lab, (max(8, tw - lw - 8), 18), F, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # timeline
    x0, x1, y0, y1 = 8, tw - 8, th - 16, th - 6
    n = max(1, b - a + 1)
    xs = lambda j: int(x0 + (x1 - x0) * (j - a) / n)
    for j in range(a, b + 1):
        cv2.rectangle(im, (xs(j), y0), (max(xs(j) + 1, xs(j + 1) - 1), y1), _match_col(tr.match[j]), -1)
    for name, t in tr.keys.items():
        if a <= t <= b:
            cv2.putText(im, name[0], (xs(t) + 1, y0 - 3), F, 0.35, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(im, (xs(i), y0 - 3), (xs(i) + 2, y1 + 3), (255, 255, 255), -1)
    put_text(im, f"match {m:.0f}%" if not np.isnan(m) else "lead-in / run-out", (8, y0 - 18), 0.45,
             (255, 255, 255), (40, 40, 40), 1)
    cv2.rectangle(im, (0, 0), (tw - 1, th - 1), (60, 60, 60), 1)
    return im


def draw_demo(img, stroke: str, u: float, rect: tuple, arm: str = "right", title: bool = True):
    """Draw the ideal stroke at loop fraction u as a shadow figure inside rect (x, y, w, h) of img."""
    x, y, w, h = rect
    roi = img[y:y + h, x:x + w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    p, tip, name = demo_pose(u, stroke)
    top, bottom = 1.75, -2.85                     # figure extent in torso units (incl. racket overhead)
    head = 24 if title else 4
    s = (h - head - 22) / (top - bottom)
    flip = -1 if arm == "left" else 1
    cx = x + w / 2
    to_px = lambda q: np.c_[cx + flip * q[:, 0] * s, y + head + (top - q[:, 1]) * s]
    P, T = to_px(p), to_px(tip[None])[0]
    if arm == "left":                             # mirror the figure so the racket arm is the left one
        P[[L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_TOE, R_TOE]] = \
            P[[R_SH, L_SH, R_EL, L_EL, R_WR, L_WR, R_HIP, L_HIP, R_KN, L_KN, R_AN, L_AN, R_TOE, L_TOE]]
    draw_ghost(img, P, thick=max(4, int(s * 0.16)), alpha=0.75)
    _, _, wr_i = _arm_idx(arm)
    cv2.line(img, tuple(int(v) for v in P[wr_i]), tuple(int(v) for v in T), (0, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(int(v) for v in T), max(3, int(s * 0.12)), (0, 200, 255), 1, cv2.LINE_AA)
    F = cv2.FONT_HERSHEY_SIMPLEX
    if title:
        cv2.putText(img, f"IDEAL {stroke.upper()}", (x + 6, y + 17), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    lab = f"{name.upper()}: {DEMO_HELP[name]}"
    sc = 0.38 if cv2.getTextSize(lab, F, 0.38, 1)[0][0] < w - 8 else 0.32
    cv2.putText(img, lab, (x + 5, y + h - 7), F, sc, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (200, 200, 200), 1)
    return img


def demo_strip(stroke: str, size=(150, 230), arm: str = "right") -> np.ndarray:
    """The ideal stroke's key moments side by side (for the web UI)."""
    tw, th = size
    out = np.full((th, tw * len(DEMO_KEYS), 3), 30, np.uint8)
    for j, u in enumerate(DEMO_T):
        draw_demo(out, stroke, u, (j * tw, 0, tw, th), arm, title=False)
    return out


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), bg=(0, 0, 0), thick=2):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + b + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def annotate_video(frames, kps, rep: Report, fps: float, out_path: str):
    """Write annotated video with skeleton, live elbow angle, phase labels and result banner."""
    h, w = frames[0].shape[:2]
    k = _smooth(_interp_nan(kps), win_frames(0.1, fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    hold = int(0.8 * fps)
    for t, fr in enumerate(frames):
        im = fr.copy()
        tr = rep.track if rep.ok else None
        if tr is not None and tr.start <= t <= tr.end:
            draw_ghost(im, tr.ghost[t], alpha=0.45)
            put_text(im, f"shadow = target  |  match {tr.match[t]:.0f}%", (20, 75), 0.6, (255, 255, 255),
                     _match_col(tr.match[t]), 1)
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
                put_text(im, f"{rep.stroke.title()} grade {rep.grade}  ({rep.score:.0f}/100)", (20, h - 20), 0.9, (255, 255, 255),
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
