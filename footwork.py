"""6-point footwork drill: court calibration, call-outs and per-rep movement scoring.

Geometry: you click the 4 corners of your training area once; a homography maps foot positions
from the image onto the floor in metres. Court coords: x = 0 (left) .. W (right) as you face the net,
y = 0 (net side) .. D (back). Works for any rectangle - a full half court or a small room - and the
target zones and time benchmarks scale with the area size.

Per rep: base -> called corner -> back to base. Measured from pose (MediaPipe):
  reaction time, split step, move time, path directness, staying low, lunge depth,
  recovery time, and for side targets whether you stay square to the net + crossover steps.

NOTE: benchmarks are coaching-heuristic placeholders, not validated norms (same as the smash ones).
"""
from __future__ import annotations

import json
import os
import random
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

import analysis as A

COURT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "court.json")
CORNER_PROMPTS = ["NET-LEFT", "NET-RIGHT", "BACK-RIGHT", "BACK-LEFT"]   # left/right as you face the net
TARGET_NAMES = ["FRONT LEFT", "FRONT RIGHT", "SIDE LEFT", "SIDE RIGHT", "BACK LEFT", "BACK RIGHT"]
SMALL_AREA_M = 2.5


# ---------------------------------------------------------------- court
@dataclass
class Court:
    corners: list            # 4 image points (raw camera frame), order = CORNER_PROMPTS
    width_m: float           # side to side
    depth_m: float           # net to back
    image_size: tuple = (0, 0)   # (w, h) of the frame the corners were clicked on
    H: np.ndarray = field(init=False, repr=False)
    Hinv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        W, D = self.width_m, self.depth_m
        src = np.float32(self.corners)
        dst = np.float32([[0, 0], [W, 0], [W, D], [0, D]])
        self.H = cv2.getPerspectiveTransform(src, dst).astype(float)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(M, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        q = np.c_[p, np.ones(len(p))] @ M.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return (q[:, :2] / q[:, 2:]).reshape(np.shape(pts))

    def to_court(self, pts):
        return self._apply(self.H, pts)

    def to_image(self, pts):
        return self._apply(self.Hinv, pts)

    # target layout, all scaled to the area
    @property
    def small(self) -> float:
        return min(self.width_m, self.depth_m)

    @property
    def margin(self) -> float:
        return float(np.clip(0.12 * self.small, 0.3, 0.8))

    @property
    def zone_r(self) -> float:
        return float(np.clip(0.12 * self.small, 0.3, 0.6))

    @property
    def base_r(self) -> float:
        return float(np.clip(0.15 * self.small, 0.35, 0.7))

    @property
    def base(self) -> np.ndarray:
        return np.array([self.width_m / 2, self.depth_m / 2])

    @property
    def targets(self) -> dict:
        W, D, m = self.width_m, self.depth_m, self.margin
        xs = {"LEFT": m, "RIGHT": W - m}
        ys = {"FRONT": m, "SIDE": D / 2, "BACK": D - m}
        return {n: np.array([xs[n.split()[1]], ys[n.split()[0]]]) for n in TARGET_NAMES}

    def save(self, path: str | None = None):
        path = path or COURT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(corners=[list(map(float, c)) for c in self.corners], width_m=self.width_m,
                           depth_m=self.depth_m, image_size=list(self.image_size)), f, indent=1)

    @classmethod
    def load(cls, path: str | None = None) -> "Court | None":
        path = path or COURT_PATH
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(d["corners"], d["width_m"], d["depth_m"], tuple(d.get("image_size", (0, 0))))
        except (OSError, ValueError, KeyError):
            return None


# ---------------------------------------------------------------- config
@dataclass
class FootworkConfig:
    # Benchmarks (pass line -> 60 pts, benchmark -> 100 pts). DEMO placeholders - tune on real players.
    max_reaction_s: float = 0.50        # call -> first visible movement (includes camera/pose lag)
    bench_reaction_s: float = 0.28
    pass_move_speed: float = 1.6        # move time limits are distance / speed (+ allowance), so they scale with the area
    bench_move_speed: float = 3.0       # m/s
    pass_recover_speed: float = 1.3
    bench_recover_speed: float = 2.4
    recover_allow_s: float = 0.6        # extra time allowed for the lunge + push-back at the pass line
    bench_recover_allow_s: float = 0.25
    max_lunge_knee_deg: float = 135.0   # most-bent knee at the corner
    bench_lunge_knee_deg: float = 105.0
    min_low_drop: float = 0.03          # how much lower than standing you move (fraction of standing leg height)
    bench_low_drop: float = 0.12
    min_split_amp: float = 0.03         # hip bounce just before moving (torso lengths)
    bench_split_amp: float = 0.10
    min_path_eff: float = 0.75          # straight-line distance / distance actually travelled
    bench_path_eff: float = 0.93
    min_square: float = 0.60            # side moves: shoulder width vs standing (1 = still square to the net)
    bench_square: float = 0.85
    # --- drill flow ---
    base_hold_s: float = 0.6            # stand still at base this long before the next call
    call_delay_s: tuple = (0.4, 1.2)    # random wait after settling, so you can't anticipate
    move_thresh_m: float = 0.15         # feet moved this far = movement started
    timeout_s: float = 4.0
    ema: float = 0.5                    # smoothing for foot positions (0..1, higher = less smoothing)


WEIGHTS = {"reaction": 0.15, "move": 0.25, "recover": 0.20, "lunge": 0.15, "low": 0.08,
           "split": 0.07, "path": 0.10, "square": 0.10}


@dataclass
class Rep:
    idx: int
    target: str
    score: float
    grade: str
    checks: list
    top_fix: str
    times: dict
    crossovers: int | None = None

    def detail(self) -> dict:
        return dict(target=self.target, score=round(self.score, 1),
                    metrics={c.name: (None if c.value is None else round(c.value, 3)) for c in self.checks})


# ---------------------------------------------------------------- per-frame features
def _foot(xy, ids):
    pts = xy[list(ids)]
    good = ~np.isnan(pts).any(axis=1)
    return pts[good].mean(axis=0) if good.any() else np.array([np.nan, np.nan])


def feet_visible(xy) -> bool:
    return all(not np.isnan(xy[i]).any() for i in (A.L_HIP, A.R_HIP)) and \
        not np.isnan(_foot(xy, (A.L_AN, A.L_HEEL, A.L_TOE))).any() and \
        not np.isnan(_foot(xy, (A.R_AN, A.R_HEEL, A.R_TOE))).any()


def frame_features(xy, court: Court) -> dict:
    lf, rf = _foot(xy, (A.L_AN, A.L_HEEL, A.L_TOE)), _foot(xy, (A.R_AN, A.R_HEEL, A.R_TOE))
    feet = court.to_court(np.stack([lf, rf]))
    hip = (xy[A.L_HIP] + xy[A.R_HIP]) / 2
    sh = (xy[A.L_SH] + xy[A.R_SH]) / 2
    torso = float(np.linalg.norm(sh - hip))
    if not torso > 5:
        torso = np.nan
    ank_y = np.nanmean([lf[1], rf[1]]) if not (np.isnan(lf[1]) and np.isnan(rf[1])) else np.nan
    knees = [A.angle(xy[h], xy[k], xy[a]) for h, k, a in ((A.L_HIP, A.L_KN, A.L_AN), (A.R_HIP, A.R_KN, A.R_AN))]
    return dict(feet=feet, hip_y=hip[1] / torso, leg=(ank_y - hip[1]) / torso,
                shw=float(np.linalg.norm(xy[A.L_SH] - xy[A.R_SH])) / torso, knee=np.nanmin(knees)
                if not np.isnan(knees).all() else np.nan)


# ---------------------------------------------------------------- drill engine
class FootworkDrill:
    """Frame-by-frame state machine. Feed it update(t_seconds, keypoints[33,2]); it returns events:
        ("call", target)    - announce this corner (call mode only)
        ("rep", Rep)        - a completed, scored rep
        ("miss", reason)    - rep thrown away (wrong corner, timeout, ...)
        ("done", None)      - reached the rep count
    calls=False ("free mode", used for recorded video): no call-outs; a rep is any trip from base to a
    corner and back, and reaction time is not measured.
    """

    def __init__(self, court: Court, cfg: FootworkConfig | None = None, calls: bool = True,
                 reps: int | None = 12, seed: int | None = None):
        self.court, self.cfg, self.calls, self.n_reps = court, cfg or FootworkConfig(), calls, reps
        self.rng = random.Random(seed)
        self.trace: deque = deque(maxlen=2000)   # (t, features)
        self.results: list[Rep] = []
        self.misses: list[str] = []
        self.state = "settle"
        self.hint = ""
        self.cur: dict | None = None
        self.settle_since = None
        self.call_at = None
        self.base_leg = self.base_shw = None
        self.mid = self.feet = None
        self.last_target = None
        self.pending_ms = None       # free mode: time feet first moved off-centre

    # ------------------------------------------------------------ helpers
    def _window(self, t0, t1):
        return [f for (t, f) in self.trace if t0 <= t <= t1]

    def _pick(self):
        opts = [n for n in TARGET_NAMES if n != self.last_target]
        self.last_target = self.rng.choice(opts)
        return self.last_target

    def _closest(self, name):
        if self.feet is None:
            return np.inf
        d = np.linalg.norm(self.feet - self.court.targets[name], axis=1)
        return float(np.nanmin(d)) if not np.isnan(d).all() else np.inf

    # ------------------------------------------------------------ main update
    def update(self, t: float, xy: np.ndarray) -> list:
        ev = []
        if self.state == "done":
            return ev
        c, cfg = self.court, self.cfg
        f = frame_features(xy, c)
        feet = f["feet"]
        if not np.isnan(feet).any():
            self.feet = feet if self.feet is None else cfg.ema * feet + (1 - cfg.ema) * self.feet
            self.mid = self.feet.mean(axis=0)
            f["mid"] = self.mid.copy()
            f["feet"] = self.feet.copy()
            self.trace.append((t, f))
            visible = True
        else:
            visible = False
        at_base = visible and np.linalg.norm(self.mid - c.base) <= c.base_r

        st = self.state
        if st == "settle":
            self.hint = "Go to the base (centre circle) and stand still"
            if not at_base:
                self.settle_since = None
            else:
                self.settle_since = self.settle_since if self.settle_since is not None else t
                if t - self.settle_since >= cfg.base_hold_s:
                    w = self._window(self.settle_since, t)
                    legs = [x["leg"] for x in w if not np.isnan(x["leg"])]
                    shws = [x["shw"] for x in w if not np.isnan(x["shw"])]
                    if legs:
                        self.base_leg = float(np.median(legs))
                    if shws:
                        self.base_shw = float(np.median(shws))
                    if self.calls:
                        self.state, self.call_at = "wait_call", t + self.rng.uniform(*cfg.call_delay_s)
                    else:
                        self.state, self.pending_ms = "ready", None
        elif st == "wait_call":
            self.hint = "Ready..."
            if visible and not at_base:
                self.hint, self.state, self.settle_since = "Wait for the call!", "settle", None
            elif t >= self.call_at and visible:
                name = self._pick()
                self.cur = dict(target=name, t_call=t, start=self.mid.copy(), t_move=None)
                self.state = "moving"
                ev.append(("call", name))
        elif st == "ready":   # free mode
            self.hint = "Move to any corner and back"
            if visible:
                d = np.linalg.norm(self.mid - c.base)
                if d > cfg.move_thresh_m and self.pending_ms is None:
                    self.pending_ms = t
                elif d <= cfg.move_thresh_m:
                    self.pending_ms = None
                if not at_base:
                    self.cur = dict(target=None, t_call=None, start=c.base.copy(),
                                    t_move=self.pending_ms if self.pending_ms is not None else t)
                    self.state = "moving"
        elif st == "moving":
            cur = self.cur
            self.hint = f"GO: {cur['target']}" if cur["target"] else "Moving..."
            t0 = cur["t_call"] if cur["t_call"] is not None else cur["t_move"]
            if visible and cur["t_move"] is None and np.linalg.norm(self.mid - cur["start"]) > cfg.move_thresh_m:
                cur["t_move"] = t
            hit = None
            if visible:
                ds = {n: self._closest(n) for n in TARGET_NAMES}
                n = min(ds, key=ds.get)
                if ds[n] <= c.zone_r:
                    hit = n
            if hit:
                if cur["target"] and hit != cur["target"]:
                    ev.append(self._miss(f"Wrong corner - went {hit}, call was {cur['target']}", t))
                else:
                    cur["target"] = hit
                    cur["t_arrive"] = t
                    if cur["t_move"] is None:
                        cur["t_move"] = t
                    self.state = "lunge"
            elif t - t0 > cfg.timeout_s:
                ev.append(self._miss("Too slow / lost tracking - didn't reach the corner", t))
        elif st == "lunge":
            cur = self.cur
            self.hint = "Push back to base!"
            left = visible and self._closest(cur["target"]) > c.zone_r + 0.1
            if left or at_base or t - cur["t_arrive"] > cfg.timeout_s:
                cur["t_leave"] = t
                self.state = "recover"
        if self.state == "recover":
            cur = self.cur
            self.hint = "Back to base!"
            done_back = at_base
            if done_back or t - cur["t_arrive"] > cfg.timeout_s + 2:
                cur["t_back"] = t
                rep = self._score(cur, returned=done_back)
                self.results.append(rep)
                ev.append(("rep", rep))
                self.state, self.settle_since, self.cur = "settle", (t if at_base else None), None
                if self.n_reps and len(self.results) >= self.n_reps:
                    self.state = "done"
                    ev.append(("done", None))
        return ev

    def _miss(self, reason, t):
        self.misses.append(reason)
        self.state, self.settle_since, self.cur = "settle", None, None
        return ("miss", reason)

    # ------------------------------------------------------------ scoring
    def _score(self, cur: dict, returned: bool) -> Rep:
        c, cfg = self.court, self.cfg
        name = cur["target"]
        tgt = c.targets[name]
        d_eff = max(float(np.linalg.norm(tgt - c.base)) - c.zone_r, 0.3)
        tm, ta, tl, tb = cur["t_move"], cur["t_arrive"], cur["t_leave"], cur["t_back"]
        checks = []

        def add(key, *a, **kw):
            checks.append(A.make_check(WEIGHTS[key], *a, **kw))

        # 1. reaction
        if cur["t_call"] is not None:
            add("reaction", "Reaction time", tm - cur["t_call"], cfg.max_reaction_s, cfg.bench_reaction_s, " s",
                "quick first step.", "react faster - stay on your toes and push off as soon as you hear the call.",
                "{:.2f}")
        # 2. split step (hip bounce just before the first step)
        t_s0 = (cur["t_call"] if cur["t_call"] is not None else tm) - 0.5
        hy = [f["hip_y"] for f in self._window(t_s0, tm) if not np.isnan(f["hip_y"])]
        add("split", "Split step", (max(hy) - min(hy)) if len(hy) >= 3 else None, cfg.min_split_amp,
            cfg.bench_split_amp, " torso", "good split step before moving.",
            "do a small split-step hop as the call comes, land and push off from it.", "{:.2f}")
        # 3. move time, benchmarks scaled with the distance to this corner
        add("move", "Move time to corner", ta - tm, d_eff / cfg.pass_move_speed + 0.2,
            d_eff / cfg.bench_move_speed, " s", "fast to the corner.",
            "get there faster - bigger first step, then chasse; don't run with small steps.", "{:.2f}")
        # 4. directness of the path
        mids = [f["mid"] for f in self._window(tm, ta)]
        if len(mids) >= 3:
            seg = np.linalg.norm(np.diff(np.array(mids), axis=0), axis=1).sum()
            straight = float(np.linalg.norm(mids[-1] - mids[0]))
            eff = straight / seg if seg > 0.05 else None
        else:
            eff = None
        add("path", "Direct path", eff, cfg.min_path_eff, cfg.bench_path_eff, "",
            "efficient, direct line.", "you're curving or stuttering - move in a straight line to the corner.",
            "{:.2f}")
        # 5. staying low while moving
        legs = [f["leg"] for f in self._window(tm, ta) if not np.isnan(f["leg"])]
        low = 1 - float(np.mean(legs)) / self.base_leg if legs and self.base_leg else None
        add("low", "Staying low", low, cfg.min_low_drop, cfg.bench_low_drop, "",
            "good low, athletic movement.", "you're moving upright - bend the knees and keep your centre of gravity low.",
            "{:.2f}")
        # 6. lunge depth at the corner
        kn = [f["knee"] for f in self._window(ta - 0.2, tl + 0.1) if not np.isnan(f["knee"])]
        add("lunge", "Lunge depth", min(kn) if kn else None, cfg.max_lunge_knee_deg, cfg.bench_lunge_knee_deg,
            " deg", "solid lunge.", "lunge deeper - long last step, front knee bent over the foot.")
        # 7. recovery to base
        rec = tb - ta if returned else max(tb - ta, cfg.timeout_s + 2)
        add("recover", "Recovery to base", rec, d_eff / cfg.pass_recover_speed + cfg.recover_allow_s,
            d_eff / cfg.bench_recover_speed + cfg.bench_recover_allow_s, " s", "quick recovery.",
            "push back off the front foot straight away and get back to base.", "{:.2f}")
        crossovers = None
        if name.startswith("SIDE"):
            # 8. stay square to the net on side moves; count crossover steps (info)
            sw = [f["shw"] for f in self._window(tm, tl) if not np.isnan(f["shw"])]
            sq = float(np.percentile(sw, 20)) / self.base_shw if sw and self.base_shw else None
            add("square", "Square to the net", sq, cfg.min_square, cfg.bench_square, "",
                "chest stays facing the net.", "you're turning your body - stay square, side-step (chasse) across.",
                "{:.2f}")
            crossed = [bool(f["feet"][0, 0] - f["feet"][1, 0] > 0.05) for f in self._window(tm, ta)]
            crossovers = sum(1 for a, b in zip(crossed, crossed[1:]) if b and not a)
            checks.append(A.Check("Side-step pattern", float(crossovers), "", "info",
                                  f"{crossovers} crossover step(s) - " +
                                  ("pure chasse." if crossovers == 0 else "fine to finish with one, but chasse first.")))
        score, grade, top_fix = A.score_checks(checks)
        times = dict(reaction=None if cur["t_call"] is None else tm - cur["t_call"], move=ta - tm, recover=rec)
        return Rep(len(self.results) + 1, name, score, grade, checks, top_fix, times, crossovers)


# ---------------------------------------------------------------- session summary
def summarize(reps: list[Rep]) -> dict:
    if not reps:
        return dict(n=0)
    scores = np.array([r.score for r in reps])
    by_t = {}
    for r in reps:
        by_t.setdefault(r.target, []).append(r.score)
    per_target = {k: float(np.mean(v)) for k, v in by_t.items()}
    metric_pts = {}
    for r in reps:
        for ch in r.checks:
            if ch.points is not None:
                metric_pts.setdefault(ch.name, []).append(ch.points)
    metric_avg = {k: float(np.mean(v)) for k, v in metric_pts.items()}
    weakest_metric = min(metric_avg, key=metric_avg.get) if metric_avg else ""
    tips = {}
    for r in reps:
        for ch in r.checks:
            if ch.name == weakest_metric and ch.status == "warn":
                tips[ch.name] = ch.message.split(" - ", 1)[-1]
    avg = float(scores.mean())

    def mean_time(k):
        v = [r.times[k] for r in reps if r.times.get(k) is not None]
        return float(np.mean(v)) if v else None

    return dict(n=len(reps), avg=avg, grade=A.letter(avg), best=float(scores.max()), worst=float(scores.min()),
                consistency=float(scores.std()), per_target=per_target, metric_avg=metric_avg,
                weakest_corner=min(per_target, key=per_target.get), weakest_metric=weakest_metric,
                tip=tips.get(weakest_metric, ""), reaction=mean_time("reaction"), move=mean_time("move"),
                recover=mean_time("recover"))


# ---------------------------------------------------------------- drawing
ORANGE, GREEN, GREY, WHITE, BLUE = (0, 140, 255), (0, 190, 0), (150, 150, 150), (255, 255, 255), (200, 120, 0)


def _circle_pts(center, r, n=32):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], axis=1)


def draw_floor(img, court: Court, active: str | None = None, feet=None):
    """Project the area, base and target zones onto the camera image."""
    poly = court.to_image(np.array([[0, 0], [court.width_m, 0], [court.width_m, court.depth_m], [0, court.depth_m]]))
    cv2.polylines(img, [poly.astype(np.int32)], True, WHITE, 2, cv2.LINE_AA)
    cv2.polylines(img, [court.to_image(_circle_pts(court.base, court.base_r)).astype(np.int32)], True, BLUE, 2,
                  cv2.LINE_AA)
    for n, p in court.targets.items():
        pts = court.to_image(_circle_pts(p, court.zone_r)).astype(np.int32)
        if n == active:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], ORANGE)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [pts], True, ORANGE if n == active else GREY, 2, cv2.LINE_AA)
    net = court.to_image(np.array([[0, 0], [court.width_m, 0]])).astype(int)
    A.put_text(img, "NET", tuple((net.mean(axis=0) + [-20, -8]).astype(int)), 0.5, bg=(60, 60, 60), thick=1)
    return img


def draw_minimap(img, court: Court, drill: "FootworkDrill | None" = None, active=None, origin=None,
                 height=200, labels: dict | None = None):
    """Top-down view: net at the top. labels = {target: text} drawn at each target (e.g. average score)."""
    W, D = court.width_m, court.depth_m
    s = height / D
    w = int(W * s)
    h_img, w_img = img.shape[:2]
    ox, oy = origin if origin else (w_img - w - 20, 20)
    cv2.rectangle(img, (ox - 8, oy - 8), (ox + w + 8, oy + height + 8), (30, 30, 30), -1)
    cv2.rectangle(img, (ox, oy), (ox + w, oy + height), WHITE, 1)
    cv2.line(img, (ox, oy), (ox + w, oy), (220, 220, 220), 3)
    P = lambda p: (int(ox + p[0] * s), int(oy + p[1] * s))
    cv2.circle(img, P(court.base), int(court.base_r * s), BLUE, 1, cv2.LINE_AA)
    for n, p in court.targets.items():
        cv2.circle(img, P(p), max(4, int(court.zone_r * s)), ORANGE if n == active else GREY,
                   -1 if n == active else 1, cv2.LINE_AA)
        if labels and n in labels:
            cv2.putText(img, labels[n], (P(p)[0] - 12, P(p)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1,
                        cv2.LINE_AA)
    if drill is not None:
        trail = [f["mid"] for t, f in list(drill.trace)[-40:]]
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, P(a), P(b), (0, 200, 255), 1, cv2.LINE_AA)
        if drill.feet is not None:
            for ft in drill.feet:
                if not np.isnan(ft).any():
                    cv2.circle(img, P(ft), 4, GREEN, -1, cv2.LINE_AA)
    return img


def rep_line(r: Rep) -> str:
    bits = [f"Rep {r.idx} {r.target}: {r.score:.0f} ({r.grade})"]
    if r.times.get("reaction") is not None:
        bits.append(f"react {r.times['reaction']:.2f}s")
    bits.append(f"move {r.times['move']:.2f}s")
    bits.append(f"back {r.times['recover']:.2f}s")
    return "  ".join(bits)


def analyze_video(frames, kps, fps, court: Court, cfg: FootworkConfig | None = None, out_path: str | None = None):
    """Free-mode analysis of a recorded clip. Returns (reps, misses); optionally writes an annotated video."""
    drill = FootworkDrill(court, cfg, calls=False, reps=None)
    wr = None
    if out_path:
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    last = None
    for i, xy in enumerate(kps):
        for kind, val in drill.update(i / fps, xy):
            if kind == "rep":
                last = val
        if wr is not None:
            im = frames[i].copy()
            active = drill.cur["target"] if drill.cur else None
            draw_floor(im, court, active)
            A.draw_pose(im, xy)
            draw_minimap(im, court, drill, active)
            A.put_text(im, drill.hint, (20, 40), 0.8, bg=(60, 60, 60))
            if last:
                A.put_text(im, rep_line(last), (20, im.shape[0] - 20), 0.6, bg=(0, 110, 0) if last.score >= 70
                           else (0, 110, 200))
            wr.write(im)
    if wr is not None:
        wr.release()
    return drill.results, drill.misses


def default_corners(w, h):
    """A rough starting guess for the 4 corners (camera behind / above the base line)."""
    return [[0.3 * w, 0.45 * h], [0.7 * w, 0.45 * h], [0.95 * w, 0.95 * h], [0.05 * w, 0.95 * h]]
