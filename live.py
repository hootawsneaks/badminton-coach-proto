"""Real-time analysis: rolling buffer + automatic swing detection.

`LiveAnalyzer.process(bgr)` draws the skeleton and, once a swing is detected, runs the same
`analyze()` as the offline path and grades the stroke (smash / clear / drop) against its benchmark.

Flow: get ready -> swing -> the screen freezes on your contact frame with a grade card ->
click REDO (or press r) to try again.

Run standalone:  python live.py                    (smash, webcam 0)
                 python live.py --drill drop       (or clear)
                 python live.py --lite             (faster, lighter pose model -> higher frame rate)
                 python live.py clip.mp4           (replay a file in real time)
                 python live.py --drill footwork   (6-point footwork with call-outs, see footwork.py)
Keys: d = next drill (smash -> clear -> drop -> footwork), r = redo / new session, c = recalibrate the court (footwork), q = quit.
Every rep is saved to data/history.json so the result cards can compare against your best rep / session.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import analysis as A
import footwork as FW
from history import History


DRILLS = (*A.STROKES, "footwork")   # order the d key cycles through
GRADE_COL = {"A": (0, 150, 0), "B": (60, 150, 0), "C": (0, 140, 200), "D": (0, 90, 220), "F": (0, 0, 180)}


class LiveAnalyzer:
    """Three states:
      arming - just (re)started; ignore motion for `arm_s` so walking into position isn't scored
      ready  - watching for a swing
      result - swing captured: the display freezes on the contact frame with the grade card until reset()
    """

    def __init__(self, cfg: A.Config | None = None, buffer_s: float = 2.5, arm_s: float = 1.5, lite: bool = False,
                 history: History | None = None, stroke: str = "smash"):
        self.cfg = cfg or A.Config()
        self.stroke = stroke
        self.session = history.start(stroke) if history else None
        self.lm = A.make_landmarker(video_mode=True, lite=lite)
        self.buffer_s = buffer_s
        self.arm_s = arm_s
        self._last_ts = -1
        self.t0 = time.time()
        self.history: list[A.Report] = []
        self.redo_rect = None        # (x0, y0, x1, y1) of the on-screen REDO button, display coords
        self.reset()

    def reset(self):
        """Start a new attempt (the 'redo' action)."""
        self.buf = deque()           # (t_seconds, kps[33,2], raw frame)
        self.report: A.Report | None = None
        self.result_img = None
        self.state = "arming"
        self.state_t = time.time() - self.t0
        self.redo_rect = None

    @property
    def paused(self):
        return self.state == "result"

    def click(self, x: int, y: int):
        """Mouse click in display coords; returns True if it hit the REDO button."""
        r = self.redo_rect
        if self.state == "result" and r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            self.reset()
            return True
        return False

    def process(self, bgr: np.ndarray, mirror: bool = False) -> np.ndarray:
        """Analyse a raw (un-flipped) camera frame. If mirror=True the returned display image is
        flipped like a mirror, but pose detection still runs on the raw frame so that left/right
        arm labels stay anatomically correct."""
        if self.state == "result":
            return self.result_img.copy()   # paused: keep showing the grade until redo

        h, w = bgr.shape[:2]
        now = time.time() - self.t0
        ts = max(int(now * 1000), self._last_ts + 1)
        self._last_ts = ts
        xy, _ = A.detect(self.lm, bgr, ts)
        self.buf.append((now, xy, bgr.copy()))
        while self.buf and now - self.buf[0][0] > self.buffer_s:
            self.buf.popleft()

        im, xy_d = self._display(bgr, xy, mirror)
        A.draw_pose(im, xy_d)

        if self.state == "arming":
            left = self.arm_s - (now - self.state_t)
            if left <= 0:
                self.state, self.buf = "ready", deque(list(self.buf)[-3:])
            A.put_text(im, f"Get ready... {max(left, 0):.1f}s", (20, 40), 0.9, bg=(120, 70, 0))
            return im

        # ready: look for a finished swing (contact plus some follow-through already in the buffer)
        if len(self.buf) > 12:
            ts_arr = np.array([b[0] for b in self.buf])
            fps = (len(ts_arr) - 1) / max(ts_arr[-1] - ts_arr[0], 1e-3)
            kps = np.array([b[1] for b in self.buf])
            rep = A.analyze(kps, fps, self.cfg, stroke=self.stroke)
            if rep.ok and rep.contact_idx < len(kps) - int(self.cfg.follow_window_s * fps):
                self.report = rep
                self.history.append(rep)
                if self.session:
                    self.session.add(rep.score)
                self.state = "result"
                self.result_img = self._render_result(rep, kps, mirror)
                return self.result_img.copy()

        if np.isnan(xy[A.NOSE, 0]):
            A.put_text(im, "No player detected - step back so your whole body is visible", (20, 40), 0.7, bg=(0, 0, 150))
        elif not A.body_in_frame(xy):
            A.put_text(im, A.FRAME_HINT[:78], (20, 40), 0.65, bg=(0, 0, 150))
        else:
            A.put_text(im, f"READY - hit your {self.stroke}!", (20, 40), 0.9, bg=(0, 120, 0))
        return im

    # ------------------------------------------------------------ drawing helpers
    @staticmethod
    def _display(bgr, xy, mirror):
        if not mirror:
            return bgr.copy(), xy
        w = bgr.shape[1]
        xy_d = xy.copy()
        xy_d[:, 0] = (w - 1) - xy_d[:, 0]
        return cv2.flip(bgr, 1), xy_d

    def _render_result(self, r: A.Report, kps: np.ndarray, mirror: bool) -> np.ndarray:
        """Frozen contact frame + grade card + REDO button."""
        k = A._smooth(A._interp_nan(kps), A.win_frames(0.1, r.fps))
        im, xy_d = self._display(self.buf[r.contact_idx][2], k[r.contact_idx], mirror)
        A.draw_pose(im, xy_d, highlight_arm=r.arm)
        h, w = im.shape[:2]

        # layout: grade card on the left, contact frame cropped around the player on the right
        pw = min(w // 2, 560)
        cw = w - pw
        body_x = xy_d[[A.L_SH, A.R_SH, A.L_HIP, A.R_HIP, A.L_WR, A.R_WR, A.L_AN, A.R_AN], 0]
        px = float(np.nanmean(body_x)) if not np.isnan(body_x).all() else w / 2
        cx0 = int(np.clip(px - cw / 2, 0, w - cw))
        canvas = np.full_like(im, 25)
        canvas[:, pw:] = im[:, cx0:cx0 + cw]
        im, ox = canvas, 0
        F = cv2.FONT_HERSHEY_SIMPLEX
        col = GRADE_COL.get(r.grade, (80, 80, 80))
        cv2.putText(im, r.grade, (ox + 24, 110), F, 3.6, col, 10, cv2.LINE_AA)
        cv2.putText(im, f"{r.score:.0f}/100", (ox + 150, 70), F, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        best = max(x.score for x in self.history)
        cv2.putText(im, f"{self.stroke.title()}  |  attempt {len(self.history)}  |  best {best:.0f}", (ox + 150, 105), F, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        y = 150
        if self.session:
            cmp = self.session.compare()
            if cmp:
                cv2.putText(im, cmp[0], (ox + 24, 130), F, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        for c in r.checks:
            if c.points is None:
                continue
            ok = c.status == "good"
            cv2.putText(im, c.name, (ox + 24, y), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            bx, bw = ox + 250, pw - 250 - 50
            cv2.rectangle(im, (bx, y - 12), (bx + bw, y), (70, 70, 70), -1)
            cv2.rectangle(im, (bx, y - 12), (bx + int(bw * c.points / 100), y), (0, 170, 0) if ok else (0, 120, 230), -1)
            cv2.putText(im, f"{c.points:.0f}", (bx + bw + 8, y), F, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y += 26

        y += 10
        if r.top_fix:
            cv2.putText(im, "Top fix:", (ox + 24, y), F, 0.6, (0, 200, 255), 2, cv2.LINE_AA); y += 26
            for line in _wrap(r.top_fix, 52)[:4]:
                cv2.putText(im, line, (ox + 24, y), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA); y += 22
        else:
            cv2.putText(im, f"Clean {self.stroke} - nice form!", (ox + 24, y), F, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

        # REDO button
        bx0, by0, bx1, by1 = ox + 24, h - 70, ox + 224, h - 20
        cv2.rectangle(im, (bx0, by0), (bx1, by1), (0, 140, 255), -1)
        cv2.putText(im, "REDO  (R)", (bx0 + 30, by1 - 16), F, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, "Q to quit", (bx1 + 20, by1 - 16), F, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        self.redo_rect = (bx0, by0, bx1, by1)
        return im

    def close(self):
        self.lm.close()


class FootworkLive:
    """6-point footwork in the live window.
      calibrate - click the 4 corners of your area (saved to data/court.json, reused next time)
      drill     - calls a corner, scores each rep; a summary card appears after `reps` reps
      done      - frozen summary card until reset()
    """

    def __init__(self, size_m: tuple | None = None, reps: int = 12, calls: bool = True, lite: bool = False,
                 voice: bool = True, recalibrate: bool = False, history: History | None = None,
                 cfg: FW.FootworkConfig | None = None):
        self.lm = A.make_landmarker(video_mode=True, lite=lite)
        self.size_m, self.reps, self.calls, self.cfg = size_m, reps, calls, cfg or FW.FootworkConfig()
        self.voice = voice and sys.platform == "darwin" and shutil.which("say") is not None
        self.hist = history
        self.court = None if recalibrate else FW.Court.load()
        if self.court and size_m and (self.court.width_m, self.court.depth_m) != tuple(size_m):
            self.court = None        # different area size requested -> click the corners again
        self.clicks: list = []
        self.t0 = time.time()
        self._last_ts = -1
        self.last_frame = None
        self.button_rect = None
        self.reset()

    @property
    def paused(self):
        return self.state == "done"

    def reset(self):
        """New session (same court)."""
        self.state = "calibrate" if self.court is None else "drill"
        self.drill = FW.FootworkDrill(self.court, self.cfg, self.calls, self.reps) if self.court else None
        self.session = self.hist.start("footwork", dict(width_m=self.court.width_m, depth_m=self.court.depth_m,
                                                        calls=self.calls)) if self.hist and self.court else None
        self.last_rep = None
        self.flash = ("", 0.0)
        self.summary_img = None
        self.button_rect = None

    def recalibrate(self):
        self.court, self.clicks = None, []
        self.reset()

    def click(self, x: int, y: int):
        if self.state == "calibrate" and self.last_frame is not None and len(self.clicks) < 4:
            self.clicks.append((float(x), float(y)))
            if len(self.clicks) == 4:
                h, w = self.last_frame.shape[:2]
                W, D = self.size_m or (5.2, 6.7)
                self.court = FW.Court(self.clicks, W, D, (w, h))
                self.court.save()
                self.reset()
            return True
        r = self.button_rect
        if self.state == "done" and r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            self.reset()
            return True
        return False

    def _say(self, text):
        if self.voice:
            subprocess.Popen(["say", "-r", "250", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def process(self, bgr: np.ndarray, mirror: bool = False, t: float | None = None) -> np.ndarray:
        """mirror is ignored: the court overlay must line up with the real floor."""
        if self.state == "done":
            return self.summary_img.copy()
        h, w = bgr.shape[:2]
        self.last_frame = bgr
        if self.court and tuple(self.court.image_size) not in ((w, h), (0, 0)):
            self.recalibrate()       # camera resolution changed -> saved corners no longer line up
        now = (time.time() - self.t0) if t is None else t
        im = bgr.copy()
        if self.state == "calibrate":
            return self._draw_calibration(im)

        ts = max(int(now * 1000), self._last_ts + 1)
        self._last_ts = ts
        xy, _ = A.detect(self.lm, bgr, ts)
        for kind, val in self.drill.update(now, xy):
            if kind == "call":
                self._say(val.lower())
            elif kind == "rep":
                self.last_rep = val
                if self.session:
                    self.session.add(val.score, val.detail())
            elif kind == "miss":
                self.flash = (val, now + 2.0)
                self._say("miss")
            elif kind == "done":
                self._say("session complete")
                self.state = "done"
                self.summary_img = self._render_summary(im)
                return self.summary_img.copy()

        d, c = self.drill, self.court
        active = d.cur["target"] if d.cur and d.state in ("moving", "lunge") else None
        FW.draw_floor(im, c, active)
        A.draw_pose(im, xy)
        FW.draw_minimap(im, c, d, active, height=160)
        if active and d.calls:
            A.put_text(im, active, (20, 70), 1.8, (255, 255, 255), FW.ORANGE, 4)
        elif np.isnan(xy[A.NOSE, 0]) or not FW.feet_visible(xy):
            A.put_text(im, "Feet not visible - the camera must see your feet in the whole area", (20, 40), 0.65,
                       bg=(0, 0, 150))
        else:
            A.put_text(im, d.hint, (20, 40), 0.8, bg=(60, 60, 60))
        if self.flash[1] > now:
            A.put_text(im, self.flash[0][:70], (20, 110), 0.65, bg=(0, 0, 170))
        A.put_text(im, f"Rep {len(d.results) + (1 if d.cur else 0)}/{self.reps}" if self.reps else
                   f"Reps {len(d.results)}", (20, h - 50), 0.6, bg=(40, 40, 40))
        if self.last_rep:
            r = self.last_rep
            A.put_text(im, FW.rep_line(r), (20, h - 20), 0.55,
                       bg=GRADE_COL.get(r.grade, (60, 60, 60)))
        return im

    def _draw_calibration(self, im):
        h, w = im.shape[:2]
        for i, p in enumerate(self.clicks):
            cv2.circle(im, (int(p[0]), int(p[1])), 7, FW.ORANGE, -1, cv2.LINE_AA)
            cv2.putText(im, FW.CORNER_PROMPTS[i], (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        FW.ORANGE, 2, cv2.LINE_AA)
        if len(self.clicks) > 1:
            cv2.polylines(im, [np.int32(self.clicks)], False, FW.ORANGE, 2, cv2.LINE_AA)
        W, D = self.size_m or (5.2, 6.7)
        A.put_text(im, f"SETUP: click the {FW.CORNER_PROMPTS[len(self.clicks)]} corner of your area", (20, 40),
                   0.75, bg=(150, 80, 0))
        A.put_text(im, "(left/right as you face the net)  area %.1f x %.1f m - change with --court WxD" % (W, D),
                   (20, 75), 0.5, bg=(60, 60, 60), thick=1)
        if min(W, D) < FW.SMALL_AREA_M:
            A.put_text(im, "Small area: scores lean on reaction, split step and lunge form", (20, 105), 0.5,
                       bg=(0, 90, 160), thick=1)
        return im

    def _render_summary(self, last_img):
        sm = FW.summarize(self.drill.results)
        im = (last_img * 0.25).astype(np.uint8)
        h, w = im.shape[:2]
        F = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, sm["grade"], (24, 110), F, 3.6, GRADE_COL.get(sm["grade"], (80, 80, 80)), 10, cv2.LINE_AA)
        cv2.putText(im, f"{sm['avg']:.0f}/100 avg", (150, 65), F, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"6-point footwork | {sm['n']} reps | best {sm['best']:.0f} | worst {sm['worst']:.0f}",
                    (150, 100), F, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        y = 150
        lines = self.session.compare() if self.session else []
        for line in lines:
            cv2.putText(im, line, (24, y), F, 0.55, (0, 220, 255), 1, cv2.LINE_AA); y += 26
        y += 8
        fmt = lambda v: "-" if v is None else f"{v:.2f}s"
        for line in (f"Avg reaction {fmt(sm['reaction'])}   move {fmt(sm['move'])}   recovery {fmt(sm['recover'])}",
                     f"Consistency: scores vary +/- {sm['consistency']:.0f}",
                     f"Weakest corner: {sm['weakest_corner']} ({sm['per_target'][sm['weakest_corner']]:.0f})",
                     f"Work on: {sm['weakest_metric']} ({sm['metric_avg'].get(sm['weakest_metric'], 0):.0f}/100)"):
            cv2.putText(im, line, (24, y), F, 0.55, (255, 255, 255), 1, cv2.LINE_AA); y += 26
        for line in _wrap(sm["tip"], 60)[:3]:
            cv2.putText(im, line, (24, y), F, 0.5, (0, 200, 255), 1, cv2.LINE_AA); y += 22
        labels = {k: f"{v:.0f}" for k, v in sm["per_target"].items()}
        FW.draw_minimap(im, self.court, None, None, height=min(260, h - 140), labels=labels)
        bx0, by0, bx1, by1 = 24, h - 70, 284, h - 20
        cv2.rectangle(im, (bx0, by0), (bx1, by1), (0, 140, 255), -1)
        cv2.putText(im, "NEW SESSION (R)", (bx0 + 18, by1 - 16), F, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, "C recalibrate   Q quit", (bx1 + 20, by1 - 16), F, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        self.button_rect = (bx0, by0, bx1, by1)
        return im

    def close(self):
        self.lm.close()


def _wrap(text: str, width: int):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur); cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    return lines + ([cur] if cur else [])


class CameraReader:
    """Grabs webcam frames on a background thread so pose detection never waits on the camera;
    read() always returns the newest frame (older ones are dropped)."""

    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.ok = cap, None, 0, True
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, fr = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, fr
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq: int):
        """Block until a frame newer than last_seq arrives; returns (ok, frame, seq)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > last_seq or not self.ok, timeout=1.0)
            return self.ok and self.frame is not None, self.frame, self.seq

    def stop(self):
        self.ok = False


def _court_size(v: str):
    try:
        W, D = (float(x) for x in v.lower().replace(" ", "").split("x"))
        assert W > 0 and D > 0
        return W, D
    except Exception:
        raise argparse.ArgumentTypeError("use WIDTHxDEPTH in metres, e.g. 5.2x6.7 or 3x3")


def main():
    ap = argparse.ArgumentParser(description="Live badminton coach")
    ap.add_argument("source", nargs="?", default=0, help="video file to replay (default: webcam 0)")
    ap.add_argument("--drill", choices=DRILLS, default="smash")
    ap.add_argument("--lite", action="store_true", help="lighter pose model, higher frame rate")
    ap.add_argument("--court", type=_court_size, default=None,
                    help="footwork area WIDTHxDEPTH in metres (default 5.2x6.7 = singles half court)")
    ap.add_argument("--reps", type=int, default=12, help="footwork reps per session (0 = endless)")
    ap.add_argument("--recalibrate", action="store_true", help="click the court corners again")
    ap.add_argument("--quiet", action="store_true", help="no spoken call-outs")
    ap.add_argument("--no-save", action="store_true", help="don't record this session in data/history.json")
    a = ap.parse_args()

    src = a.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Cannot open video source")
    live_cam = isinstance(src, int)
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if live_cam:
        cap.set(cv2.CAP_PROP_FPS, 60)          # ask for the camera's fastest mode; ignored if unsupported
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reader = CameraReader(cap) if live_cam else None
    hist = None if a.no_save else History()
    first = {"recalibrate": a.recalibrate}

    def make(drill):
        """Build the analyser for a drill. Switching keeps the camera open; each drill keeps its own history."""
        if drill == "footwork":
            # a replayed file can't react to call-outs, so it runs in free mode (any corner, no reaction time)
            an = FootworkLive(a.court, a.reps or None, calls=live_cam, lite=a.lite, voice=not a.quiet,
                              recalibrate=first.pop("recalibrate", False), history=hist)
        else:
            an = LiveAnalyzer(lite=a.lite, history=hist, stroke=drill)
        print(f"Drill: {drill}")
        return an

    drill = a.drill
    la = make(drill)
    win = "Badminton coach (d = switch drill, r = redo / new session, c = recalibrate court, q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: la.click(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    cv2.resizeWindow(win, 960, 540)
    cv2.moveWindow(win, 40, 40)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    print("Camera started. A window titled '%s' should appear (check the Dock / Cmd+Tab if it is hidden). Press q in it to quit." % win)
    fr, seq, fps, t_prev, n = None, 0, 0.0, time.time(), 0
    while True:
        if not la.paused or fr is None:   # paused on a result card: stop consuming frames
            ok, fr, seq = reader.read(seq) if reader else (*cap.read(), 0)
            if not ok:
                break
            n += 1
            if fr.shape[1] > 960:
                fr = cv2.resize(fr, (960, int(fr.shape[0] * 960 / fr.shape[1])))
        if drill == "footwork":
            out = la.process(fr, t=None if live_cam else n / file_fps)
        else:
            out = la.process(fr, mirror=live_cam)
        if not la.paused:
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            A.put_text(out, f"{fps:.0f} fps", (out.shape[1] - 110, out.shape[0] - 20), 0.6, bg=(40, 40, 40))
        nxt = DRILLS[(DRILLS.index(drill) + 1) % len(DRILLS)]
        A.put_text(out, f"D: switch to {nxt}", (out.shape[1] - 200, out.shape[0] - 55), 0.5, bg=(40, 40, 40), thick=1)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            la.reset()
            t_prev = time.time()
        if key == ord("c") and drill == "footwork":
            la.recalibrate()
        if key == ord("d"):
            la.close()
            drill = DRILLS[(DRILLS.index(drill) + 1) % len(DRILLS)]
            la = make(drill)       # the mouse callback looks up `la` at click time, so it follows the switch
            t_prev = time.time()
    if reader:
        reader.stop()
    la.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
