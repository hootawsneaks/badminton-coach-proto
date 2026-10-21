"""Real-time analysis: rolling buffer + automatic swing detection.

`LiveAnalyzer.process(bgr)` draws the skeleton and, once a swing is detected, runs the same
`analyze()` as the offline path and grades the smash against the benchmark in `analysis.Config`.

Flow: get ready -> swing -> the screen freezes on your contact frame with a grade card ->
click REDO (or press r) to try again.

Run standalone:  python live.py            (webcam 0)
                 python live.py clip.mp4   (replay a file in real time)
Keys: r = redo, q = quit.
"""
from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import numpy as np

import analysis as A


GRADE_COL = {"A": (0, 150, 0), "B": (60, 150, 0), "C": (0, 140, 200), "D": (0, 90, 220), "F": (0, 0, 180)}


class LiveAnalyzer:
    """Three states:
      arming - just (re)started; ignore motion for `arm_s` so walking into position isn't scored
      ready  - watching for a smash
      result - swing captured: the display freezes on the contact frame with the grade card until reset()
    """

    def __init__(self, cfg: A.Config | None = None, buffer_s: float = 2.5, arm_s: float = 1.5):
        self.cfg = cfg or A.Config()
        self.lm = A.make_landmarker(video_mode=True)
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
            rep = A.analyze(kps, fps, self.cfg)
            if rep.ok and rep.contact_idx < len(kps) - int(self.cfg.follow_window_s * fps):
                self.report = rep
                self.history.append(rep)
                self.state = "result"
                self.result_img = self._render_result(rep, kps, mirror)
                return self.result_img.copy()

        if np.isnan(xy[A.NOSE, 0]):
            A.put_text(im, "No player detected - step back so your whole body is visible", (20, 40), 0.7, bg=(0, 0, 150))
        elif not A.body_in_frame(xy):
            A.put_text(im, A.FRAME_HINT[:78], (20, 40), 0.65, bg=(0, 0, 150))
        else:
            A.put_text(im, "READY - hit your smash!", (20, 40), 0.9, bg=(0, 120, 0))
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
        k = A._smooth(A._interp_nan(kps), 5)
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
        cv2.putText(im, f"Smash  |  attempt {len(self.history)}  |  best {best:.0f}", (ox + 150, 105), F, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        y = 150
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
            cv2.putText(im, "Clean smash - nice form!", (ox + 24, y), F, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

        # REDO button
        bx0, by0, bx1, by1 = ox + 24, h - 70, ox + 224, h - 20
        cv2.rectangle(im, (bx0, by0), (bx1, by1), (0, 140, 255), -1)
        cv2.putText(im, "REDO  (R)", (bx0 + 30, by1 - 16), F, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, "Q to quit", (bx1 + 20, by1 - 16), F, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        self.redo_rect = (bx0, by0, bx1, by1)
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


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Cannot open video source")
    la = LiveAnalyzer()
    win = "Badminton smash coach (r = redo, q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: la.click(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    cv2.resizeWindow(win, 960, 540)
    cv2.moveWindow(win, 40, 40)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    print("Camera started. A window titled '%s' should appear (check the Dock / Cmd+Tab if it is hidden). Press q in it to quit." % win)
    fr = None
    while True:
        if la.state != "result" or fr is None:   # paused on the grade card: stop consuming frames
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] > 960:
                fr = cv2.resize(fr, (960, int(fr.shape[0] * 960 / fr.shape[1])))
        cv2.imshow(win, la.process(fr, mirror=isinstance(src, int)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            la.reset()
    la.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
