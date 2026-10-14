"""Real-time analysis: rolling buffer + automatic swing detection.

`LiveAnalyzer.process(bgr)` draws the skeleton, shows live elbow angle, and every time a swing
is detected it runs the same `analyze()` as the offline path and shows the feedback on screen.

Run standalone:  python live.py            (webcam 0)
                 python live.py clip.mp4   (replay a file in real time)
Press q to quit.
"""
from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import numpy as np

import analysis as A


class LiveAnalyzer:
    def __init__(self, cfg: A.Config | None = None, buffer_s: float = 2.5, cooldown_s: float = 2.0):
        self.cfg = cfg or A.Config()
        self.lm = A.make_landmarker(video_mode=True)
        self.buf = deque()           # (t_seconds, kps[33,2])
        self.buffer_s = buffer_s
        self.cooldown_s = cooldown_s
        self.last_report_t = -1e9
        self.report: A.Report | None = None
        self.report_t = -1e9
        self._last_ts = -1
        self.t0 = time.time()
        self.history: list[A.Report] = []

    def process(self, bgr: np.ndarray, mirror: bool = False) -> np.ndarray:
        """Analyse a raw (un-flipped) camera frame. If mirror=True the returned display image is
        flipped like a mirror, but pose detection still runs on the raw frame so that left/right
        arm labels stay anatomically correct."""
        h, w = bgr.shape[:2]
        now = time.time() - self.t0
        ts = max(int(now * 1000), self._last_ts + 1)
        self._last_ts = ts
        xy, _ = A.detect(self.lm, bgr, ts)
        self.buf.append((now, xy))
        while self.buf and now - self.buf[0][0] > self.buffer_s:
            self.buf.popleft()

        if mirror:
            im = cv2.flip(bgr, 1)
            xy_d = xy.copy()
            xy_d[:, 0] = (w - 1) - xy_d[:, 0]
        else:
            im, xy_d = bgr, xy
        A.draw_pose(im, xy_d, highlight_arm=self.report.arm if self.report and self.report.ok else None)

        if len(self.buf) > 12 and now - self.last_report_t > self.cooldown_s:
            ts_arr = np.array([b[0] for b in self.buf])
            fps = (len(ts_arr) - 1) / max(ts_arr[-1] - ts_arr[0], 1e-3)
            kps = np.array([b[1] for b in self.buf])
            rep = A.analyze(kps, fps, self.cfg)
            # only report once the swing has finished (contact is not at the very end of the buffer)
            if rep.ok and rep.contact_idx < len(kps) - int(0.25 * fps):
                self.report, self.report_t, self.last_report_t = rep, now, now
                self.history.append(rep)

        # HUD
        if np.isnan(xy[A.NOSE, 0]):
            A.put_text(im, "No player detected - step back so your whole body is visible", (20, 40), 0.7, bg=(0, 0, 150))
        r = self.report
        if r and now - self.report_t < 6.0:
            col = (0, 130, 0) if r.score >= 75 else (0, 110, 200) if r.score >= 50 else (0, 0, 170)
            A.put_text(im, f"Last stroke: {r.score:.0f}/100 ({r.arm} arm)", (20, 40), 0.9, bg=col)
            y = 80
            for c in r.checks:
                if c.status == "warn":
                    A.put_text(im, "! " + c.message[:70], (20, y), 0.55, bg=(0, 0, 0)); y += 30
            if y == 80:
                A.put_text(im, "Clean stroke - nice form!", (20, y), 0.7, bg=(0, 100, 0))
        return im

    def close(self):
        self.lm.close()


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Cannot open video source")
    la = LiveAnalyzer()
    win = "Badminton form coach (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    cv2.moveWindow(win, 40, 40)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    print("Camera started. A window titled '%s' should appear (check the Dock / Cmd+Tab if it is hidden). Press q in it to quit." % win)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] > 960:
            fr = cv2.resize(fr, (960, int(fr.shape[0] * 960 / fr.shape[1])))
        cv2.imshow(win, la.process(fr, mirror=isinstance(src, int)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    la.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
