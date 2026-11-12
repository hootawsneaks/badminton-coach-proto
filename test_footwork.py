"""Footwork drill + session history tests on synthetic players (no camera / model needed).
Run: python test_footwork.py"""
import os
import tempfile

import numpy as np

import analysis as A
import footwork as FW
from history import History

FPS = 30
PX = 100   # synthetic camera: 1 m on the floor = 100 px, straight top-down-ish mapping


def make_court(W=5.2, D=6.7):
    return FW.Court([[100, 100], [100 + W * PX, 100], [100 + W * PX, 100 + D * PX], [100, 100 + D * PX]], W, D)


def pose(court, mid, bend=0.0, bounce=0.0, crossed=False, turn=0.0):
    """Synthetic keypoints for a player whose feet are centred on `mid` (court metres).
    bend: 0 = standing, 1 = deep lunge (knee ~95 deg, hips lower). turn: 0 = square, 1 = side-on."""
    xy = np.full((33, 2), np.nan)
    off = -0.15 if not crossed else 0.15
    lf, rf = court.to_image(np.array([mid + [off, 0], mid - [off, 0]]))
    for (an, he, to), f in (((A.L_AN, A.L_HEEL, A.L_TOE), lf), ((A.R_AN, A.R_HEEL, A.R_TOE), rf)):
        xy[an] = f; xy[he] = f + [0, 3]; xy[to] = f + [0, -6]
    foot_y = (lf[1] + rf[1]) / 2
    cx = (lf[0] + rf[0]) / 2
    leg = 160 - 35 * bend
    hip_y = foot_y - leg - bounce
    shw = 50 * (1 - 0.6 * turn)
    xy[A.L_HIP] = (cx - 15, hip_y); xy[A.R_HIP] = (cx + 15, hip_y)
    xy[A.L_SH] = (cx - shw / 2, hip_y - 120); xy[A.R_SH] = (cx + shw / 2, hip_y - 120)
    xy[A.NOSE] = (cx, hip_y - 150)
    for kn, hp, an in ((A.L_KN, A.L_HIP, A.L_AN), (A.R_KN, A.R_HIP, A.R_AN)):
        m = (xy[hp] + xy[an]) / 2
        xy[kn] = m + [10 + 70 * bend, 0]       # knee pushed forward -> bent
    for el, sh, wr in ((A.L_EL, A.L_SH, A.L_WR), (A.R_EL, A.R_SH, A.R_WR)):
        xy[el] = xy[sh] + [0, 50]; xy[wr] = xy[el] + [0, 50]
    return xy


class Player:
    """Scripted athlete: reacts to calls (or a fixed plan in free mode) and runs base -> corner -> base."""

    def __init__(self, court, react=0.3, move=0.7, hold=0.4, back=0.8, wrong=False, split=True):
        self.c, self.react, self.move, self.hold, self.back = court, react, move, hold, back
        self.wrong, self.split = wrong, split
        self.plan = None   # (t_start, target_pos)

    def go(self, t, target):
        if self.wrong:   # deliberately run to the opposite side
            target = target.replace("LEFT", "X").replace("RIGHT", "LEFT").replace("X", "RIGHT")
        self.plan = (t + self.react, self.c.targets[target], target.startswith("SIDE"))

    def at(self, t):
        b = self.c.base
        if not self.plan:
            return pose(self.c, b)
        t0, tgt, side = self.plan
        dt = t - t0
        # stop 0.1 m short of the target centre: arrival is judged on the closest foot
        tgt = b + (tgt - b) * (1 - 0.1 / max(np.linalg.norm(tgt - b), 1e-6))
        if dt < 0:
            bounce = 12 * np.sin(np.pi * (dt + self.react) / self.react) if self.split else 0
            return pose(self.c, b, bounce=max(bounce, 0))
        if dt < self.move:
            u = dt / self.move
            return pose(self.c, b + (tgt - b) * u, bend=0.4 * u)
        if dt < self.move + self.hold:
            return pose(self.c, tgt, bend=1.0)
        if dt < self.move + self.hold + self.back:
            u = (dt - self.move - self.hold) / self.back
            return pose(self.c, tgt + (b - tgt) * u, bend=0.4 * (1 - u))
        self.plan = None
        return pose(self.c, b)


def run(drill, player, seconds, free_plan=None):
    events = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if free_plan and free_plan and t >= free_plan[0][0]:
            player.go(t - player.react, free_plan.pop(0)[1])
        for ev in drill.update(t, player.at(t)):
            events.append((t, ev))
            if ev[0] == "call":
                player.go(t, ev[1])
    return events


def test_court_mapping():
    c = make_court()
    p = np.array([[1.0, 2.0], [5.2, 6.7]])
    assert np.allclose(c.to_court(c.to_image(p)), p)
    assert np.allclose(c.to_court(np.array([[100 + 260, 100 + 335]])), [[2.6, 3.35]])
    nan = c.to_court(np.array([[np.nan, np.nan]]))
    assert np.isnan(nan).all()
    for W, D in ((5.2, 6.7), (2.5, 2.5), (3.0, 2.0)):
        cc = make_court(W, D)
        ts = list(cc.targets.values()) + [cc.base]
        gaps = [np.linalg.norm(a - b) for i, a in enumerate(ts) for b in ts[i + 1:]]
        assert min(gaps) > cc.zone_r, (W, D, min(gaps))   # zones never overlap
        assert all(0 < t[0] < W and 0 < t[1] < D for t in ts)


def test_called_session():
    c = make_court()
    d = FW.FootworkDrill(c, reps=6, seed=1)
    ev = run(d, Player(c), 40)
    kinds = [e[1][0] for e in ev]
    calls = [e[1][1] for e in ev if e[1][0] == "call"]
    assert kinds.count("rep") == 6 and kinds[-1] == "done", kinds
    assert not d.misses, d.misses
    assert [r.target for r in d.results] == calls
    for r in d.results:
        assert 0.25 <= r.times["reaction"] <= 0.5, r.times
        assert 0 <= r.score <= 100 and r.grade in "ABCDF"
        names = {ch.name for ch in r.checks}
        assert {"Reaction time", "Move time to corner", "Recovery to base", "Lunge depth"} <= names
        lunge = next(ch for ch in r.checks if ch.name == "Lunge depth")
        assert lunge.value < 130, lunge.value
        if r.target.startswith("SIDE"):
            assert r.crossovers == 0 and "Square to the net" in names
    sm = FW.summarize(d.results)
    assert sm["n"] == 6 and sm["weakest_corner"] in calls and 0 <= sm["avg"] <= 100


def test_slow_player_scores_lower():
    c = make_court()
    fast = FW.FootworkDrill(c, reps=4, seed=3)
    run(fast, Player(c, react=0.3, move=0.7, back=0.8), 30)
    slow = FW.FootworkDrill(c, reps=4, seed=3)
    run(slow, Player(c, react=0.7, move=1.8, back=2.0, split=False), 45)
    f, s = FW.summarize(fast.results), FW.summarize(slow.results)
    assert f["n"] == s["n"] == 4
    assert f["avg"] > s["avg"] + 10, (f["avg"], s["avg"])


def test_small_area_benchmarks_scale():
    """Same movement speed in a 2.5 m room should score about as well as on a full court."""
    big, small = make_court(), make_court(2.5, 2.5)
    db = FW.FootworkDrill(big, reps=4, seed=5); run(db, Player(big, move=0.75, back=0.85), 30)
    speed = np.linalg.norm(big.targets["BACK LEFT"] - big.base) / 0.75
    dist_small = np.linalg.norm(small.targets["BACK LEFT"] - small.base)
    ds = FW.FootworkDrill(small, reps=4, seed=5)
    run(ds, Player(small, move=dist_small / speed, back=dist_small / speed + 0.1), 30)
    mb = np.mean([ch.points for r in db.results for ch in r.checks if ch.name == "Move time to corner"])
    ms = np.mean([ch.points for r in ds.results for ch in r.checks if ch.name == "Move time to corner"])
    assert len(ds.results) == 4 and abs(mb - ms) < 25, (mb, ms)


def test_wrong_corner_and_anticipation():
    c = make_court()
    d = FW.FootworkDrill(c, reps=3, seed=2)
    run(d, Player(c, wrong=True), 15)
    assert d.misses and "Wrong corner" in d.misses[0] and not d.results
    # leaving the base before the call is not scored
    d2 = FW.FootworkDrill(c, reps=3, seed=2)
    early = Player(c)
    for i in range(int(3 * FPS)):
        t = i / FPS
        if abs(t - 0.75) < 1e-9 + 0.5 / FPS and early.plan is None:
            early.plan = (t, c.targets["FRONT LEFT"], False)   # jumps the gun
        d2.update(t, early.at(t))
    assert not d2.results


def test_free_mode():
    c = make_court()
    d = FW.FootworkDrill(c, calls=False, reps=None)
    p = Player(c)
    ev = run(d, p, 9, free_plan=[(1.5, "BACK RIGHT"), (5.0, "SIDE LEFT")])
    reps = [e[1][1] for e in ev if e[1][0] == "rep"]
    assert [r.target for r in reps] == ["BACK RIGHT", "SIDE LEFT"], [r.target for r in reps]
    assert all(r.times["reaction"] is None for r in reps)
    assert not any(ch.name == "Reaction time" for ch in reps[0].checks)


def test_crossover_counted():
    c = make_court()
    d = FW.FootworkDrill(c, calls=False, reps=None)
    b, tgt = c.base, c.targets["SIDE RIGHT"]
    for i in range(int(5 * FPS)):
        t = i / FPS
        u = np.clip((t - 1.0) / 0.8, 0, 1)
        crossed = 0.3 < u < 0.6
        mid = b + (tgt - b) * u * 0.95
        if t > 2.5:
            mid = b
        d.update(t, pose(c, mid, bend=1.0 if 1.8 < t < 2.5 else 0, crossed=crossed))
    assert d.results and d.results[0].crossovers == 1, [r.crossovers for r in d.results]


def test_history():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "h.json")
        h = History(path)
        s1 = h.start("footwork")
        for x in (60, 70, 80):
            s1.add(x)
        assert not os.path.exists(path + ".tmp") and os.path.exists(path)
        h2 = History(path)
        assert h2.best_session("footwork")["avg"] == 70
        s2 = h2.start("footwork")
        assert h2.last_session("footwork")["id"] == s1.id
        for x in (85, 90, 75):
            s2.add(x)
        lines = s2.compare()
        assert "best session 70 (+13)" in lines[0], lines
        assert any("NEW best rep: 90" in l for l in lines), lines
        empty = h2.start("smash")
        assert empty.compare() == [] and len(History(path).sessions) == 2   # empty sessions aren't saved
        assert h2.best_session("smash") is None
        with open(path, "w") as f:
            f.write("{broken")
        assert History(path).sessions == [] and os.path.exists(path + ".corrupt")


def test_live_calibrate_and_session():
    import live as L
    with tempfile.TemporaryDirectory() as td:
        FW.COURT_PATH = os.path.join(td, "court.json")
        A.make_landmarker = lambda video_mode=True, lite=False: type("LM", (), {"close": lambda self: None})()
        c = make_court()
        player = Player(c)
        clock = {"t": 0.0}
        A.detect = lambda lm, bgr, ts=None: (player.at(clock["t"]), np.ones(33))
        fl = L.FootworkLive((5.2, 6.7), reps=3, voice=False, history=History(os.path.join(td, "h.json")))
        fl.drill = None
        img = np.zeros((800, 700, 3), np.uint8)
        fl.process(img, t=0)
        assert fl.state == "calibrate"
        for p in c.corners:
            fl.click(*p)
        assert fl.state == "drill" and os.path.exists(FW.COURT_PATH)
        assert np.allclose(FW.Court.load().corners, c.corners)
        for i in range(int(25 * FPS)):
            clock["t"] = i / FPS
            prev = len(fl.drill.results)
            fl.process(img, t=clock["t"])
            if fl.drill.cur and fl.drill.cur.get("t_call") == clock["t"]:
                player.go(clock["t"], fl.drill.cur["target"])
            if fl.state == "done":
                break
        assert fl.state == "done" and fl.paused, (fl.state, len(fl.drill.results), fl.drill.misses)
        assert fl.session.n == 3
        x0, y0, x1, y1 = fl.button_rect
        assert fl.click((x0 + x1) // 2, (y0 + y1) // 2) and fl.state == "drill" and fl.session.n == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("ALL FOOTWORK TESTS PASSED")
