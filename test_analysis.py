"""Sanity tests with synthetic skeletons (no video needed).  Run: python test_analysis.py"""
import numpy as np
import analysis as A

FPS = 30


def make_swing(good: bool, n=60):
    """Right-handed overhead swing, camera sees player's side. y is down (pixels)."""
    k = np.zeros((n, 33, 2))
    for t in range(n):
        base = np.zeros((33, 2))
        base[A.NOSE] = (300, 100)
        base[A.L_SH] = (280, 150); base[A.R_SH] = (320, 150)
        base[A.L_HIP] = (285, 270); base[A.R_HIP] = (315, 270)
        kn = 175 if not good else 150  # bend knees in good
        flex = np.radians(180 - kn)
        for kni, hip, an, x in ((A.L_KN, A.L_HIP, A.L_AN, 285), (A.R_KN, A.R_HIP, A.R_AN, 315)):
            base[kni] = (x + 40 * np.sin(flex), 270 + 60 * np.cos(flex) * 0 + 60)
            base[an] = (x, 270 + 60 + 60)
        base[A.L_EL] = (270, 200); base[A.L_WR] = (265, 245)
        # swing: t<30 prep (elbow flexed), 30..40 accelerate, contact at 36
        u = np.clip((t - 25) / 11, 0, 1)
        sh = base[A.R_SH]
        elb_y = sh[1] - (10 if good else -40) * 1.0     # elbow height at prep
        ext = 1.0 if good else 0.55                     # how far the arm extends
        L = 60.0
        elbow = sh + np.array([15, -(10 if good else -40)]) * np.array([1, 1])
        elbow = np.array([sh[0] + 20, elb_y])
        wrist_prep = elbow + np.array([-40, -5])        # hand behind head, flexed
        wrist_hit = np.array([sh[0] + 5, sh[1] - L * 2 * ext])  # hand above head
        elbow_hit = (sh + wrist_hit) / 2 + np.array([4 if good else 25, 0])
        s = u ** 2
        base[A.R_EL] = elbow * (1 - s) + elbow_hit * s
        base[A.R_WR] = wrist_prep * (1 - s) + wrist_hit * s
        k[t] = base
    return k


def run(good):
    return A.analyze(make_swing(good), FPS, A.Config(swing_speed_thresh=1.0))


def make_smash(fps, dur=2.0):
    """Wrist rises to a high contact (~0.96 torso above the shoulder) at t=1.0s, then whips down past the hip."""
    n = int(dur * fps); base = make_swing(True)[0]
    k = np.tile(base, (n, 1, 1)).astype(float)
    sh = base[A.R_SH]
    for i in range(n):
        t = i / fps
        if t < 0.7: y = sh[1] + 20
        elif t < 1.0: y = sh[1] + 20 - (t - 0.7) / 0.3 * 135
        elif t < 1.15: y = sh[1] - 115 + (t - 1.0) / 0.15 * 230
        else: y = sh[1] + 115
        k[i, A.R_WR] = (sh[0] + 10, y); k[i, A.R_EL] = (sh[0] + 8, (sh[1] + y) / 2)
    return k


def test_contact_height_at_low_fps():
    """A high contact followed by a big follow-through must score well on BOTH, even at live-camera frame rates."""
    for fps in (10, 15, 30):
        r = A.analyze(make_smash(fps), fps, A.Config(swing_speed_thresh=1.0))
        v = {c.name: c.value for c in r.checks}
        assert v["Contact height"] > 0.75, (fps, v["Contact height"])
        assert v["Downward follow-through"] > 0.6, (fps, v["Downward follow-through"])


def test_live_pause_and_redo():
    """Live mode: arming -> ready -> swing freezes on a result card -> reset() starts over."""
    import live as L
    clock = {"t": 0.0}
    L.time.time = lambda: clock["t"]
    A.make_landmarker = lambda video_mode=True, lite=False: type("LM", (), {"close": lambda self: None})()
    swing = make_swing(True, n=80)
    frame = {"i": 0}
    A.detect = lambda lm, bgr, ts=None: (swing[min(frame["i"], len(swing) - 1)], np.ones(33))
    la = L.LiveAnalyzer(A.Config(swing_speed_thresh=1.0), arm_s=0.5)
    img = np.zeros((480, 640, 3), np.uint8)

    def step(i):
        frame["i"] = i
        clock["t"] += 1 / FPS
        return la.process(img)

    for _ in range(20):          # standing still during arming
        step(0)
    assert la.state == "ready", la.state
    for i in range(80):
        step(i)
        if la.state == "result":
            break
    assert la.state == "result" and la.report is not None
    frozen = step(0)
    assert np.array_equal(frozen, step(10)), "display should stay frozen on the result card"
    x0, y0, x1, y1 = la.redo_rect
    assert la.click((x0 + x1) // 2, (y0 + y1) // 2) and la.state == "arming"
    assert len(la.history) == 1


def make_overhead(fps, slow=1.0, face=0.0, reach=10.0, dur=2.5):
    """Like make_smash, but the downswing takes `slow` times longer (slow > 1 = softer, drop-like), the
    head is turned `face` px towards +x (side-on camera) and contact is `reach` px in front of the shoulder."""
    n = int(dur * fps); base = make_swing(True)[0]
    k = np.tile(base, (n, 1, 1)).astype(float)
    sh = base[A.R_SH]
    k[:, A.NOSE, 0] = 300 + face
    down = 0.15 * slow
    for i in range(n):
        t = i / fps
        if t < 0.7: y = sh[1] + 20
        elif t < 1.0: y = sh[1] + 20 - (t - 0.7) / 0.3 * 135
        elif t < 1.0 + down: y = sh[1] - 115 + (t - 1.0) / down * 230 / slow
        else: y = sh[1] - 115 + 230 / slow
        k[i, A.R_WR] = (sh[0] + reach, y); k[i, A.R_EL] = (sh[0] + reach * 0.8, (sh[1] + y) / 2)
    return k


def test_strokes():
    cfg = A.Config(swing_speed_thresh=1.0)
    # sweet-spot scoring
    b = lambda v: A.make_band_check(1, "x", v, 4.5, 6.0, 8.5, 10.0, "", "ok", "low", "high", "{:.1f}")
    assert b(7).points == 100 and b(6.0).points == 100 and b(8.5).points == 100
    assert b(4.5).points == 60 and b(10.0).points == 60 and b(4.5).status == b(10).status == "good"
    assert b(12).status == "warn" and "high" in b(12).message and b(12).points < 60
    assert b(3).status == "warn" and "low" in b(3).message
    assert b(float("nan")).status == "na"
    # every stroke's weights sum to 1 and smash is unchanged by the refactor
    for s in A.STROKES:
        assert abs(sum(m.weight for m in A.stroke_metrics(s, cfg)) - 1) < 1e-9, s
    assert A.analyze(make_smash(30), 30, cfg).score == A.analyze(make_smash(30), 30, cfg, stroke="smash").score
    try:
        A.analyze(make_smash(30), 30, cfg, stroke="lob"); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # a full-power smash graded as a drop: too fast, big follow-through
    fast = A.analyze(make_overhead(30, face=25, reach=40), 30, cfg, stroke="drop")
    ck = {c.name: c for c in fast.checks}
    assert fast.ok and fast.stroke == "drop"
    assert ck["Racket-hand speed"].status == "warn" and "too fast" in ck["Racket-hand speed"].message
    assert ck["Controlled follow-through"].status == "warn"
    # a soft swing with a checked follow-through is a good drop
    soft = A.analyze(make_overhead(30, slow=1.4, face=25, reach=40), 30, cfg, stroke="drop")
    ck = {c.name: c for c in soft.checks}
    assert ck["Racket-hand speed"].points == 100, ck["Racket-hand speed"].message
    assert ck["Controlled follow-through"].status == "good"
    assert ck["Contact in front"].status == "good"
    assert soft.score > fast.score + 10, (soft.score, fast.score)
    # contact behind the head: wrong for a drop, fine-ish for a clear; facing unknown -> not scored
    behind = make_overhead(30, slow=1.4, face=25, reach=-5)
    d = {c.name: c for c in A.analyze(behind, 30, cfg, stroke="drop").checks}
    c = {c.name: c for c in A.analyze(behind, 30, cfg, stroke="clear").checks}
    assert d["Contact in front"].status == "warn" and c["Contact position"].status == "good"
    flat = {c.name: c for c in A.analyze(make_overhead(30), 30, cfg, stroke="clear").checks}
    assert flat["Contact position"].status == "na"
    # the same powerful swing scores better as a clear than as a drop
    clear = A.analyze(make_overhead(30, face=25, reach=0), 30, cfg, stroke="clear")
    assert clear.score > fast.score, (clear.score, fast.score)


def test_phases_and_shadow():
    """The swing is broken into ordered snapshots; a shadow (target pose) is built where the form misses."""
    g, b = run(True), run(False)
    for r in (g, b):
        assert [p.name for p in r.phases] and len(r.phases) == 4
        assert [p.idx for p in r.phases] == sorted(p.idx for p in r.phases)
        assert {p.name for p in r.phases} == {"Load", "Trophy", "Contact", "Finish"}
    ph = {p.name: p for p in b.phases}
    sh, el, wr = A._arm_idx(b.arm)
    c = ph["Contact"]
    assert c.status == "warn"
    assert abs(A.angle(c.ghost[sh], c.ghost[el], c.ghost[wr]) - 170) < 1.5     # straightened to the benchmark
    for i, j in ((sh, el), (el, wr)):                                           # limb lengths kept
        assert abs(np.linalg.norm(c.ghost[i] - c.ghost[j]) - np.linalg.norm(c.pose[i] - c.pose[j])) < 1e-6
    ld = ph["Load"]
    knee = min(A.angle(ld.ghost[h], ld.ghost[k], ld.ghost[a]) for h, k, a in A.LEGS)
    assert abs(knee - 140) < 1.5, knee
    tr = ph["Trophy"]
    assert (tr.ghost[sh, 1] - tr.ghost[el, 1]) > (tr.pose[sh, 1] - tr.pose[el, 1])  # elbow lifted
    # a phase that is already at the benchmark keeps the player's own pose
    for p in g.phases:
        if all("ok" in n for n in p.notes):
            assert np.allclose(p.ghost, p.pose, equal_nan=True)
    frames = [np.zeros((480, 640, 3), np.uint8)] * len(make_swing(False))
    tiles = A.phase_strip(frames, b, (240, 270), mirror=True)
    assert len(tiles) == 4 and tiles[0].shape == (270, 240, 3)


if __name__ == "__main__":
    g, b = run(True), run(False)
    for name, r in (("GOOD", g), ("BAD", b)):
        print(f"== {name}: ok={r.ok} arm={r.arm} score={r.score} contact={r.contact_idx} prep={r.prep_idx}")
        for c in r.checks:
            print(f"   [{c.status:4}] {c.name}: {c.value if c.value is None else round(c.value, 2)}  -> {c.message}")
    assert g.ok and b.ok
    assert g.arm == "right" and b.arm == "right"
    assert g.score > b.score, (g.score, b.score)
    assert 30 <= g.contact_idx <= 42
    # angle sanity
    assert abs(A.angle((0, 1), (0, 0), (1, 0)) - 90) < 1e-6
    assert abs(A.angle((-1, 0), (0, 0), (1, 0)) - 180) < 1e-6
    # no-swing / no-person handling
    still = np.tile(make_swing(True)[0], (40, 1, 1))
    assert not A.analyze(still, FPS).ok
    assert not A.analyze(np.full((40, 33, 2), np.nan), FPS).ok
    # partially visible body (legs cut off) must be refused, not scored
    cut = make_swing(True).copy(); cut[:, [A.L_KN, A.R_KN, A.L_AN, A.R_AN]] = np.nan
    r = A.analyze(cut, FPS, A.Config(swing_speed_thresh=1.0))
    assert not r.ok and "frame" in r.reason.lower(), r.reason
    # grading
    assert g.grade in "ABCDF" and A.letter(95) == "A" and A.letter(10) == "F"
    assert A.points(170, 150, 170) == 100 and A.points(150, 150, 170) == 60
    assert A.points(140, 165, 140) == 100   # lower-is-better metric (knee angle)
    test_contact_height_at_low_fps()
    test_live_pause_and_redo()
    test_strokes()
    test_phases_and_shadow()
    print("ALL TESTS PASSED")

