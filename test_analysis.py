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
    print("ALL TESTS PASSED")
