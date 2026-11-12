"""Streamlit demo: badminton coach - smash / clear / drop form + 6-point footwork.   Run:  streamlit run app.py"""
import hashlib
import os
import tempfile

import numpy as np
import streamlit as st

import analysis as A
import footwork as FW
from history import History

st.set_page_config(page_title="Badminton Coach", layout="wide")
st.title("Badminton Coach")
st.caption("Computer-vision training: pick a drill, then analyse a clip, train live, and track your progress.")

# ------------------------------------------------------------ sidebar: drill + thresholds
d = A.Config()
with st.sidebar:
    drill = st.radio("What are you training?", ["Smash", "Clear", "Drop", "6-point footwork"])
    footwork = drill == "6-point footwork"
    stroke = None if footwork else drill.lower()
    st.divider()
if footwork:
    with st.sidebar:
        st.header("Training area")
        st.caption("Full singles half court is about 5.2 x 6.7 m. Smaller areas work too - targets and "
                   "time benchmarks scale with the size.")
        area_w = st.number_input("Width, side to side (m)", 1.0, 7.0, 5.2, 0.1)
        area_d = st.number_input("Depth, net to back (m)", 1.0, 8.0, 6.7, 0.1)
        if min(area_w, area_d) < FW.SMALL_AREA_M:
            st.info("Small area: scores lean on reaction, split step and lunge form more than speed.")
    cfg, arm = A.Config(), None
elif stroke == "smash":
    with st.sidebar:
        st.header("Smash pass lines")
        st.caption("Below the pass line = FIX. Full marks at the benchmark (see analysis.Config). "
                   "Demo placeholder numbers, not validated norms.")
        cfg = A.Config(
            min_smash_speed=st.slider("Min racket-hand speed (torso/s)", 2.0, 20.0, d.min_smash_speed, 0.5),
            min_elbow_extension_deg=st.slider("Min elbow extension at contact (deg)", 120, 180, int(d.min_elbow_extension_deg)),
            min_contact_height=st.slider("Min contact height (torso above shoulder)", 0.0, 1.5, d.min_contact_height, 0.05),
            min_trophy_elbow_height=st.slider("Min prep elbow height (rel. shoulder)", -0.6, 0.4, d.min_trophy_elbow_height, 0.05),
            max_knee_angle_prep=st.slider("Max knee angle in prep (deg)", 130, 180, int(d.max_knee_angle_prep)),
            swing_speed_thresh=st.slider("Swing detection sensitivity (lower = more sensitive)", 1.0, 10.0, d.swing_speed_thresh, 0.5),
        )
else:
    with st.sidebar:
        st.header(f"{drill} benchmarks")
        st.caption("Placeholder targets, not validated norms (see `analysis.stroke_metrics`). The coach grades "
                   f"whether your form matches a {stroke} - it can't see where the shuttle went.")
        cfg = A.Config(swing_speed_thresh=st.slider("Swing detection sensitivity (lower = more sensitive)", 1.0,
                                                    10.0, d.swing_speed_thresh, 0.5))
        st.dataframe([{"metric": m.name,
                       "target": (f"{m.fmt.format(m.bench)} to {m.fmt.format(m.band[0])}" if m.band
                                  else m.fmt.format(m.bench)) + m.unit,
                       "weight": f"{m.weight:.0%}"} for m in A.stroke_metrics(stroke, A.Config())],
                     hide_index=True)
if not footwork:
    with st.sidebar:
        arm_opt = st.radio("Racket arm", ["Auto-detect", "Right", "Left"])
        arm = None if arm_opt == "Auto-detect" else arm_opt.lower()

tab_video, tab_live, tab_prog = st.tabs(["Analyze a video", "Live webcam", "Progress"])

ICON = {"good": "OK", "warn": "FIX", "info": "INFO", "na": "-"}


def show_check(ch):
    fn = {"good": st.success, "warn": st.error, "info": st.info}.get(ch.status, st.write)
    pts = f" - {ch.points:.0f}/100" if ch.points is not None else ""
    target = f" (pass {ch.target})" if ch.target else ""
    fn(f"**{ICON[ch.status]} - {ch.name}**{pts}{target}: {ch.message}")


def show_report(rep: A.Report):
    if not rep.ok:
        st.warning(rep.reason)
        return
    c1, c2 = st.columns(2)
    c1.metric(f"{rep.stroke.title()} grade", rep.grade)
    c2.metric("Score", f"{rep.score:.0f} / 100")
    st.caption(f"Racket arm: {rep.arm}")
    if rep.top_fix:
        st.warning(f"**Top fix** - {rep.top_fix}")
    for ch in rep.checks:
        show_check(ch)


def video_bytes(writer):
    """Run writer(out_path) in a temp dir, re-encode for the browser, return the mp4 bytes."""
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "ann.mp4")
        writer(raw)
        h264 = os.path.join(td, "ann_h264.mp4")
        final = h264 if A.to_h264(raw, h264) else raw
        with open(final, "rb") as f:
            return f.read()


def load_clip(up):
    """Pose-extract an uploaded clip once; reruns (slider changes, buttons) reuse the result."""
    key = hashlib.md5(up.getbuffer()).hexdigest()
    if st.session_state.get("clip_key") != key:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in" + os.path.splitext(up.name)[1])
            with open(src, "wb") as f:
                f.write(up.getbuffer())
            bar = st.progress(0.0, text="Running pose estimation...")
            try:
                data = A.extract_video(src, progress_cb=lambda p: bar.progress(p, text="Running pose estimation..."))
            except Exception as e:
                st.error(f"Could not process video: {e}")
                st.stop()
            bar.empty()
        st.session_state.clip_key, st.session_state.clip = key, data
        st.session_state.pop("saved_clip", None)
    return key, st.session_state.clip


def stroke_video(data):
    rep = A.analyze(data["kps"], data["fps"], cfg, arm, stroke=stroke)
    vid = video_bytes(lambda out: A.annotate_video(data["frames"], data["kps"], rep, data["fps"], out))
    left, right = st.columns([3, 2])
    with left:
        st.video(vid)
        st.download_button("Download annotated video", vid, "annotated.mp4", "video/mp4")
    with right:
        show_report(rep)
    if rep.ok:
        import cv2
        import pandas as pd
        t = np.arange(len(rep.speed)) / rep.fps
        st.subheader("Swing signals")
        st.line_chart(pd.DataFrame({"time (s)": t, "wrist speed (torso/s)": rep.speed,
                                    "elbow angle (deg)": rep.elbow}).set_index("time (s)"))
        c1, c2 = st.columns(2)
        k = A._smooth(A._interp_nan(data["kps"]), 5)
        for col, idx, lab in ((c1, rep.prep_idx, "Preparation"), (c2, rep.contact_idx, "Contact")):
            im = data["frames"][idx].copy()
            A.draw_pose(im, k[idx], highlight_arm=rep.arm)
            col.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=f"{lab} (frame {idx})")


def footwork_video(key, data):
    import cv2
    import pandas as pd
    first = data["frames"][0]
    h, w = first.shape[:2]
    saved = FW.Court.load()
    guess = saved.corners if saved and tuple(saved.image_size) == (w, h) else FW.default_corners(w, h)
    st.markdown("**1. Mark your area.** Enter the pixel position of each corner (left/right as you face the net), "
                "until the white outline sits on your area's edges.")
    cols = st.columns(4)
    corners = []
    for i, (col, name) in enumerate(zip(cols, FW.CORNER_PROMPTS)):
        with col:
            x = st.number_input(f"{name} x", 0, w, int(guess[i][0]), key=f"cx{i}")
            y = st.number_input(f"{name} y", 0, h, int(guess[i][1]), key=f"cy{i}")
            corners.append([x, y])
    try:
        court = FW.Court(corners, area_w, area_d, (w, h))
    except Exception:
        st.error("Those corners don't form a usable area - check the order (NET-LEFT, NET-RIGHT, BACK-RIGHT, BACK-LEFT).")
        return
    prev = first.copy()
    FW.draw_floor(prev, court)
    for (x, y), name in zip(corners, FW.CORNER_PROMPTS):
        A.put_text(prev, name, (int(x) + 6, int(y) - 6), 0.5, bg=(0, 100, 200), thick=1)
    st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), caption=f"Frame size {w} x {h} px. Orange/grey rings = target "
             "zones, blue = base.", width=min(w, 900))
    st.caption("Tip: `python live.py --drill footwork` lets you click the corners instead; they're saved and reused "
               "here when the camera framing is the same.")

    st.markdown("**2. Analyse.** Recorded clips run in free mode: each base -> corner -> base trip is one rep "
                "(no call-outs, so no reaction time).")
    if not st.button("Analyse footwork", type="primary") and st.session_state.get("fw_key") != (key, str(corners), area_w, area_d):
        return
    st.session_state.fw_key = (key, str(corners), area_w, area_d)
    res = {}

    def writer(out):
        res["reps"], res["misses"] = FW.analyze_video(data["frames"], data["kps"], data["fps"], court, out_path=out)

    vid = video_bytes(writer)
    reps, misses = res["reps"], res["misses"]
    left, right = st.columns([3, 2])
    with left:
        st.video(vid)
        st.download_button("Download annotated video", vid, "footwork.mp4", "video/mp4")
    with right:
        if not reps:
            st.warning("No complete reps found. Check the corners line up with the floor, the feet are visible "
                       "the whole time, and each rep starts and ends standing in the base (blue circle).")
            for m in misses:
                st.caption(m)
            return
        sm = FW.summarize(reps)
        c1, c2, c3 = st.columns(3)
        c1.metric("Session grade", sm["grade"])
        c2.metric("Average", f"{sm['avg']:.0f} / 100")
        c3.metric("Best rep", f"{sm['best']:.0f}")
        st.write(f"**Weakest corner:** {sm['weakest_corner']} ({sm['per_target'][sm['weakest_corner']]:.0f})  \n"
                 f"**Work on:** {sm['weakest_metric']} - {sm['tip'] or 'keep it up'}")
        if misses:
            st.caption(f"{len(misses)} incomplete rep(s) skipped.")
        if st.session_state.get("saved_clip") == st.session_state.fw_key:
            st.success("Saved to your progress.")
        elif st.button("Save as a training session"):
            sess = History().start("footwork", dict(source="video", width_m=area_w, depth_m=area_d))
            for r in reps:
                sess.add(r.score, r.detail())
            st.session_state.saved_clip = st.session_state.fw_key
            for line in sess.compare():
                st.write(line)
    rows = [dict(rep=r.idx, corner=r.target, score=round(r.score), grade=r.grade,
                 move_s=round(r.times["move"], 2), recovery_s=round(r.times["recover"], 2),
                 **{c.name: (None if c.points is None else round(c.points)) for c in r.checks if c.weight})
            for r in reps]
    st.subheader("Reps (metric columns are points out of 100)")
    st.dataframe(pd.DataFrame(rows).set_index("rep"), width="stretch")
    for r in reps:
        with st.expander(f"Rep {r.idx} - {r.target}: {r.score:.0f} ({r.grade})"):
            if r.top_fix:
                st.warning(f"**Top fix** - {r.top_fix}")
            for ch in r.checks:
                show_check(ch)


with tab_video:
    if footwork:
        st.write("Film from behind (or a high corner) so the **whole area and your feet** are in view the entire "
                 "time. Start each rep standing still in the centre.")
    else:
        st.write("Record a side-on or 45-degree clip (phone is fine) with the whole body in frame, one overhead "
                 "stroke per clip works best."
                 + (" For clears and drops, side-on is best: the contact-position check needs to see which way "
                    "you face." if stroke != "smash" else ""))
    up = st.file_uploader("Upload a clip", type=["mp4", "mov", "avi", "mkv", "webm"])
    if up is not None:
        key, data = load_clip(up)
        if footwork:
            footwork_video(key, data)
        else:
            stroke_video(data)

with tab_live:
    if footwork:
        st.write("Live footwork needs mouse clicks to mark your area and spoken call-outs, so it runs in the "
                 "native window:")
        st.code(f"python live.py --drill footwork --court {area_w:g}x{area_d:g}", language="bash")
        st.markdown("- First run: click the 4 corners of your area (saved for next time; `c` to redo).\n"
                    "- Stand in the blue base circle. When a corner is called (on screen + spoken), go, lunge, "
                    "come back.\n"
                    "- After 12 reps you get a session card comparing against your best session. `r` = new session.")
    else:
        st.write(f"Real-time mode: stand so your whole body is visible and play your {stroke}. The video freezes on "
                 "your contact frame with your grade; press **Redo** to try again.")
        try:
            from streamlit_webrtc import webrtc_streamer
            import av
            from live import LiveAnalyzer

            @st.cache_resource
            def get_live():
                return {"la": None}

            holder = get_live()
            if holder["la"] is None or holder["la"].cfg != cfg or holder["la"].stroke != stroke:
                holder["la"] = LiveAnalyzer(cfg, history=History(), stroke=stroke)

            def cb(frame):
                img = frame.to_ndarray(format="bgr24")
                if img.shape[1] > 960:
                    import cv2
                    img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
                return av.VideoFrame.from_ndarray(holder["la"].process(img, mirror=True), format="bgr24")

            if st.button("Redo", type="primary"):
                holder["la"].reset()

            webrtc_streamer(key="live", video_frame_callback=cb,
                            media_stream_constraints={"video": True, "audio": False},
                            async_processing=True)
            st.caption("If the browser stream is flaky, run `python live.py` for the same analysis in a native OpenCV window.")
        except Exception as e:
            st.info(f"Browser live mode unavailable ({e}). Run `python live.py` in a terminal for the native-window version.")

with tab_prog:
    import pandas as pd
    key_d = "footwork" if footwork else stroke
    hist = History()
    ss = hist.for_drill(key_d)
    st.subheader(f"{drill} progress")
    if not ss:
        st.info("No sessions yet. Every live rep is saved automatically "
                + ("(and analysed clips when you press Save)." if footwork else "."))
    else:
        bs = hist.best_session(key_d)
        last = ss[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions", len(ss))
        c2.metric("Best rep", f"{hist.best_rep(key_d):.0f}")
        c3.metric("Best session avg", f"{bs['avg']:.0f}" if bs else "-",
                  help=f"Sessions with at least 3 reps")
        prev = ss[-2]["avg"] if len(ss) > 1 else None
        c4.metric("Last session avg", f"{last['avg']:.0f}",
                  None if prev is None else f"{last['avg'] - prev:+.0f} vs previous")
        df = pd.DataFrame([dict(session=i + 1, date=s["started"], reps=s["n"], average=round(s["avg"], 1),
                                best=s["best"]) for i, s in enumerate(ss)]).set_index("session")
        st.line_chart(df[["average", "best"]])
        st.dataframe(df, width="stretch")
        if footwork:
            per = {}
            for s in ss:
                for dt in s.get("details", []):
                    per.setdefault(dt["target"], []).append(dt["score"])
            if per:
                st.subheader("Average score per corner (all sessions)")
                st.bar_chart(pd.Series({k: float(np.mean(v)) for k, v in per.items()}, name="score"))
