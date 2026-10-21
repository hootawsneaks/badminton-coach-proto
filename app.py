"""Streamlit demo: badminton form coach (overhead clear / smash).   Run:  streamlit run app.py"""
import os
import tempfile

import numpy as np
import streamlit as st

import analysis as A

st.set_page_config(page_title="Badminton Smash Coach", layout="wide")
st.title("Badminton Smash Coach")
st.caption("Computer-vision smash analysis. Pose estimation -> swing detection -> "
           "metrics graded against a smash benchmark -> letter grade + coaching feedback.")

# ------------------------------------------------------------ sidebar: thresholds
d = A.Config()
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
    arm_opt = st.radio("Racket arm", ["Auto-detect", "Right", "Left"])
    arm = None if arm_opt == "Auto-detect" else arm_opt.lower()

tab_video, tab_live = st.tabs(["Analyze a video", "Live webcam"])

ICON = {"good": "OK", "warn": "FIX", "info": "INFO", "na": "-"}


def show_report(rep: A.Report):
    if not rep.ok:
        st.warning(rep.reason)
        return
    c1, c2 = st.columns(2)
    c1.metric("Smash grade", rep.grade)
    c2.metric("Score", f"{rep.score:.0f} / 100")
    st.caption(f"Racket arm: {rep.arm}")
    if rep.top_fix:
        st.warning(f"**Top fix** - {rep.top_fix}")
    for ch in rep.checks:
        fn = {"good": st.success, "warn": st.error, "info": st.info}.get(ch.status, st.write)
        pts = f" - {ch.points:.0f}/100" if ch.points is not None else ""
        fn(f"**{ICON[ch.status]} - {ch.name}**{pts} (pass {ch.target}): {ch.message}")


with tab_video:
    st.write("Record a side-on or 45-degree clip (phone is fine) with the whole body in frame, one overhead stroke per clip works best.")
    up = st.file_uploader("Upload a clip", type=["mp4", "mov", "avi", "mkv", "webm"])
    if up is not None:
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
            rep = A.analyze(data["kps"], data["fps"], cfg, arm)
            raw = os.path.join(td, "ann.mp4")
            A.annotate_video(data["frames"], data["kps"], rep, data["fps"], raw)
            h264 = os.path.join(td, "ann_h264.mp4")
            final = h264 if A.to_h264(raw, h264) else raw
            with open(final, "rb") as f:
                vid_bytes = f.read()
        left, right = st.columns([3, 2])
        with left:
            st.video(vid_bytes)
            st.download_button("Download annotated video", vid_bytes, "annotated.mp4", "video/mp4")
        with right:
            show_report(rep)
        if rep.ok:
            import pandas as pd
            t = np.arange(len(rep.speed)) / rep.fps
            st.subheader("Swing signals")
            st.line_chart(pd.DataFrame({"time (s)": t, "wrist speed (torso/s)": rep.speed, "elbow angle (deg)": rep.elbow}).set_index("time (s)"))
            # key-frame stills
            c1, c2 = st.columns(2)
            import cv2
            k = A._smooth(A._interp_nan(data["kps"]), 5)
            for col, idx, lab in ((c1, rep.prep_idx, "Preparation"), (c2, rep.contact_idx, "Contact")):
                im = data["frames"][idx].copy()
                A.draw_pose(im, k[idx], highlight_arm=rep.arm)
                col.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=f"{lab} (frame {idx})")

with tab_live:
    st.write("Real-time mode: stand so your whole body is visible and smash. The video freezes on your "
             "contact frame with your grade; press **Redo** to try again.")
    try:
        from streamlit_webrtc import webrtc_streamer
        import av
        from live import LiveAnalyzer

        @st.cache_resource
        def get_live():
            return {"la": None}

        holder = get_live()
        if holder["la"] is None or holder["la"].cfg != cfg:
            holder["la"] = LiveAnalyzer(cfg)

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
