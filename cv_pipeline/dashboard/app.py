"""
Streamlit live dashboard.
Run: streamlit run cv_pipeline/dashboard/app.py
"""
from __future__ import annotations

import base64
import json
import time
from collections import defaultdict, deque
from threading import Event, Thread
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

# --- Page config (warm white + little black theme) ---
st.set_page_config(
    page_title="CV Pipeline — Object Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Warm white background */
    .stApp { background-color: #FAF9F6; color: #1A1A1A; }
    section[data-testid="stSidebar"] { background-color: #F0EDE8; }
    section[data-testid="stSidebar"] * { color: #1A1A1A !important; }

    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8E4DF;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label { font-size: 12px; color: #6B6560; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1A1A1A; margin-top: 2px; }
    .metric-sub   { font-size: 12px; color: #9B9590; }

    /* Alert banner */
    .alert-box {
        background: #FFF3E0;
        border-left: 4px solid #F5A623;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #7A4800;
    }

    /* Header */
    .cv-header {
        background: linear-gradient(135deg, #1A1A1A 0%, #2C2C2C 100%);
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .cv-header h1 { color: #FAF9F6; font-size: 22px; margin: 0; font-weight: 700; }
    .cv-header p  { color: #B0ADA8; font-size: 13px; margin: 4px 0 0 0; }

    /* Feed frame */
    img { border-radius: 8px; }

    /* Chart label */
    .chart-title { font-size: 13px; font-weight: 600; color: #4A4540; margin: 12px 0 4px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar config ---
with st.sidebar:
    st.markdown("### Source")
    source_type = st.selectbox("Input", ["Webcam", "File", "RTSP URL"])
    if source_type == "Webcam":
        cam_index = st.number_input("Camera index", value=0, min_value=0, step=1)
        source = str(int(cam_index))
    elif source_type == "File":
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        source = None
    else:
        source = st.text_input("RTSP URL", placeholder="rtsp://...")

    st.markdown("---")
    st.markdown("### Model")
    model_name = st.selectbox("YOLOv8 variant", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    confidence = st.slider("Confidence threshold", 0.1, 0.95, 0.4, 0.05)

    st.markdown("---")
    st.markdown("### Overlays")
    show_heatmap = st.checkbox("Heatmap", value=False)
    show_line = st.checkbox("Line counter", value=True)
    show_zones = st.checkbox("Zone counter", value=False)

    st.markdown("---")
    st.markdown("### Recording")
    record = st.checkbox("Save annotated video", value=False)
    out_path = st.text_input("Output path", "output.mp4")

    st.markdown("---")
    st.markdown("### Alerts")
    alert_class = st.text_input("Alert class (e.g. person)", "person")
    alert_threshold = st.number_input("Max count before alert", value=5, min_value=1)


# --- Header ---
st.markdown(
    """
    <div class="cv-header">
      <h1>Computer Vision — Real-Time Object Intelligence</h1>
      <p>YOLOv8 detection · ByteTrack tracking · line counting · zone analytics · heatmap</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Layout ---
col_feed, col_stats = st.columns([3, 1])

with col_feed:
    feed_placeholder = st.empty()
    alert_placeholder = st.empty()

with col_stats:
    fps_placeholder = st.empty()
    obj_placeholder = st.empty()
    in_placeholder = st.empty()
    out_placeholder = st.empty()
    st.markdown('<div class="chart-title">Class Distribution</div>', unsafe_allow_html=True)
    pie_placeholder = st.empty()

st.markdown('<div class="chart-title">Detection Timeline</div>', unsafe_allow_html=True)
timeline_placeholder = st.empty()

# --- Run button ---
run_col, stop_col = st.columns([1, 1])
run_btn = run_col.button("▶  Start", type="primary", use_container_width=True)
stop_btn = stop_col.button("⏹  Stop", use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = False
if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# --- Main processing loop ---
if st.session_state.running:
    from cv_pipeline.analytics.counter import LineCounter, ZoneCounter
    from cv_pipeline.analytics.heatmap import HeatmapGenerator
    from cv_pipeline.detector.yolo_detector import get_detector
    from cv_pipeline.stream.video_source import VideoSource
    from cv_pipeline.tracker.bytetrack import ObjectTracker
    from cv_pipeline.utils.drawing import Annotator

    detector = get_detector(model_name=model_name, confidence=confidence)
    tracker = ObjectTracker()
    annotator = Annotator()

    src = source if source_type != "File" else None
    if source_type == "File" and uploaded:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        src = tmp.name

    if src is None:
        st.error("Please provide a valid source.")
        st.stop()

    video_src = VideoSource.from_string(str(src))
    video_src.open()
    w, h = video_src.frame_size

    heatmap_gen = HeatmapGenerator((h, w))
    lx1, ly1, lx2, ly2 = 0, h // 2, w, h // 2
    line_counter = LineCounter((lx1, ly1), (lx2, ly2))

    writer: Optional[cv2.VideoWriter] = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, video_src.fps, (w, h))

    det_timeline: deque = deque(maxlen=120)
    fps_hist: deque = deque(maxlen=30)
    import plotly.graph_objects as go
    import pandas as pd

    try:
        for frame in video_src.frames():
            if not st.session_state.running:
                break

            t0 = time.monotonic()
            detections = detector.detect(frame)
            tracked = tracker.update(detections, frame.shape)

            heatmap_gen.update(tracked)
            in_c, out_c = line_counter.update(tracked)

            annotated = frame.copy()
            if show_heatmap:
                annotated = heatmap_gen.overlay(annotated)
            annotated = annotator.draw_detections(annotated, tracked)
            if show_line:
                annotated = annotator.draw_line(annotated, (lx1, ly1), (lx2, ly2), in_c, out_c)

            fps = 1.0 / max(time.monotonic() - t0, 1e-6)
            fps_hist.append(fps)
            avg_fps = sum(fps_hist) / len(fps_hist)
            cls_cnt: Dict[str, int] = defaultdict(int)
            for d in tracked:
                cls_cnt[d.class_name] += 1

            annotated = annotator.draw_stats(annotated, avg_fps, len(tracked), cls_cnt)

            if writer:
                writer.write(annotated)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            feed_placeholder.image(rgb, channels="RGB", use_column_width=True)

            # Stats cards
            fps_placeholder.markdown(
                f'<div class="metric-card"><div class="metric-label">FPS</div>'
                f'<div class="metric-value">{avg_fps:.1f}</div></div>',
                unsafe_allow_html=True,
            )
            obj_placeholder.markdown(
                f'<div class="metric-card"><div class="metric-label">Objects</div>'
                f'<div class="metric-value">{len(tracked)}</div></div>',
                unsafe_allow_html=True,
            )
            in_placeholder.markdown(
                f'<div class="metric-card"><div class="metric-label">Line IN</div>'
                f'<div class="metric-value">{in_c}</div></div>',
                unsafe_allow_html=True,
            )
            out_placeholder.markdown(
                f'<div class="metric-card"><div class="metric-label">Line OUT</div>'
                f'<div class="metric-value">{out_c}</div></div>',
                unsafe_allow_html=True,
            )

            # Alert
            if alert_class and cls_cnt.get(alert_class, 0) >= alert_threshold:
                alert_placeholder.markdown(
                    f'<div class="alert-box">⚠ Alert: {cls_cnt[alert_class]} "{alert_class}" detected (threshold: {alert_threshold})</div>',
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.empty()

            # Pie chart
            if cls_cnt:
                fig_pie = go.Figure(go.Pie(
                    labels=list(cls_cnt.keys()),
                    values=list(cls_cnt.values()),
                    hole=0.45,
                    marker=dict(colors=["#D4A574", "#8B7355", "#C4956A", "#A0785A", "#6B4F3A"]),
                    textfont=dict(size=11),
                ))
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1A1A1A", size=10),
                    legend=dict(font=dict(size=9)),
                )
                pie_placeholder.plotly_chart(fig_pie, use_container_width=True)

            # Timeline
            det_timeline.append(len(tracked))
            fig_line = go.Figure(go.Scatter(
                y=list(det_timeline),
                mode="lines",
                fill="tozeroy",
                line=dict(color="#D4A574", width=2),
                fillcolor="rgba(212,165,116,0.15)",
            ))
            fig_line.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=120,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#E8E4DF", zeroline=False, tickfont=dict(size=9)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            timeline_placeholder.plotly_chart(fig_line, use_container_width=True)

    finally:
        video_src.close()
        if writer:
            writer.release()
