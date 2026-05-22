# ============================================================
# streamlit_app/app.py
# Flood Victim Detection v3 - Compact Single-Page Dashboard
#
# Run: streamlit run streamlit_app/app.py
# ============================================================

import os
import sys
import cv2
import numpy as np
import tempfile
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.realtime_pipeline import RealTimeFloodSystem
from video.video_processor import VideoProcessor
from visualization.visualizer import overlay_mask, draw_detections
from configs.config import APP_TITLE

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS - Compact single-page professional theme
# ============================================================
st.markdown("""
<style>
    /* ── Global reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0b0f1a !important;
    }
    .main .block-container {
        padding: 0.6rem 1.2rem 0.4rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111828 !important;
        border-right: 1px solid #253350 !important;
    }
    [data-testid="stSidebar"] .block-container { padding: 1rem 0.8rem !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p { color: #b0c4e8 !important; font-size: 12px !important; }
    [data-testid="stSidebar"] h1 {
        color: #e4eeff !important; font-size: 15px !important;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem;
    }
    [data-testid="stSidebar"] hr { border-color: #253350 !important; margin: 0.5rem 0 !important; }

    /* ── Header ── */
    .fvd-header {
        display: flex; align-items: baseline; gap: 14px;
        padding: 0.3rem 0 0.5rem;
        border-bottom: 1px solid #253350;
        margin-bottom: 0.5rem;
    }
    .fvd-title {
        font-size: 18px; font-weight: 700; letter-spacing: 0.04em;
        color: #eef4ff; font-family: 'Courier New', monospace;
        text-transform: uppercase;
    }
    .fvd-subtitle { font-size: 11px; color: #7a98c0; letter-spacing: 0.06em; }

    /* ── Status badges ── */
    .status-flood {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,60,60,0.12); border: 1px solid #ff4040;
        border-radius: 4px; padding: 4px 12px;
        color: #ff7070; font-size: 11px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .status-no-flood {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(30,210,110,0.12); border: 1px solid #1ed878;
        border-radius: 4px; padding: 4px 12px;
        color: #1ed878; font-size: 11px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; }
    .dot-flood { background: #ff4040; box-shadow: 0 0 5px #ff4040; animation: pulse 1.2s infinite; }
    .dot-safe  { background: #1ed878; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 8px; margin: 0.4rem 0 0.5rem; }
    .metric-card {
        flex: 1; background: #111828; border: 1px solid #253350;
        border-radius: 6px; padding: 8px 10px; text-align: center;
    }
    .metric-label { font-size: 10px; color: #7a98c0; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px; font-weight: 600; }
    .metric-value { font-size: 22px; font-weight: 700; color: #ddeaff; font-family: 'Courier New', monospace; }
    .metric-critical .metric-value { color: #ff5555; }
    .metric-high .metric-value     { color: #ff8833; }
    .metric-medium .metric-value   { color: #ffd700; }
    .metric-low .metric-value      { color: #22e87a; }

    /* ── Confidence bar ── */
    .conf-bar-wrap {
        background: #111828; border: 1px solid #253350; border-radius: 6px;
        padding: 7px 14px; margin-bottom: 0.5rem;
        display: flex; align-items: center; gap: 12px;
    }
    .conf-label { font-size: 10px; color: #7a98c0; letter-spacing: 0.1em; font-weight: 600; white-space: nowrap; }
    .conf-track {
        flex: 1; height: 5px; background: #1e2e46; border-radius: 3px; overflow: hidden;
    }
    .conf-fill  { height: 100%; border-radius: 3px; transition: width 0.4s; }
    .conf-value { font-size: 12px; font-weight: 700; font-family: 'Courier New', monospace; white-space: nowrap; }

    /* ── Panel labels ── */
    .panel-label {
        font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
        color: #8ab0d8; margin-bottom: 4px; font-weight: 700;
        padding-bottom: 4px; border-bottom: 1px solid #253350;
    }

    /* ── Risk table ── */
    .risk-table {
        width: 100%; border-collapse: collapse; font-size: 12px;
        font-family: 'Courier New', monospace;
    }
    .risk-table th {
        background: #111828; color: #7a98c0; font-size: 10px;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 6px 10px; border-bottom: 1px solid #253350;
        text-align: left; font-weight: 700;
    }
    .risk-table td { padding: 5px 10px; border-bottom: 1px solid #1a2840; color: #c4d8f4; }
    .risk-table tr:last-child td { border-bottom: none; }
    .risk-table tr:hover td { background: #141e30; }
    .risk-badge {
        display: inline-block; padding: 2px 8px; border-radius: 3px;
        font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    }
    .r-critical { background:rgba(255,60,60,0.2);  color:#ff6666; border:1px solid rgba(255,60,60,0.5); }
    .r-high     { background:rgba(255,136,51,0.2); color:#ffaa55; border:1px solid rgba(255,136,51,0.5); }
    .r-medium   { background:rgba(255,215,0,0.18); color:#ffd700; border:1px solid rgba(255,215,0,0.5); }
    .r-low      { background:rgba(34,232,122,0.18);color:#22e87a; border:1px solid rgba(34,232,122,0.5); }
    .r-unknown  { background:rgba(122,152,192,0.16);color:#9bb4d8; border:1px solid rgba(122,152,192,0.45); }

    .context-note {
        background: #0d1424; border: 1px solid #253350; border-left: 3px solid #80aaff;
        border-radius: 4px; padding: 9px 12px; margin: 0.45rem 0 0.6rem;
        color: #9db8dc; font-size: 12px; line-height: 1.55;
    }
    .context-note strong { color: #dbe9ff; letter-spacing: 0.08em; text-transform: uppercase; }

    /* ── System info chip ── */
    .sys-chip {
        background: #0d1424; border: 1px solid #253350; border-radius: 4px;
        padding: 7px 10px; font-size: 11px; color: #8ab0d8;
        font-family: 'Courier New', monospace; line-height: 2.0;
    }

    /* ── Tight column images ── */
    [data-testid="stImage"] img { border-radius: 4px !important; }

    /* ── Reduce Streamlit default spacing ── */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
    .stColumns { gap: 0.5rem !important; }
    div[data-testid="column"] { padding: 0 !important; }
    [data-testid="stFileUploader"] { padding: 0 !important; }
    [data-testid="stFileUploader"] > div { padding: 6px !important; }
    h1, h2, h3 { margin: 0 !important; padding: 0 !important; }
    hr { margin: 0.4rem 0 !important; border-color: #1a2540 !important; }
    p  { margin: 0 !important; }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] section {
        border: 1px dashed #253350 !important;
        background: #111828 !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] label { color: #7a98c0 !important; font-size: 11px !important; }
    [data-testid="stFileUploader"] p { color: #7a98c0 !important; }

    /* ── Download btn ── */
    .stDownloadButton button {
        background: transparent !important;
        border: 1px solid #253350 !important;
        color: #7a98c0 !important;
        font-size: 11px !important;
        padding: 4px 12px !important;
        border-radius: 4px !important;
        letter-spacing: 0.08em;
    }
    .stDownloadButton button:hover { border-color: #4a80ff !important; color: #80aaff !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-color: #4a80ff transparent transparent !important; }

    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label { font-size: 12px !important; color: #8ab0d8 !important; }

    /* ── Caption / footer ── */
    .stCaption { color: #4a6888 !important; font-size: 11px !important; }

    /* ── Sidebar section header ── */
    .sys-section-hdr {
        font-size: 10px; color: #5a7898; letter-spacing: .12em;
        text-transform: uppercase; margin-bottom: 6px; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INIT (cached)
# ============================================================
@st.cache_resource
def load_system():
    return RealTimeFloodSystem()

@st.cache_resource
def load_video_processor():
    return VideoProcessor()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<h1 style="margin-bottom:0.6rem;">Controls</h1>', unsafe_allow_html=True)
    st.markdown("---")

    input_mode = st.radio(
        "Input Source",
        ["Image Upload", "Video Upload", "Webcam"],
        index=0,
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown('<p class="sys-section-hdr">System Info</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sys-chip">'
        'Model &nbsp;:&nbsp; Attention U-Net<br>'
        'Detector :&nbsp; YOLOv8n<br>'
        'Tracker &nbsp;:&nbsp; Centroid<br>'
        'Device &nbsp;:&nbsp; CPU'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    show_mask    = st.checkbox("Flood Mask",    value=True)
    show_overlay = st.checkbox("Flood Overlay", value=True)

    st.markdown("---")
    st.markdown(
        '<p class="stCaption"></p>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="fvd-header">'
    '<span class="fvd-title">Flood Victim Detection v3</span>'
    '<span class="fvd-subtitle">Attention U-Net &amp; YOLOv8 · Real-Time Pipeline</span>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_model_bar(label_text, confidence):
    pct   = confidence * 100
    color = "#1ec864" if pct < 30 else "#ff8833" if pct < 60 else "#ff3232"
    label = "LOW" if pct < 30 else "MODERATE" if pct < 60 else "HIGH"
    st.markdown(
        f'<div class="conf-bar-wrap">'
        f'<span class="conf-label">{label_text}</span>'
        f'<div class="conf-track"><div class="conf-fill" style="width:{pct:.0f}%;background:{color};"></div></div>'
        f'<span class="conf-value" style="color:{color};">{label} {pct:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_conf_bar(confidence):
    render_model_bar("SEG CONFIDENCE", confidence)


def render_metrics(result):
    dets = result.get("detections", [])
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    for d in dets:
        r = d.get("risk", "LOW")
        counts[r] = counts.get(r, 0) + 1

    st.markdown(
        f'<div class="metric-row">'
        f'<div class="metric-card"><div class="metric-label">Persons</div><div class="metric-value">{len(dets)}</div></div>'
        f'<div class="metric-card metric-critical"><div class="metric-label">Critical</div><div class="metric-value">{counts["CRITICAL"]}</div></div>'
        f'<div class="metric-card metric-high"><div class="metric-label">High</div><div class="metric-value">{counts["HIGH"]}</div></div>'
        f'<div class="metric-card metric-medium"><div class="metric-label">Medium</div><div class="metric-value">{counts["MEDIUM"]}</div></div>'
        f'<div class="metric-card metric-low"><div class="metric-label">Low</div><div class="metric-value">{counts["LOW"]}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_risk_table(detections):
    if not detections:
        return
    rows_html = ""
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box"]
        risk  = det.get("risk", "LOW")
        cls   = f"r-{risk.lower()}"
        rows_html += (
            f'<tr>'
            f'<td>{i}</td>'
            f'<td>{det.get("track_id", "-")}</td>'
            f'<td><span class="risk-badge {cls}">{risk}</span></td>'
            f'<td>{det.get("overlap", 0):.1f}%</td>'
            f'<td>({x1},{y1})→({x2},{y2})</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="risk-table">'
        f'<thead><tr><th>#</th><th>Track ID</th><th>Risk</th><th>Overlap</th><th>Bounding Box</th></tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )


def panel(label):
    st.markdown(f'<p class="panel-label">{label}</p>', unsafe_allow_html=True)


def is_likely_map_or_diagram(image, detections):
    if detections:
        return False

    small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    white_ratio = float(np.mean(np.all(small > 215, axis=2)))
    pale_ratio = float(np.mean((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 130)))
    edge_ratio = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    quantized = (small // 32).reshape(-1, 3)
    color_bins = len(np.unique(quantized, axis=0))

    return (
        white_ratio > 0.30
        or (white_ratio > 0.18 and edge_ratio > 0.045)
        or (pale_ratio > 0.55 and color_bins < 120)
    )


def render_context_note():
    st.markdown(
        '<div class="context-note">'
        '<strong>Flood-related image detected.</strong><br>'
        'No persons detected. Segmentation is not applicable for maps, diagrams, or infographics.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN INFERENCE  — single-page compact layout
# ============================================================
def run_inference(frame, system, show_input=True):
    with st.spinner("Analysing…"):
        result = system.process_frame(frame)

    decision = result["decision"]
    cls_conf = result.get("classification", {}).get("flood_probability", 0.0)

    # ── Status + confidence row ──────────────────────────────
    row_a, row_b = st.columns([1, 3])
    with row_a:
        if decision == "NO FLOOD":
            st.markdown(
                '<div class="status-no-flood">'
                '<span class="status-dot dot-safe"></span>NO FLOOD</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-flood">'
                '<span class="status-dot dot-flood"></span>FLOOD DETECTED</div>',
                unsafe_allow_html=True,
            )

    with row_b:
        render_model_bar("CLS CONFIDENCE", cls_conf)

    if decision == "NO FLOOD":
        st.markdown("---")
        panel("Uploaded Image")
        st.image(bgr_to_rgb(frame), width="stretch")
        return

    seg  = result["segmentation"]
    mask = seg["mask"]
    conf = seg["confidence"]
    dets = result.get("detections", [])
    map_or_diagram = is_likely_map_or_diagram(frame, dets)

    with row_b:
        render_conf_bar(conf)

    # ── Metrics row ──────────────────────────────────────────
    render_metrics(result)

    if map_or_diagram:
        render_context_note()

    st.markdown("---")

    # ── Image grid: determine active columns ─────────────────
    annotated = frame.copy()
    annotated = draw_detections(annotated, dets)

    active_cols  = ["input"] if map_or_diagram else ["input", "detection"]
    if not map_or_diagram and show_overlay: active_cols.append("overlay")
    if not map_or_diagram and show_mask:    active_cols.append("mask")

    cols = st.columns(len(active_cols))
    col_map = dict(zip(active_cols, cols))

    with col_map["input"]:
        panel("Uploaded Image")
        st.image(bgr_to_rgb(frame), width="stretch")

    if "detection" in col_map:
        with col_map["detection"]:
            panel("Detection Result")
            st.image(bgr_to_rgb(annotated), width="stretch")

    if "overlay" in col_map:
        with col_map["overlay"]:
            panel("Flood Overlay")
            colored_mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            overlay_img  = cv2.addWeighted(frame, 0.65, colored_mask, 0.35, 0)
            st.image(bgr_to_rgb(overlay_img), width="stretch")

    if "mask" in col_map:
        with col_map["mask"]:
            panel("Flood Mask")
            st.image(mask, width="stretch", clamp=True)

    # ── Risk table + download ────────────────────────────────
    st.markdown("---")
    tbl_col, dl_col = st.columns([5, 1])
    with tbl_col:
        panel("Per-Person Risk Details")
        if dets:
            render_risk_table(dets)
        else:
            st.markdown('<p style="font-size:12px;color:#7a98c0;">No persons detected.</p>', unsafe_allow_html=True)

    with dl_col:
        _, buf = cv2.imencode(".png", annotated)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="↓ Download",
            data=buf.tobytes(),
            file_name="flood_detection.png",
            mime="image/png",
        )


# ============================================================
# INPUT MODES
# ============================================================
system          = load_system()
video_processor = load_video_processor()

if input_mode == "Image Upload":
    up_col, _ = st.columns([2, 5])
    with up_col:
        uploaded = st.file_uploader(
            "Upload flood image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        run_inference(frame, system)
    else:
        st.markdown(
            '<p style="font-size:12px;color:#5a7898;margin-top:2rem;text-align:center;">'
            'Upload a JPG / PNG to begin analysis.</p>',
            unsafe_allow_html=True,
        )

elif input_mode == "Video Upload":
    up_col, _ = st.columns([2, 5])
    with up_col:
        uploaded_vid = st.file_uploader(
            "Upload flood video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
        )
    if uploaded_vid:
        tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_in.write(uploaded_vid.read()); tmp_in.flush()

        if st.button("Process Video"):
            with st.spinner("Processing on CPU — may take a while…"):
                out_path = video_processor.process(tmp_in.name, tmp_out.name)
            st.success("Done.")
            with open(out_path, "rb") as f:
                st.download_button(
                    "↓ Download Annotated Video",
                    data=f.read(),
                    file_name="annotated_flood_video.mp4",
                    mime="video/mp4",
                )

elif input_mode == "Webcam":
    st.markdown('<p style="font-size:12px;color:#7a98c0;">Capture a frame — the system will analyse it instantly.</p>', unsafe_allow_html=True)
    cam_img = st.camera_input("", label_visibility="collapsed")
    if cam_img:
        file_bytes = np.frombuffer(cam_img.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        run_inference(frame, system)
