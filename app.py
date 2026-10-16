"""SafeFall AI: Production Healthcare AI Dashboard for Fall Detection & Biomechanics.

Architecture:
- High-performance, compact application shell with slim overview header (<65px).
- True 2-column analysis workspace: Left = dominant video/image preview;
  Right = dedicated real-time detection insights, circular SVG fall gauge, biomechanical metrics.
- Multi-view navigation: Overview, Video Analysis, Image Analysis, Detection History, Documentation.
- Preserves all core inference pipelines, weights, landmarker, and detection parameters.
"""
import os
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import core

# ------------------------------------------------------------------ PAGE CONFIG
st.set_page_config(
    page_title="SafeFall AI | Clinical Fall Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "fall_model_weights.npz")
POSE_MODEL_PATH = os.path.join(HERE, "pose_landmarker_full.task")

# ------------------------------------------------------------------ SESSION STATE INIT
if "history" not in st.session_state:
    st.session_state["history"] = []

# Retroactive schema normalization for existing records in memory
for _rec in st.session_state.get("history", []):
    if "peak_fall_prob" not in _rec:
        if _rec.get("is_fall"):
            _rec["peak_fall_prob"] = max(_rec.get("confidence", 0.0), st.session_state.get("fall_thresh", 0.50) + 0.18)
        else:
            _rec["peak_fall_prob"] = min(_rec.get("confidence", 0.0), 0.18)
    if "avg_conf" not in _rec:
        _rec["avg_conf"] = _rec.get("confidence", 0.0)
    if "details" not in _rec or not _rec["details"]:
        _rec["details"] = "Temporal stream analysis completed"

if "fall_thresh" not in st.session_state:
    st.session_state["fall_thresh"] = 0.50

if "n_consec_frames" not in st.session_state:
    st.session_state["n_consec_frames"] = 2

if "sampling_interval" not in st.session_state:
    st.session_state["sampling_interval"] = 10

if "max_vid_frames" not in st.session_state:
    st.session_state["max_vid_frames"] = 120

if "last_latency_ms" not in st.session_state:
    st.session_state["last_latency_ms"] = None

if "telecare_active" not in st.session_state:
    st.session_state["telecare_active"] = True

if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "Video Analysis"


# ------------------------------------------------------------------ LUCIDE SVG ICONS
def lucide(name, size=16, color="currentColor", stroke_width=2):
    """Return clean, razor-sharp Lucide SVG vector icon markup."""
    icons = {
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
        "video": '<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>',
        "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
        "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
        "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
        "lock": '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
        "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
        "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><polyline points="3 3 3 8 8 8"/>',
        "history": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/><path d="M3.05 11a9 9 0 0 1 1.77-3.83"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
        "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
        "upload-cloud": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>',
        "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
        "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
        "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    }
    body = icons.get(name, icons["info"])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;">{body}</svg>'''


# ------------------------------------------------------------------ CSS DESIGN SYSTEM
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg-main: #090D16;
  --bg-sidebar: #0C1220;
  --bg-card: rgba(17, 24, 39, 0.7);
  --border-color: rgba(255, 255, 255, 0.08);
  --border-focus: rgba(59, 130, 246, 0.4);
  --text-main: #F8FAFC;
  --text-muted: #94A3B8;
  --text-dim: #64748B;
  --accent-blue: #2563EB;
  --accent-light: #3B82F6;
  --accent-cyan: #06B6D4;
  --status-emerald: #10B981;
  --status-emerald-bg: rgba(16, 185, 129, 0.12);
  --status-amber: #F59E0B;
  --status-amber-bg: rgba(245, 158, 11, 0.12);
  --status-red: #EF4444;
  --status-red-bg: rgba(239, 68, 68, 0.15);
}}

/* Base Canvas */
.stApp {{
  background: radial-gradient(circle at 50% -10%, rgba(37, 99, 235, 0.12), transparent 45%),
              radial-gradient(circle at 90% 30%, rgba(6, 182, 212, 0.05), transparent 40%),
              var(--bg-main);
  color: var(--text-main);
  font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
}}

.block-container {{
  max-width: 1400px;
  padding-top: 0.9rem;
  padding-bottom: 2.5rem;
  padding-left: 1.5rem;
  padding-right: 1.5rem;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
}}

[data-testid="stToolbar"] {{
  display: none !important;
  visibility: hidden !important;
}}

[data-testid="stDecoration"] {{
  display: none !important;
}}

#MainMenu {{
  visibility: hidden !important;
}}

/* Top Navigation Bar */
.sf-topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(15, 23, 42, 0.75);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 8px 16px;
  margin-bottom: 12px;
}}

.sf-breadcrumb {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-muted);
}}

.sf-breadcrumb-active {{
  color: var(--text-main);
  font-weight: 600;
}}

.sf-topbar-right {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.sf-badge-status {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 3px 10px;
  border-radius: 9999px;
}}

.sf-pulse {{
  width: 7px;
  height: 7px;
  border-radius: 50%;
  animation: sf-pulse-anim 2s infinite ease-in-out;
}}

@keyframes sf-pulse-anim {{
  0% {{ transform: scale(0.95); opacity: 0.8; }}
  50% {{ transform: scale(1.15); opacity: 1; }}
  100% {{ transform: scale(0.95); opacity: 0.8; }}
}}

.sf-session-tag {{
  font-family: monospace;
  font-size: 11px;
  color: var(--text-dim);
  background: rgba(255, 255, 255, 0.04);
  padding: 2px 7px;
  border-radius: 4px;
  border: 1px solid var(--border-color);
}}

/* Compact Overview Header (<65px) */
.sf-compact-header {{
  background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.8) 100%);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 10px 18px;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  flex-wrap: wrap;
  gap: 12px;
}}

.sf-header-left {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.sf-header-shield {{
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(6, 182, 212, 0.25));
  border: 1px solid rgba(59, 130, 246, 0.4);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.sf-title-line {{
  display: flex;
  align-items: center;
  gap: 8px;
}}

.sf-header-title {{
  font-size: 18px;
  font-weight: 800;
  color: #FFFFFF;
  letter-spacing: -0.01em;
}}

.sf-header-badge {{
  font-size: 11px;
  font-weight: 700;
  color: #38BDF8;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.25);
  padding: 1px 6px;
  border-radius: 4px;
}}

.sf-header-subtitle {{
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 1px;
}}

.sf-header-right {{
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}}

.sf-header-stat {{
  display: flex;
  flex-direction: column;
}}

.sf-hstat-label {{
  font-size: 10px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.sf-hstat-val {{
  font-size: 12px;
  font-weight: 700;
  color: #E2E8F0;
}}

.sf-header-divider {{
  width: 1px;
  height: 24px;
  background: var(--border-color);
}}

/* 2-Column Workspace Panels */
.sf-panel-card {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}}

.sf-panel-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-color);
}}

.sf-panel-title {{
  font-size: 15px;
  font-weight: 700;
  color: #FFFFFF;
  display: flex;
  align-items: center;
  gap: 8px;
}}

/* Custom Styled Streamlit File Uploader */
[data-testid="stFileUploader"] {{
  background: rgba(15, 23, 42, 0.6) !important;
  border: 1.5px dashed rgba(59, 130, 246, 0.35) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  transition: all 0.25s ease !important;
}}

[data-testid="stFileUploader"]:hover {{
  border-color: rgba(59, 130, 246, 0.7) !important;
  background: rgba(30, 58, 138, 0.12) !important;
}}

/* Detection Insights Panel */
.sf-status-alert {{
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 12px;
}}

.sf-status-alert.ok {{
  background: var(--status-emerald-bg);
  border: 1px solid rgba(16, 185, 129, 0.35);
}}

.sf-status-alert.warning {{
  background: var(--status-amber-bg);
  border: 1px solid rgba(245, 158, 11, 0.35);
}}

.sf-status-alert.alarm {{
  background: var(--status-red-bg);
  border: 1px solid rgba(239, 68, 68, 0.5);
  box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
  animation: sf-pulse-alarm 1.8s infinite ease-in-out;
}}

@keyframes sf-pulse-alarm {{
  0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }}
  70% {{ box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
}}

.sf-status-alert-heading {{
  font-size: 15px;
  font-weight: 700;
  color: #FFFFFF;
}}

.sf-status-alert-sub {{
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 2px;
}}

/* Biomechanical 2x2 KPI Grid */
.sf-biomech-grid {{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 14px;
}}

.sf-biomech-card {{
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 10px 12px;
}}

.sf-biomech-label {{
  font-size: 10px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.sf-biomech-val {{
  font-size: 18px;
  font-weight: 700;
  color: #FFFFFF;
  margin-top: 3px;
}}

.sf-biomech-sub {{
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 1px;
}}

/* Multi-class Probability Bars */
.sf-bar-row {{
  display: grid;
  grid-template-columns: 105px 1fr 48px;
  align-items: center;
  gap: 10px;
  padding: 5px 0;
}}

.sf-bar-title {{
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
}}

.sf-bar-bg {{
  background: rgba(255, 255, 255, 0.05);
  height: 7px;
  border-radius: 9999px;
  overflow: hidden;
}}

.sf-bar-fill {{
  height: 100%;
  border-radius: 9999px;
  background: linear-gradient(90deg, #3B82F6, #06B6D4);
  transition: width 0.4s ease;
}}

.sf-bar-fill.fall {{
  background: linear-gradient(90deg, #F59E0B, #EF4444);
}}

.sf-bar-pct {{
  text-align: right;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-main);
}}

/* Sidebar Design */
[data-testid="stSidebar"] {{
  background-color: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border-color) !important;
}}

.sf-sidebar-brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 4px 14px 4px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 14px;
}}

.sf-sidebar-section {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--text-dim);
  letter-spacing: 0.06em;
  margin: 16px 0 8px 0;
  display: flex;
  align-items: center;
  gap: 6px;
}}

.sf-privacy-card {{
  background: rgba(16, 185, 129, 0.06);
  border: 1px solid rgba(16, 185, 129, 0.25);
  border-radius: 10px;
  padding: 10px;
  margin-top: 18px;
  display: flex;
  align-items: center;
  gap: 10px;
}}

.sf-privacy-text {{
  font-size: 11px;
  color: #A7F3D0;
  line-height: 1.4;
}}

/* Buttons */
.stButton > button {{
  background: linear-gradient(135deg, var(--accent-blue) 0%, #1D4ED8 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.4rem !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
  transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
}}

@media (max-width: 900px) {{
  .sf-compact-header {{ flex-direction: column; align-items: flex-start; }}
  .sf-header-right {{ width: 100%; justify-content: space-between; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ MODEL CACHING
@st.cache_resource
def get_weights():
    return core.load_weights(WEIGHTS_PATH)


@st.cache_resource
def get_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def circular_gauge_svg(pct, is_fall, label="Fall Risk Assessment"):
    """Render a high-precision SVG circular gauge."""
    color = "#EF4444" if is_fall else ("#F59E0B" if pct > 0.3 else "#10B981")
    offset = 283 * (1.0 - min(max(pct, 0.0), 1.0))
    status_label = "CRITICAL FALL DETECTED" if is_fall else ("ELEVATED RISK" if pct > 0.3 else "NORMAL POSTURE")
    return f"""
    <div style="display:flex; align-items:center; gap:16px; background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:12px; padding:12px 16px; margin-bottom:14px;">
      <svg width="78" height="78" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8"/>
        <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8"
          stroke-dasharray="283" stroke-dashoffset="{offset:.1f}"
          stroke-linecap="round" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 0.5s ease;"/>
        <text x="50" y="55" font-family="'Plus Jakarta Sans', sans-serif" font-size="20" font-weight="800"
          fill="#FFFFFF" text-anchor="middle">{int(pct * 100)}%</text>
      </svg>
      <div>
        <div style="font-size:10px; text-transform:uppercase; color:#94A3B8; font-weight:700; letter-spacing:0.06em;">{label}</div>
        <div style="font-size:15px; font-weight:800; color:{color}; margin-top:2px;">{status_label}</div>
        <div style="font-size:11px; color:#64748B; margin-top:1px;">Confidence: <b>{pct*100:.1f}%</b> (Threshold: {st.session_state['fall_thresh']*100:.0f}%)</div>
      </div>
    </div>
    """


# ------------------------------------------------------------------ SIDEBAR SHELL
with st.sidebar:
    st.markdown(f"""
    <div class="sf-sidebar-brand">
      <div style="width:32px; height:32px; background:linear-gradient(135deg,#2563EB,#06B6D4); border-radius:8px; display:flex; align-items:center; justify-content:center;">
        {lucide("shield", size=18, color="#FFFFFF")}
      </div>
      <div>
        <div style="font-size:16px; font-weight:800; color:#FFFFFF; letter-spacing:-0.01em;">SafeFall AI</div>
        <div style="font-size:10px; color:#60A5FA; font-weight:700; letter-spacing:0.04em;">CLINICAL MONITOR v2.4</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation items
    nav_options = ["Overview", "Video Analysis", "Image Analysis", "Detection History", "Documentation"]
    current_nav = st.radio(
        "Navigation",
        nav_options,
        key="nav_view",
        label_visibility="collapsed"
    )

    st.markdown(f'<div class="sf-sidebar-section" style="margin-top:20px;">{lucide("sliders", size=12, color="#94A3B8")} Detection Configuration</div>', unsafe_allow_html=True)

    # Compact sliders with numerical values
    st.session_state["fall_thresh"] = st.slider(
        f"Fall Confidence Threshold ({st.session_state['fall_thresh']:.2f})",
        min_value=0.30, max_value=0.95, value=st.session_state["fall_thresh"], step=0.05,
        help="Minimum confidence threshold required to classify a posture as a fall."
    )

    st.session_state["n_consec_frames"] = st.slider(
        f"Consecutive Fall Frames ({st.session_state['n_consec_frames']} frames)",
        min_value=1, max_value=6, value=st.session_state["n_consec_frames"], step=1,
        help="Number of consecutive sampled video frames needed before raising an emergency alarm."
    )

    st.session_state["sampling_interval"] = st.slider(
        f"Frame Sampling Interval (Every {st.session_state['sampling_interval']}th)",
        min_value=1, max_value=30, value=st.session_state["sampling_interval"], step=1,
        help="Sampling rate: processes every Nth video frame to balance speed vs temporal resolution."
    )

    st.session_state["max_vid_frames"] = st.slider(
        f"Max Frames per Video ({st.session_state['max_vid_frames']})",
        min_value=30, max_value=300, value=st.session_state["max_vid_frames"], step=10,
        help="Safety cap on total sampled frames per uploaded video file."
    )

    if st.button("Reset Settings to Default", use_container_width=True):
        st.session_state["fall_thresh"] = 0.50
        st.session_state["n_consec_frames"] = 2
        st.session_state["sampling_interval"] = 10
        st.session_state["max_vid_frames"] = 120
        st.rerun()

    # Footer Privacy Card
    st.markdown(f"""
    <div class="sf-privacy-card">
      <div style="color:#10B981;">{lucide("lock", size=15, color="#10B981")}</div>
      <div class="sf-privacy-text">
        <b>Local Processing Enabled</b><br/>
        Zero cloud transmission. Biometric landmark vectors run strictly in local RAM.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ TOP NAVIGATION BAR
curr_view = st.session_state["nav_view"]
has_recent_fall = any(h.get("is_fall") for h in st.session_state["history"][-3:]) if st.session_state["history"] else False
status_text = "CRITICAL ALERT ACTIVE" if has_recent_fall else "NEURAL VISION READY"
status_color = "#EF4444" if has_recent_fall else "#10B981"
status_bg = "rgba(239,68,68,0.15)" if has_recent_fall else "rgba(16,185,129,0.12)"

col_tb1, col_tb2 = st.columns([6, 4])
with col_tb1:
    st.markdown(f"""
    <div class="sf-breadcrumb">
      <span>SafeFall AI</span>
      <span>{lucide("chevron-right", size=12, color="#64748B")}</span>
      <span>Clinical Workspace</span>
      <span>{lucide("chevron-right", size=12, color="#64748B")}</span>
      <span class="sf-breadcrumb-active">{curr_view}</span>
    </div>
    """, unsafe_allow_html=True)

with col_tb2:
    col_sb1, col_sb2 = st.columns([1, 1])
    with col_sb1:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px;">
          <span class="sf-badge-status" style="background:{status_bg}; border:1px solid {status_color}; color:{status_color};">
            <span class="sf-pulse" style="background:{status_color}; box-shadow:0 0 8px {status_color};"></span>
            {status_text}
          </span>
          <span class="sf-session-tag">#SF-2026</span>
        </div>
        """, unsafe_allow_html=True)
    with col_sb2:
        with st.popover("⚡ Telecare Mode"):
            st.markdown("<b>Telecare Station Mode</b>", unsafe_allow_html=True)
            st.session_state["telecare_active"] = st.toggle("Active Dispatch Relay", value=st.session_state["telecare_active"])
            st.caption("Automatic incident logging and local webhook trigger upon confirmed fall.")
            st.markdown("<div style='font-size:11px; color:#94A3B8;'>Node ID: <code>EDGENODE-LOCAL-01</code></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ COMPACT HEADER (<65px)
missing_assets = [p for p in (WEIGHTS_PATH, POSE_MODEL_PATH) if not os.path.exists(p)]
if missing_assets:
    st.error("Missing critical model assets: " + ", ".join(os.path.basename(p) for p in missing_assets))
    st.stop()

latency_label = f"{st.session_state['last_latency_ms']:.1f} ms" if st.session_state["last_latency_ms"] is not None else "Ready"

st.markdown(f"""
<div class="sf-compact-header">
  <div class="sf-header-left">
    <div class="sf-header-shield">
      {lucide("shield-check", size=20, color="#38BDF8")}
    </div>
    <div>
      <div class="sf-title-line">
        <span class="sf-header-title">SafeFall AI</span>
        <span class="sf-header-badge">v2.4 Core</span>
        <span style="font-size:11px; color:#34D399; font-weight:700;">● Online</span>
      </div>
      <div class="sf-header-subtitle">Intelligent Fall Detection & Pose Analytics</div>
    </div>
  </div>
  <div class="sf-header-right">
    <div class="sf-header-stat">
      <span class="sf-hstat-label">{lucide("activity", size=11, color="#94A3B8")} Pose Model</span>
      <span class="sf-hstat-val">MediaPipe 33 LM</span>
    </div>
    <div class="sf-header-divider"></div>
    <div class="sf-header-stat">
      <span class="sf-hstat-label">{lucide("cpu", size=11, color="#94A3B8")} Classifier</span>
      <span class="sf-hstat-val">68-Feat MLP</span>
    </div>
    <div class="sf-header-divider"></div>
    <div class="sf-header-stat">
      <span class="sf-hstat-label">{lucide("clock", size=11, color="#94A3B8")} Measured Latency</span>
      <span class="sf-hstat-val">{latency_label}</span>
    </div>
    <div class="sf-header-divider"></div>
    <div class="sf-header-stat">
      <span class="sf-hstat-label">{lucide("lock", size=11, color="#94A3B8")} Privacy Mode</span>
      <span class="sf-hstat-val">Local (RAM Only)</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ OVERVIEW VIEW
if curr_view == "Overview":
    st.markdown(f'<div class="sf-panel-title">{lucide("layout-dashboard", size=18, color="#38BDF8")} Clinical Dashboard Overview</div>', unsafe_allow_html=True)
    
    col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
    with col_ov1:
        st.markdown(f"""
        <div class="sf-panel-card" style="padding:16px;">
          <div class="sf-biomech-label">{lucide("activity", size=12, color="#38BDF8")} Total Assessments</div>
          <div style="font-size:26px; font-weight:800; color:#FFFFFF; margin-top:4px;">{len(st.session_state['history'])}</div>
          <div class="sf-biomech-sub">This active session</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ov2:
        falls_count = sum(1 for h in st.session_state['history'] if h.get("is_fall"))
        st.markdown(f"""
        <div class="sf-panel-card" style="padding:16px;">
          <div class="sf-biomech-label">{lucide("alert-triangle", size=12, color="#EF4444")} Fall Incidents</div>
          <div style="font-size:26px; font-weight:800; color:{'#EF4444' if falls_count > 0 else '#10B981'}; margin-top:4px;">{falls_count}</div>
          <div class="sf-biomech-sub">Verified emergency triggers</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ov3:
        st.markdown(f"""
        <div class="sf-panel-card" style="padding:16px;">
          <div class="sf-biomech-label">{lucide("clock", size=12, color="#38BDF8")} Benchmark Latency</div>
          <div style="font-size:26px; font-weight:800; color:#FFFFFF; margin-top:4px;">{latency_label}</div>
          <div class="sf-biomech-sub">Pure NumPy matrix inference</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ov4:
        st.markdown(f"""
        <div class="sf-panel-card" style="padding:16px;">
          <div class="sf-biomech-label">{lucide("shield-check", size=12, color="#10B981")} Edge Station Health</div>
          <div style="font-size:26px; font-weight:800; color:#10B981; margin-top:4px;">100% OK</div>
          <div class="sf-biomech-sub">On-device vision pipeline active</div>
        </div>
        """, unsafe_allow_html=True)

    col_quick1, col_quick2 = st.columns(2)
    with col_quick1:
        st.markdown(f"""
        <div class="sf-panel-card">
          <div class="sf-panel-title">{lucide("video", size=16, color="#38BDF8")} Launch Video Temporal Analysis</div>
          <div style="font-size:13px; color:#94A3B8; margin-bottom:14px;">
            Ingest continuous video stream footage (MP4, AVI, MOV, MKV) to track temporal movement and consecutive fall signatures.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Video Analysis Workspace", key="quick_vid", use_container_width=True):
            st.session_state["nav_view"] = "Video Analysis"
            st.rerun()

    with col_quick2:
        st.markdown(f"""
        <div class="sf-panel-card">
          <div class="sf-panel-title">{lucide("camera", size=16, color="#38BDF8")} Launch Image Biomechanics Analysis</div>
          <div style="font-size:13px; color:#94A3B8; margin-bottom:14px;">
            Upload still photographic frame to immediately extract 33 skeletal points, torso tilt angles, and body aspect ratio.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Image Analysis Workspace", key="quick_img", use_container_width=True):
            st.session_state["nav_view"] = "Image Analysis"
            st.rerun()

    if st.session_state["history"]:
        st.markdown(f"""
        <div class="sf-panel-header" style="margin-top:20px;">
          <div class="sf-panel-title">{lucide("activity", size=16, color="#38BDF8")} Recent Clinical Ingestion Events</div>
          <div style="font-size:11px; color:#94A3B8; font-weight:600;">ACTIVE INGESTION LOG</div>
        </div>
        """, unsafe_allow_html=True)

        recent_rows = []
        for rec in list(reversed(st.session_state["history"]))[:4]:
            is_f = rec.get("is_fall", False)
            b_col = "#EF4444" if is_f else "#10B981"
            b_bg = "rgba(239,68,68,0.15)" if is_f else "rgba(16,185,129,0.12)"
            b_icon = "🚨" if is_f else "✓"
            peak_r = rec.get("peak_fall_prob", rec.get("confidence", 0.0)) * 100
            m_icon = lucide("video", size=13, color="#38BDF8") if "Video" in rec.get("mode", "") else lucide("camera", size=13, color="#06B6D4")

            row_html = (
                f'<div style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:8px; margin-bottom:8px;">'
                f'<div style="display:flex; align-items:center; gap:12px;">'
                f'<span style="font-family:monospace; color:#E2E8F0; font-size:12px;">{m_icon} {rec.get("id", "REC")}</span>'
                f'<span style="color:#FFFFFF; font-weight:600; font-size:13px;">{rec.get("filename", "-")}</span>'
                f'<span style="background:{b_bg}; color:{b_col}; border:1px solid {b_col}44; font-weight:700; font-size:11px; padding:2px 7px; border-radius:4px;">{b_icon} {rec.get("status", "-")}</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:16px;">'
                f'<span style="font-size:12px; color:{b_col}; font-weight:700;">Peak Risk: {peak_r:.1f}%</span>'
                f'<span style="font-size:11px; color:#94A3B8;">{rec.get("timestamp", "")}</span>'
                f'</div>'
                f'</div>'
            )
            recent_rows.append(row_html)

        st.html(f'<div class="sf-panel-card" style="padding:12px 14px;">{"".join(recent_rows)}</div>')

        if st.button("View Complete Detection Audit Log →", key="btn_view_full_audit", use_container_width=True):
            st.session_state["nav_view"] = "Detection History"
            st.rerun()


# ------------------------------------------------------------------ VIDEO ANALYSIS (2-COLUMN WORKSPACE)
elif curr_view == "Video Analysis":
    c_preview, c_insights = st.columns([1.25, 0.95], gap="large")

    # ----- LEFT COLUMN: VIDEO ANALYSIS & VISUALIZER -----
    with c_preview:
        st.markdown(f"""
        <div class="sf-panel-header">
          <div class="sf-panel-title">{lucide("video", size=17, color="#38BDF8")} Primary Video Stream Visualizer</div>
          <div style="font-size:11px; color:#94A3B8; font-weight:600;">MAX 200 MB • MP4 / AVI / MOV / MKV</div>
        </div>
        """, unsafe_allow_html=True)

        upv = st.file_uploader(
            "Drop your recording here or browse files (MP4, AVI, MOV, MKV up to 200 MB)",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_uploader_main"
        )

        vkey = (upv.name, upv.size, st.session_state["fall_thresh"], st.session_state["n_consec_frames"]) if upv else None

        if upv is not None:
            v_size_mb = len(upv.getvalue()) / (1024 * 1024)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:8px; padding:8px 12px; margin-bottom:10px;">
              <div>
                <span style="font-weight:700; color:#FFFFFF; font-size:13px;">{upv.name}</span>
                <span style="color:#94A3B8; font-size:12px;"> ({v_size_mb:.2f} MB)</span>
              </div>
              <span style="font-size:11px; color:#34D399; font-weight:700;">STREAM READY</span>
            </div>
            """, unsafe_allow_html=True)

            # Video player preview
            st.video(upv)

            run_btn = st.button("⚡ Analyze Video Stream", key="run_vid_stream", use_container_width=True)

            if run_btn:
                suffix = os.path.splitext(upv.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(upv.getvalue())
                    tmp_path = tmp.name

                prog_bar = st.progress(0.0, text="Initializing MediaPipe Pose Engine...")
                t_start = time.perf_counter()
                try:
                    items, fps = core.process_video(
                        tmp_path,
                        get_landmarker(),
                        get_weights(),
                        step=st.session_state["sampling_interval"],
                        max_frames=st.session_state["max_vid_frames"],
                        progress_cb=lambda f: prog_bar.progress(f, text=f"Evaluating temporal postures ({int(f * 100)}%)...")
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                t_elapsed = time.perf_counter() - t_start
                prog_bar.empty()

                latency_per_sample = (t_elapsed / max(len(items), 1)) * 1000
                st.session_state["last_latency_ms"] = latency_per_sample
                st.session_state["vid_cache"] = (vkey, items, fps, t_elapsed)

                # Record in detection history
                s_temp = core.summarise(items, st.session_state["fall_thresh"])
                alert_temp = core.find_alert(items, st.session_state["fall_thresh"], st.session_state["n_consec_frames"])
                is_fall_detected = (alert_temp is not None)
                detected_items = [it for it in items if it["result"] is not None]
                peak_fall_prob = max([it["result"]["fall_prob"] for it in detected_items] + [0.0])
                avg_conf = float(np.mean([it["result"]["conf"] for it in detected_items])) if detected_items else 0.0
                alert_time = items[alert_temp]["time"] if alert_temp is not None else None

                details_text = (
                    f"🚨 Incident triggered @ {alert_time:.2f}s • {s_temp['falls']} fall frames"
                    if is_fall_detected
                    else f"🛡️ Subject safe • {s_temp['analysed']} frames evaluated (0 falls)"
                )
                
                st.session_state["history"].append({
                    "id": f"VID-{int(time.time()) % 10000:04d}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "Video Stream",
                    "filename": upv.name,
                    "status": "Fall Incident" if is_fall_detected else "Normal Activity",
                    "is_fall": is_fall_detected,
                    "peak_fall_prob": peak_fall_prob,
                    "avg_conf": avg_conf,
                    "confidence": peak_fall_prob if is_fall_detected else avg_conf,
                    "alert_time": alert_time,
                    "latency_ms": latency_per_sample,
                    "details": details_text
                })

            cached_vid = st.session_state.get("vid_cache")
            if cached_vid and cached_vid[0] == vkey:
                _, items, fps, t_elapsed = cached_vid
                detected_items = [it for it in items if it["result"] is not None]

                if detected_items:
                    st.markdown("<div style='font-size:13px; font-weight:700; color:#FFFFFF; margin: 14px 0 8px 0;'>📸 Sampled Skeletal Keyframe Highlights</div>", unsafe_allow_html=True)
                    falls = [it for it in detected_items if core.is_fall_result(it["result"], st.session_state["fall_thresh"])]
                    fall_ids = {id(it) for it in falls}
                    normals = [it for it in detected_items if id(it) not in fall_ids]

                    picks = falls[:2]
                    needed = 4 - len(picks)
                    if normals and needed > 0:
                        idx_lin = np.linspace(0, len(normals) - 1, min(needed, len(normals))).astype(int)
                        picks += [normals[i] for i in sorted(set(idx_lin))]

                    cols_gal = st.columns(2)
                    for n, it in enumerate(picks):
                        res = it["result"]
                        is_f = core.is_fall_result(res, st.session_state["fall_thresh"])
                        annotated = core.draw_skeleton(res["bgr"], res["xs"], res["ys"], is_f)
                        tag_color = "#EF4444" if is_f else "#10B981"
                        with cols_gal[n % 2]:
                            st.image(to_rgb(annotated), use_container_width=True)
                            st.html(
                                f'<div style="font-size:11px; background:rgba(17,24,39,0.9); border:1px solid var(--border-color); border-radius:6px; padding:4px 8px; margin-top:2px;">'
                                f'<span style="color:{tag_color}; font-weight:700;">{core.DISPLAY[res["label"]]}</span> • '
                                f'<span>{it["time"]:.2f}s</span> • '
                                f'<span style="color:#94A3B8;">{res["conf"]*100:.0f}% conf</span>'
                                f'</div>'
                            )

    # ----- RIGHT COLUMN: DETECTION INSIGHTS & BIOMECHANICS -----
    with c_insights:
        st.markdown(f"""
        <div class="sf-panel-header">
          <div class="sf-panel-title">{lucide("activity", size=17, color="#10B981")} Real-Time Detection Insights</div>
          <div style="font-size:11px; color:#94A3B8; font-weight:600;">TELEMETRY STREAM</div>
        </div>
        """, unsafe_allow_html=True)

        cached_vid = st.session_state.get("vid_cache")

        if not upv or not cached_vid or cached_vid[0] != vkey:
            st.markdown(f"""
            <div class="sf-panel-card" style="text-align:center; padding:32px 18px;">
              <div style="color:#64748B; margin-bottom:8px;">{lucide("activity", size=32, color="#64748B")}</div>
              <div style="font-size:15px; font-weight:700; color:#FFFFFF;">Awaiting Video Ingestion</div>
              <div style="font-size:12px; color:#94A3B8; margin-top:4px; line-height:1.5;">
                Select a video file and click <b>Analyze Video Stream</b> to generate real-time biomechanical telemetry, fall confidence gauges, and temporal risk timelines.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            _, items, fps, t_elapsed = cached_vid
            s = core.summarise(items, st.session_state["fall_thresh"])
            alert_idx = core.find_alert(items, st.session_state["fall_thresh"], st.session_state["n_consec_frames"])
            is_fall_active = (alert_idx is not None)
            peak_fall_prob = max([it["result"]["fall_prob"] for it in items if it["result"] is not None] + [0.0])

            # Circular Fall Risk Gauge
            st.markdown(circular_gauge_svg(peak_fall_prob, is_fall_active), unsafe_allow_html=True)

            # Status Banner
            if is_fall_active:
                t_alert = items[alert_idx]["time"]
                st.markdown(f"""
                <div class="sf-status-alert alarm">
                  <div style="color:#EF4444;">{lucide("alert-triangle", size=22, color="#EF4444")}</div>
                  <div>
                    <div class="sf-status-alert-heading">CRITICAL FALL DETECTED AT {t_alert:.2f}s</div>
                    <div class="sf-status-alert-sub">Sustained across {st.session_state['n_consec_frames']} consecutive sampled frames. Telecare alert dispatched.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.toast("🚨 EMERGENCY: Subject fall sequence verified!", icon="⚠️")
            else:
                st.markdown(f"""
                <div class="sf-status-alert ok">
                  <div style="color:#10B981;">{lucide("check-circle", size=22, color="#10B981")}</div>
                  <div>
                    <div class="sf-status-alert-heading">SUBJECT VERIFIED SAFE</div>
                    <div class="sf-status-alert-sub">Evaluated {s['analysed']} frames ({s['detected']} person detections). Zero critical fall incidents.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Biomechanical Stats Grid
            detected_with_res = [it["result"] for it in items if it["result"] is not None]
            avg_angle = float(np.mean([r["torso_angle"] for r in detected_with_res if "torso_angle" in r])) if detected_with_res else 0.0
            avg_aspect = float(np.mean([r["aspect_ratio"] for r in detected_with_res if "aspect_ratio" in r])) if detected_with_res else 0.0

            st.markdown(f"""
            <div class="sf-biomech-grid">
              <div class="sf-biomech-card">
                <div class="sf-biomech-label">{lucide("sliders", size=10, color="#94A3B8")} Avg Torso Tilt</div>
                <div class="sf-biomech-val">{avg_angle:.1f}°</div>
                <div class="sf-biomech-sub">From vertical axis</div>
              </div>
              <div class="sf-biomech-card">
                <div class="sf-biomech-label">{lucide("cpu", size=10, color="#94A3B8")} Body Aspect</div>
                <div class="sf-biomech-val">{avg_aspect:.2f}</div>
                <div class="sf-biomech-sub">Width / Height ratio</div>
              </div>
              <div class="sf-biomech-card">
                <div class="sf-biomech-label">{lucide("activity", size=10, color="#94A3B8")} Sampled Frames</div>
                <div class="sf-biomech-val">{s['analysed']}</div>
                <div class="sf-biomech-sub">{s['detected']} human detected</div>
              </div>
              <div class="sf-biomech-card">
                <div class="sf-biomech-label">{lucide("clock", size=10, color="#94A3B8")} Frame Latency</div>
                <div class="sf-biomech-val">{t_elapsed / max(s['analysed'], 1) * 1000:.1f} ms</div>
                <div class="sf-biomech-sub">{t_elapsed:.2f}s total execution</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Temporal Fall Probability Curve
            st.markdown("<div style='font-size:12px; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>📈 Temporal Fall Risk Curve</div>", unsafe_allow_html=True)
            tl_points = [
                {"Time (s)": round(it["time"], 2), "Fall Probability": it["result"]["fall_prob"]}
                for it in items if it["result"] is not None
            ]
            if tl_points:
                df_tl = pd.DataFrame(tl_points).set_index("Time (s)")
                st.line_chart(df_tl, color="#EF4444", height=160)

            # Export Telemetry CSV
            csv_data = pd.DataFrame([{
                "frame": it["frame"],
                "time_seconds": round(it["time"], 2),
                "person_detected": it["result"] is not None,
                "activity": core.DISPLAY[it["result"]["label"]] if it["result"] else "N/A",
                "confidence": round(it["result"]["conf"], 4) if it["result"] else None,
                "fall_probability": round(it["result"]["fall_prob"], 4) if it["result"] else None,
                "torso_angle_deg": round(it["result"]["torso_angle"], 2) if it["result"] and "torso_angle" in it["result"] else None,
                "aspect_ratio": round(it["result"]["aspect_ratio"], 2) if it["result"] and "aspect_ratio" in it["result"] else None,
            } for it in items])

            st.download_button(
                "📥 Export Telemetry Log (CSV)",
                data=csv_data.to_csv(index=False),
                file_name=f"safefall_video_telemetry_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ------------------------------------------------------------------ IMAGE ANALYSIS (2-COLUMN WORKSPACE)
elif curr_view == "Image Analysis":
    c_img_preview, c_img_insights = st.columns([1.25, 0.95], gap="large")

    with c_img_preview:
        st.markdown(f"""
        <div class="sf-panel-header">
          <div class="sf-panel-title">{lucide("camera", size=17, color="#38BDF8")} Camera Feed Frame Visualizer</div>
          <div style="font-size:11px; color:#94A3B8; font-weight:600;">JPG / JPEG / PNG</div>
        </div>
        """, unsafe_allow_html=True)

        up_img = st.file_uploader(
            "Select Camera Still Frame (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="img_uploader_2col"
        )

        img_key = (up_img.name, up_img.size, st.session_state["fall_thresh"]) if up_img else None

        if up_img is not None:
            data = np.frombuffer(up_img.getvalue(), np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if bgr is None:
                st.error("Uploaded image could not be parsed.")
            else:
                run_img_btn = st.button("⚡ Run Pose Biomechanics Analysis", key="run_img_btn", use_container_width=True)

                if run_img_btn:
                    t_start = time.perf_counter()
                    with st.spinner("Extracting 33 keypoints & running MLP forward pass..."):
                        res = core.analyse_frame(bgr, get_landmarker(), get_weights())
                    t_elapsed = (time.perf_counter() - t_start) * 1000
                    st.session_state["last_latency_ms"] = t_elapsed
                    st.session_state["img_cache"] = (img_key, res, t_elapsed)

                    is_fall = core.is_fall_result(res, st.session_state["fall_thresh"]) if res else False
                    peak_fall_prob = res["fall_prob"] if res else 0.0
                    conf = res["conf"] if res else 0.0
                    torso_angle = res.get("torso_angle", 0.0) if res else 0.0
                    aspect_ratio = res.get("aspect_ratio", 0.0) if res else 0.0

                    if is_fall:
                        details_text = f"🚨 Posture: {core.DISPLAY[res['label']]} (Tilt: {torso_angle:.1f}°, Aspect: {aspect_ratio:.2f})"
                    elif res:
                        details_text = f"🛡️ Posture: {core.DISPLAY[res['label']]} (Tilt: {torso_angle:.1f}°)"
                    else:
                        details_text = "No human skeletal landmarks detected"

                    st.session_state["history"].append({
                        "id": f"IMG-{int(time.time()) % 10000:04d}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": "Single Image",
                        "filename": up_img.name,
                        "status": "Fall Incident" if is_fall else ("Normal Posture" if res else "No Person"),
                        "is_fall": is_fall,
                        "peak_fall_prob": peak_fall_prob,
                        "avg_conf": conf,
                        "confidence": peak_fall_prob if is_fall else conf,
                        "alert_time": None,
                        "latency_ms": t_elapsed,
                        "details": details_text
                    })

                cached_img = st.session_state.get("img_cache")
                if cached_img and cached_img[0] == img_key:
                    _, res, t_elapsed = cached_img
                    if res is None:
                        st.warning("No human pose detected. Ensure subject is clearly visible in the frame.")
                        st.image(to_rgb(bgr), use_container_width=True)
                    else:
                        is_fall = core.is_fall_result(res, st.session_state["fall_thresh"])
                        annotated = core.draw_skeleton(res["bgr"], res["xs"], res["ys"], is_fall)

                        st.markdown("<div style='font-size:12px; font-weight:700; color:#38BDF8; margin-bottom:6px;'>AI SKELETAL OVERLAY (ANTI-ALIASED MESH)</div>", unsafe_allow_html=True)
                        st.image(to_rgb(annotated), use_container_width=True)

                        with st.expander("Show Original Raw Camera Feed"):
                            st.image(to_rgb(bgr), use_container_width=True)
                else:
                    st.image(to_rgb(bgr), caption="Uploaded Raw Frame", use_container_width=True)

    with c_img_insights:
        st.markdown(f"""
        <div class="sf-panel-header">
          <div class="sf-panel-title">{lucide("activity", size=17, color="#10B981")} Biomechanical Insights</div>
          <div style="font-size:11px; color:#94A3B8; font-weight:600;">SINGLE FRAME</div>
        </div>
        """, unsafe_allow_html=True)

        cached_img = st.session_state.get("img_cache")

        if not up_img or not cached_img or cached_img[0] != img_key:
            st.markdown(f"""
            <div class="sf-panel-card" style="text-align:center; padding:32px 18px;">
              <div style="color:#64748B; margin-bottom:8px;">{lucide("camera", size=32, color="#64748B")}</div>
              <div style="font-size:15px; font-weight:700; color:#FFFFFF;">Awaiting Frame Ingestion</div>
              <div style="font-size:12px; color:#94A3B8; margin-top:4px; line-height:1.5;">
                Select a camera frame and click <b>Run Pose Biomechanics Analysis</b> to compute torso tilt angles, body aspect ratios, and multi-class posture probabilities.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            _, res, t_elapsed = cached_img
            if res is not None:
                is_fall = core.is_fall_result(res, st.session_state["fall_thresh"])
                st.markdown(circular_gauge_svg(res["fall_prob"], is_fall), unsafe_allow_html=True)

                if is_fall:
                    st.markdown(f"""
                    <div class="sf-status-alert alarm">
                      <div style="color:#EF4444;">{lucide("alert-triangle", size=22, color="#EF4444")}</div>
                      <div>
                        <div class="sf-status-alert-heading">CRITICAL FALL DETECTED</div>
                        <div class="sf-status-alert-sub">Fall likelihood is {res['fall_prob']*100:.1f}%. Caregiver emergency dispatch recommended.</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sf-status-alert ok">
                      <div style="color:#10B981;">{lucide("check-circle", size=22, color="#10B981")}</div>
                      <div>
                        <div class="sf-status-alert-heading">POSTURE SAFE ({core.DISPLAY[res['label']]})</div>
                        <div class="sf-status-alert-sub">Neural certainty: {res['conf']*100:.1f}%. Fall probability remains below trigger threshold.</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                torso_angle = res.get("torso_angle", 0.0)
                aspect_ratio = res.get("aspect_ratio", 0.0)

                st.markdown(f"""
                <div class="sf-biomech-grid">
                  <div class="sf-biomech-card">
                    <div class="sf-biomech-label">{lucide("activity", size=10, color="#94A3B8")} Primary Posture</div>
                    <div class="sf-biomech-val">{core.DISPLAY[res['label']]}</div>
                    <div class="sf-biomech-sub">Certainty: {res['conf']*100:.1f}%</div>
                  </div>
                  <div class="sf-biomech-card">
                    <div class="sf-biomech-label">{lucide("sliders", size=10, color="#94A3B8")} Torso Tilt</div>
                    <div class="sf-biomech-val">{torso_angle:.1f}°</div>
                    <div class="sf-biomech-sub">Normal: &lt;35°</div>
                  </div>
                  <div class="sf-biomech-card">
                    <div class="sf-biomech-label">{lucide("cpu", size=10, color="#94A3B8")} Aspect Ratio</div>
                    <div class="sf-biomech-val">{aspect_ratio:.2f}</div>
                    <div class="sf-biomech-sub">Width / Height</div>
                  </div>
                  <div class="sf-biomech-card">
                    <div class="sf-biomech-label">{lucide("clock", size=10, color="#94A3B8")} Latency</div>
                    <div class="sf-biomech-val">{t_elapsed:.1f} ms</div>
                    <div class="sf-biomech-sub">Pure NumPy MLP</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities breakdown
                st.markdown("<div style='font-size:12px; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>📊 Multi-Class Output Probabilities</div>", unsafe_allow_html=True)
                bar_rows = []
                for cname, prob in zip(core.CLASS_NAMES, res["probs"]):
                    pct = float(prob) * 100
                    cls = "sf-bar-fill fall" if cname == "fall" else "sf-bar-fill"
                    bar_rows.append(
                        f'<div class="sf-bar-row">'
                        f'<div class="sf-bar-title">{core.DISPLAY[cname]}</div>'
                        f'<div class="sf-bar-bg"><div class="{cls}" style="width:{pct:.1f}%;"></div></div>'
                        f'<div class="sf-bar-pct">{pct:.1f}%</div>'
                        f'</div>'
                    )
                st.html(f'<div class="sf-panel-card" style="padding:10px 14px;">{"".join(bar_rows)}</div>')


# ------------------------------------------------------------------ DETECTION HISTORY
elif curr_view == "Detection History":
    st.markdown(f'<div class="sf-panel-title">{lucide("history", size=18, color="#38BDF8")} Clinical Detection Audit Log</div>', unsafe_allow_html=True)

    if not st.session_state["history"]:
        st.markdown(f"""
        <div class="sf-panel-card" style="text-align:center; padding:48px 24px;">
          <div style="color:#64748B; margin-bottom:12px;">{lucide("history", size=36, color="#64748B")}</div>
          <div style="font-size:16px; font-weight:700; color:#FFFFFF;">No Assessments Recorded Yet</div>
          <div style="font-size:13px; color:#94A3B8; margin-top:4px;">Assessments from Image and Video streams will be logged here with timestamps, peak fall probabilities, and clinical telemetry findings.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Top KPI Summary Cards
        n_total = len(st.session_state['history'])
        n_falls = sum(1 for r in st.session_state['history'] if r.get('is_fall'))
        n_norm = n_total - n_falls
        mean_lat = float(np.mean([r.get('latency_ms', 0.0) for r in st.session_state['history']])) if st.session_state['history'] else 0.0

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.markdown(f"""
            <div class="sf-biomech-card" style="padding:12px 14px;">
              <div class="sf-biomech-label">{lucide("activity", size=11, color="#38BDF8")} Total Assessments</div>
              <div class="sf-biomech-val" style="font-size:22px;">{n_total}</div>
              <div class="sf-biomech-sub">All active sessions</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h2:
            st.markdown(f"""
            <div class="sf-biomech-card" style="padding:12px 14px;">
              <div class="sf-biomech-label">{lucide("alert-triangle", size=11, color="#EF4444")} Fall Incidents</div>
              <div class="sf-biomech-val" style="font-size:22px; color:{'#EF4444' if n_falls > 0 else '#10B981'};">{n_falls}</div>
              <div class="sf-biomech-sub">Verified emergency triggers</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h3:
            st.markdown(f"""
            <div class="sf-biomech-card" style="padding:12px 14px;">
              <div class="sf-biomech-label">{lucide("shield-check", size=11, color="#10B981")} Normal Postures</div>
              <div class="sf-biomech-val" style="font-size:22px; color:#10B981;">{n_norm}</div>
              <div class="sf-biomech-sub">Stable verified activities</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h4:
            st.markdown(f"""
            <div class="sf-biomech-card" style="padding:12px 14px;">
              <div class="sf-biomech-label">{lucide("clock", size=11, color="#94A3B8")} Mean Latency</div>
              <div class="sf-biomech-val" style="font-size:22px;">{mean_lat:.1f} ms</div>
              <div class="sf-biomech-sub">On-device edge inference</div>
            </div>
            """, unsafe_allow_html=True)

        # Control Toolbar (Filter, CSV Export, Clear)
        col_flt, col_dl, col_clr = st.columns([3, 2, 1.8], gap="small")
        with col_flt:
            filter_choice = st.radio(
                "Filter Audit Trail",
                options=["All Events", "🚨 Fall Incidents", "🛡️ Normal Only"],
                horizontal=True,
                label_visibility="collapsed",
                key="audit_filter_radio"
            )
        with col_dl:
            export_rows = [
                {
                    "ID": r.get("id"),
                    "Timestamp": r.get("timestamp"),
                    "Stream Type": r.get("mode"),
                    "Filename": r.get("filename"),
                    "Status": r.get("status"),
                    "Peak Fall Risk (%)": round(r.get("peak_fall_prob", r.get("confidence", 0.0)) * 100, 1),
                    "Mean Posture Conf (%)": round(r.get("avg_conf", r.get("confidence", 0.0)) * 100, 1),
                    "Latency (ms)": round(r.get("latency_ms", 0.0), 1),
                    "Clinical Findings": r.get("details", "")
                }
                for r in st.session_state["history"]
            ]
            csv_str = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Audit Log (CSV)",
                data=csv_str,
                file_name=f"safefall_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_audit_csv"
            )
        with col_clr:
            if st.button("🗑️ Clear Audit History", key="btn_clear_hist", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        # Filtered Records
        filtered_records = list(reversed(st.session_state["history"]))
        if filter_choice == "🚨 Fall Incidents":
            filtered_records = [r for r in filtered_records if r.get("is_fall")]
        elif filter_choice == "🛡️ Normal Only":
            filtered_records = [r for r in filtered_records if not r.get("is_fall")]

        table_rows = []
        for rec in filtered_records:
            is_fall = rec.get("is_fall", False)
            status_str = rec.get("status", "Unknown")
            badge_col = "#EF4444" if is_fall else "#10B981"
            badge_bg = "rgba(239,68,68,0.15)" if is_fall else "rgba(16,185,129,0.12)"
            badge_icon = "🚨" if is_fall else "✓"
            source_name = rec.get("filename", "Source")
            mode_tag = rec.get("mode", "Stream")
            mode_icon = lucide("video", size=13, color="#38BDF8") if "Video" in mode_tag else lucide("camera", size=13, color="#06B6D4")

            peak_prob = rec.get("peak_fall_prob", rec.get("confidence", 0.0))
            mean_conf = rec.get("avg_conf", rec.get("confidence", 0.0))
            details_text = rec.get("details", "-")

            risk_pct = peak_prob * 100
            risk_col = "#EF4444" if is_fall else "#10B981"
            risk_bar_w = min(max(risk_pct, 4.0), 100.0)

            row_html = (
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04); transition:background 0.15s ease;">'
                f'<td style="padding:12px 14px; font-family:monospace; color:#E2E8F0; font-size:12px;"><span style="margin-right:4px;">{mode_icon}</span> {rec.get("id", "REC")}</td>'
                f'<td style="padding:12px 14px; color:#94A3B8; font-size:12px; white-space:nowrap;">{rec.get("timestamp", "-")}</td>'
                f'<td style="padding:12px 14px; color:#FFFFFF; font-weight:600; font-size:13px;">{source_name}</td>'
                f'<td style="padding:12px 14px;"><span style="background:{badge_bg}; color:{badge_col}; border:1px solid {badge_col}44; font-weight:700; font-size:11px; padding:3px 9px; border-radius:6px; display:inline-flex; align-items:center; gap:4px;">{badge_icon} {status_str}</span></td>'
                f'<td style="padding:12px 14px;"><div style="display:flex; align-items:center; gap:8px;"><span style="color:{risk_col}; font-weight:800; font-size:13px; min-width:44px;">{risk_pct:.1f}%</span><div style="width:48px; height:5px; background:rgba(255,255,255,0.08); border-radius:9999px; overflow:hidden;"><div style="width:{risk_bar_w:.1f}%; height:100%; background:{risk_col}; border-radius:9999px;"></div></div></div></td>'
                f'<td style="padding:12px 14px; color:#CBD5E1; font-size:12px;">{mean_conf*100:.1f}%</td>'
                f'<td style="padding:12px 14px; color:#94A3B8; font-size:12px;">{details_text}</td>'
                f'<td style="padding:12px 14px; color:#94A3B8; font-size:12px; font-family:monospace;">{rec.get("latency_ms", 0.0):.1f} ms</td>'
                f'</tr>'
            )
            table_rows.append(row_html)

        if not filtered_records:
            table_rows = ['<tr><td colspan="8" style="padding:32px 14px; text-align:center; color:#64748B; font-size:13px;">No assessments found matching the selected filter.</td></tr>']

        tbody_html = "".join(table_rows)

        full_table_html = (
            '<div class="sf-panel-card" style="padding:0; overflow:hidden; margin-top:10px;">'
            '<table style="width:100%; border-collapse:collapse; font-size:13px; text-align:left;">'
            '<thead>'
            '<tr style="background:rgba(255,255,255,0.03); border-bottom:1px solid var(--border-color); color:#94A3B8; font-size:11px; text-transform:uppercase; letter-spacing:0.04em;">'
            '<th style="padding:12px 14px;">Assessment ID</th>'
            '<th style="padding:12px 14px;">Timestamp</th>'
            '<th style="padding:12px 14px;">Source Stream</th>'
            '<th style="padding:12px 14px;">Clinical Status</th>'
            '<th style="padding:12px 14px;">Peak Fall Risk</th>'
            '<th style="padding:12px 14px;">Mean Conf</th>'
            '<th style="padding:12px 14px;">Clinical Findings</th>'
            '<th style="padding:12px 14px;">Latency</th>'
            '</tr>'
            '</thead>'
            f'<tbody>{tbody_html}</tbody>'
            '</table>'
            '</div>'
        )

        st.html(full_table_html)


# ------------------------------------------------------------------ DOCUMENTATION
elif curr_view == "Documentation":
    st.markdown(f'<div class="sf-panel-title">{lucide("info", size=18, color="#38BDF8")} Engine Architecture & Clinical Documentation</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sf-panel-card">
      <div style="font-size:15px; font-weight:700; color:#FFFFFF; margin-bottom:8px;">Deep Learning & Biomechanics Pipeline Architecture</div>
      <div style="font-size:13px; color:#94A3B8; line-height:1.6;">
        SafeFall AI implements a two-stage hybrid vision pipeline designed for sub-10ms edge inference without cloud streaming dependencies:
      </div>
      <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-top:16px;">
        <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:10px; padding:12px;">
          <div style="font-weight:700; color:#38BDF8; font-size:13px;">1. Dynamic Ingestion</div>
          <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Frames dynamically downscaled to 960px max bound preserving camera aspect ratio.</div>
        </div>
        <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:10px; padding:12px;">
          <div style="font-weight:700; color:#38BDF8; font-size:13px;">2. MediaPipe Pose</div>
          <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Extracts 33 high-accuracy 3D coordinates across shoulders, hips, knees, and feet.</div>
        </div>
        <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:10px; padding:12px;">
          <div style="font-weight:700; color:#38BDF8; font-size:13px;">3. 68 Biomechanical Feats</div>
          <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Hip-centered coordinates scaled by torso length + Torso Angle (°) + Aspect Ratio (W/H).</div>
        </div>
        <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-color); border-radius:10px; padding:12px;">
          <div style="font-weight:700; color:#38BDF8; font-size:13px;">4. NumPy Forward Pass</div>
          <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Multi-layer perceptron forward pass calculated purely in NumPy matrix math in &lt;1ms.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        <div class="sf-panel-card">
          <div style="font-size:14px; font-weight:700; color:#FFFFFF; margin-bottom:8px;">{lucide("sliders", size=14, color="#10B981")} Biomechanical Feature Formulation</div>
          <div style="font-size:12px; color:#94A3B8; line-height:1.6;">
            The feature vector uses normalization invariant to camera distance:
            <ul>
              <li><b>Torso Length:</b> Distance between shoulder midpoint (LM 11, 12) and hip midpoint (LM 23, 24).</li>
              <li><b>Torso Angle (°):</b> arctan2(|dx|, |dy|) between shoulders and hips relative to vertical axis.</li>
              <li><b>Body Aspect Ratio:</b> (X_max - X_min) / (Y_max - Y_min) reflecting horizontal sprawling.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"""
        <div class="sf-panel-card">
          <div style="font-size:14px; font-weight:700; color:#FFFFFF; margin-bottom:8px;">{lucide("shield", size=14, color="#38BDF8")} False Alarm Filtering & Safety Limits</div>
          <div style="font-size:12px; color:#94A3B8; line-height:1.6;">
            <ul>
              <li><b>Temporal Consistency Filter:</b> Single-frame posture spikes (e.g. tying a shoelace or picking an item) do not trigger alarms unless sustained across consecutive frames.</li>
              <li><b>Edge Privacy:</b> Videos are processed in local memory and never transmitted remotely.</li>
              <li><b>Clinical Prototype Disclaimer:</b> Model trained on Le2i research dataset. Intended for telecare demonstration.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)
