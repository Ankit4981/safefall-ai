# 🛡️ SafeFall AI: Clinical Fall Detection & Pose Biomechanics Dashboard

An advanced, edge-native computer vision system designed for real-time elderly fall detection, posture classification, and temporal biomechanics telemetry.

---

## 🌟 Key Features

- **⚡ Sub-10ms Edge Inference**: Powered by a 68-feature NumPy Multi-Layer Perceptron (MLP) running entirely on local CPU/RAM with zero cloud streaming dependencies.
- **🦴 33-Point MediaPipe Pose Extraction**: Full-body 3D coordinate estimation tracking shoulders, torso tilt, hips, knees, and ankles.
- **🎥 Temporal Stream Analysis**: Evaluates continuous video streams (`.mp4`, `.avi`, `.mov`, `.mkv`), computing posture progressions, temporal risk timelines, and consecutive frame fall triggers.
- **📸 Single-Frame Biomechanics**: Instant analysis of still images providing torso tilt angles (°), body sprawl aspect ratios, and multi-class posture probabilities.
- **📊 EHR-Grade Audit Log**: Immutable assessment logs with peak fall probability meters, exact incident timestamps, status filtering, and one-click CSV export (`safefall_audit_log.csv`).
- **🎨 Cyber-Medical UI**: Production-grade clinical dashboard designed with deep navy hues, Lucide vector icons, and an SVG Fall Risk Gauge.

---

## 🏗️ Architecture & Pipeline

```text
Raw Ingestion (960px Aspect Preserved)
               │
               ▼
   MediaPipe Pose Landmarker (33 LMs)
               │
               ▼
   68 Biomechanical Features Normalized
   (Torso Length, Torso Tilt °, Sprawl Aspect)
               │
               ▼
   NumPy MLP Classifier (<1ms Forward Pass)
               │
               ▼
   Temporal Consistency & Consecutive Frame Filter
               │
               ▼
   Clinical Alert Dispatch & Telecare Audit Log
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository
```bash
git clone https://github.com/Ankit4981/safefall-ai.git
cd safefall-ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Core Dependencies
- [Streamlit](https://streamlit.io/) - Clinical dashboard framework
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) - 33-point skeletal landmark detection
- [OpenCV](https://opencv.org/) - Video decoding and anti-aliased skeletal drawing
- [NumPy](https://numpy.org/) - Fast matrix math & forward MLP propagation
- [Pandas](https://pandas.pydata.org/) - Telemetry time-series & audit log export

---

## 🔒 Privacy & Edge Deployment
SafeFall AI operates **100% on-device**. Camera frames and video feeds are processed in temporary RAM buffers and never transmitted to external cloud servers, safeguarding patient privacy in eldercare and hospital environments.
