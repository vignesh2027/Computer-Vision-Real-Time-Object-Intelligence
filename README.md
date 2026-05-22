# Computer Vision — Real-Time Object Intelligence

> Production-grade Python pipeline for real-time object detection, multi-object tracking, and intelligent analytics — from webcam, video file, or RTSP stream.

---

## Architecture

```
cv_pipeline/
├── detector/
│   └── yolo_detector.py      # YOLOv8 wrapper (swap model = change one file)
├── tracker/
│   └── bytetrack.py          # ByteTrack — persistent IDs across frames
├── analytics/
│   ├── counter.py            # Line crossing + polygon zone counting
│   └── heatmap.py            # Movement heatmap accumulator
├── stream/
│   └── video_source.py       # Webcam / file / RTSP unified source
├── api/
│   └── main.py               # FastAPI: WebSocket live stream + REST /analyze
├── dashboard/
│   └── app.py                # Streamlit live dashboard (warm-white theme)
└── utils/
    └── drawing.py            # Annotation, overlays, stats panel
```

---

## Features

| Feature | Detail |
|---|---|
| **Detection** | YOLOv8 (80 COCO classes), CPU + GPU auto-detect |
| **Tracking** | ByteTrack — unique ID per object across frames |
| **Line Counter** | Virtual line → IN / OUT counts |
| **Zone Analytics** | Draw polygon zones → live object counts per zone |
| **Heatmap** | Movement density heatmap overlaid on video |
| **Speed Estimation** | Pixels/sec → km/h via calibration constant |
| **Alert System** | Print / webhook when object count exceeds threshold |
| **Analytics Dashboard** | FPS, class distribution pie, detection timeline |
| **Recording** | Save annotated video as MP4 |
| **REST API** | `POST /analyze` → JSON detections per frame |
| **WebSocket** | `/ws/stream` → live JPEG frames + metadata |

---

## Quick Start

```bash
git clone https://github.com/vignesh2027/Computer-Vision-Real-Time-Object-Intelligence.git
cd Computer-Vision-Real-Time-Object-Intelligence

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Webcam (downloads yolov8n.pt automatically on first run)
python main.py --source 0

# Video file with heatmap + recording
python main.py --source demo.mp4 --heatmap --record --output result.mp4

# RTSP stream
python main.py --source rtsp://admin:pass@192.168.1.10/stream

# Alerts — print when >5 persons detected
python main.py --source 0 --alert-class person --alert-threshold 5
```

---

## Streamlit Dashboard

```bash
python main.py --dashboard
# or directly:
streamlit run cv_pipeline/dashboard/app.py
```

Open `http://localhost:8501`

- Left sidebar: pick source, model, overlays, recording, alerts
- Main canvas: live annotated video via Streamlit
- Right panel: FPS, object count, line IN/OUT, class pie chart
- Bottom: detection timeline chart

![Dashboard wireframe placeholder](docs/dashboard_demo.gif)

---

## FastAPI Server

```bash
python main.py --api
# or:
uvicorn cv_pipeline.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/stream?source=0&heatmap=true` | Live JPEG stream + metadata JSON |
| `POST` | `/analyze` | Analyze a video URL, returns per-frame detections |
| `POST` | `/alerts/configure` | Set alert class + threshold + webhook URL |
| `GET` | `/stats` | Average FPS and total frames processed |
| `GET` | `/health` | Health check |

**Example `/analyze` request:**
```json
POST /analyze
{
  "video_url": "https://example.com/clip.mp4",
  "max_frames": 200,
  "model": "yolov8s",
  "confidence": 0.45
}
```

---

## Swapping the Detection Model

Only change `cv_pipeline/detector/yolo_detector.py`.  
All other files import from `BaseDetector` — they never touch model internals.

```python
# get_detector factory — change model_name to switch
det = get_detector("yolov8l", confidence=0.5)
```

---

## Docker

```bash
# CPU
docker build -t cv-pipeline .
docker run -p 8000:8000 cv-pipeline

# With webcam
docker run --device /dev/video0 -p 8000:8000 cv-pipeline

# GPU (CUDA) — see Dockerfile comments for base image swap
docker run --gpus all -p 8000:8000 cv-pipeline
```

---

## Tests

```bash
pytest tests/ -v
```

Tests mock the YOLO model — no GPU or downloaded weights required.

---

## Configuration

Copy `.env.example` to `.env` and set values:

```env
YOLO_MODEL=yolov8n
CONFIDENCE=0.4
VIDEO_SOURCE=0
API_PORT=8000
```

---

## Stack

- **Python 3.11+**
- **OpenCV** — video I/O, drawing, heatmap
- **ultralytics (YOLOv8)** — object detection
- **supervision** — ByteTrack multi-object tracker
- **FastAPI + WebSocket** — live streaming API
- **Streamlit + Plotly** — interactive dashboard
- **SQLite** (via analytics module) — detection history
- **pytest** — unit tests
- **Docker** — containerized deployment

---

## License

MIT
