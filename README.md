<div align="center">

# Computer Vision — Real-Time Object Intelligence

**Production-grade Python pipeline for real-time object detection, multi-object tracking, and intelligent analytics**

[![Python](https://img.shields.io/badge/Python-3.11+-1A1A1A?style=flat&logo=python&logoColor=FAF9F6)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-D4A574?style=flat)](https://docs.ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-1A1A1A?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-D4A574?style=flat&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-1A1A1A?style=flat)](LICENSE)

*Webcam · Video File · RTSP Stream — CPU or GPU, zero config*

</div>

---

## What it does

Feed it a webcam, video file, or RTSP stream and get back:

- **Detected objects** — 80 COCO classes, live bounding boxes with confidence scores
- **Tracked objects** — persistent IDs that follow each object across frames (ByteTrack)
- **Counted crossings** — virtual line with IN / OUT counters
- **Zone occupancy** — how many objects are inside each polygon zone at any moment
- **Movement heatmap** — density overlay showing where objects spend the most time
- **Speed estimates** — pixel velocity mapped to km/h via calibration
- **Live alerts** — fires when any class exceeds a count threshold
- **Annotated recording** — saves the full annotated stream as MP4
- **REST API** — `POST /analyze` returns JSON detections per frame for any video URL
- **WebSocket stream** — `ws://localhost:8000/ws/stream` pushes live JPEG frames + metadata

---

## Architecture

```
cv_pipeline/
├── detector/
│   └── yolo_detector.py      ← YOLOv8 wrapper (swap model = change one file)
├── tracker/
│   └── bytetrack.py          ← ByteTrack, persistent IDs across frames
├── analytics/
│   ├── counter.py            ← Virtual line crossing + polygon zone counting
│   └── heatmap.py            ← Temporal movement density accumulator
├── stream/
│   └── video_source.py       ← Webcam / file / RTSP unified interface
├── api/
│   └── main.py               ← FastAPI: WebSocket live stream + REST endpoints
├── dashboard/
│   └── app.py                ← Streamlit dashboard (warm-white theme)
└── utils/
    └── drawing.py            ← Annotation overlays and stats panel
```

> **One-file model swap:** every other module depends on `BaseDetector`, never on YOLO internals. Replacing YOLOv8 with any other model only requires changing `yolo_detector.py`.

---

## Quick Start

```bash
git clone https://github.com/vignesh2027/Computer-Vision-Real-Time-Object-Intelligence.git
cd Computer-Vision-Real-Time-Object-Intelligence

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

YOLOv8n weights download automatically on the first run (~6 MB).

```bash
# Webcam
python main.py --source 0

# Local video file + heatmap + save annotated output
python main.py --source clip.mp4 --heatmap --record --output result.mp4

# RTSP stream
python main.py --source rtsp://admin:pass@192.168.1.10/stream

# Alert when more than 5 persons are in frame
python main.py --source 0 --alert-class person --alert-threshold 5

# Heavier model for better accuracy (still works on CPU)
python main.py --source 0 --model yolov8s --confidence 0.45
```

---

## Streamlit Dashboard

```bash
streamlit run cv_pipeline/dashboard/app.py
```

Open `http://localhost:8501`

**Dashboard layout:**

```
┌─────────────────────────────┬──────────────────┐
│  Sidebar                    │  Stats panel     │
│  ─ Source: webcam/file/RTSP │  FPS             │
│  ─ Model: yolov8n … x       │  Objects         │
│  ─ Overlays: heatmap/line/  │  Line IN / OUT   │
│    zone                     │  Class pie chart │
│  ─ Recording                │                  │
│  ─ Alerts                   │                  │
├─────────────────────────────┴──────────────────┤
│  Live annotated video feed (WebSocket)         │
│  Alert banner (amber, when threshold hit)      │
├────────────────────────────────────────────────┤
│  Detection count timeline chart                │
└────────────────────────────────────────────────┘
```

---

## FastAPI Server

```bash
# Start the API
python main.py --api
# or: uvicorn cv_pipeline.api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Endpoint | Description |
|---|---|---|
| `WS` | `/ws/stream?source=0&heatmap=true` | Live JPEG frames + JSON metadata |
| `POST` | `/analyze` | Analyze a video URL, returns per-frame detections |
| `POST` | `/alerts/configure` | Set class threshold + optional webhook |
| `GET` | `/stats` | Average FPS, total frames processed |
| `GET` | `/health` | Health check |

**Analyze request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/traffic.mp4",
    "max_frames": 300,
    "model": "yolov8s",
    "confidence": 0.45
  }'
```

**Response:**
```json
{
  "frames_analyzed": 300,
  "results": [
    {
      "frame": 0,
      "detections": [
        { "class": "car", "confidence": 0.912, "bbox": [120, 80, 340, 210] },
        { "class": "person", "confidence": 0.876, "bbox": [420, 100, 480, 290] }
      ]
    }
  ]
}
```

---

## Docker

```bash
# Build and run (CPU)
docker build -t cv-pipeline .
docker run -p 8000:8000 cv-pipeline

# With webcam passthrough
docker run --device /dev/video0 -p 8000:8000 cv-pipeline

# GPU (CUDA) — swap the FROM line in Dockerfile to nvidia/cuda base
docker run --gpus all -p 8000:8000 cv-pipeline

# Run Streamlit instead
docker run -p 8501:8501 cv-pipeline \
  streamlit run cv_pipeline/dashboard/app.py --server.port 8501
```

---

## Configuration

Copy `.env.example` to `.env`:

```env
YOLO_MODEL=yolov8n          # n · s · m · l · x
CONFIDENCE=0.4
IOU_THRESHOLD=0.45
VIDEO_SOURCE=0              # 0 = webcam, or file/RTSP path
API_HOST=0.0.0.0
API_PORT=8000
ALERT_WEBHOOK_URL=          # optional POST target
PIXELS_PER_METER=50         # speed calibration
```

---

## Tests

```bash
pytest tests/ -v
```

Tests mock the YOLO model — no GPU or downloaded weights required. Covers detector output shapes, confidence values, line crossing logic, zone containment, and edge cases.

---

## Model Options

| Model | Size | Speed (CPU) | Accuracy |
|---|---|---|---|
| `yolov8n` | 6 MB | ~30 FPS | Good — default |
| `yolov8s` | 22 MB | ~18 FPS | Better |
| `yolov8m` | 52 MB | ~10 FPS | Strong |
| `yolov8l` | 87 MB | ~6 FPS | Very strong |
| `yolov8x` | 131 MB | ~4 FPS | Best |

GPU speeds are 5–10× higher. Swap via `--model yolov8s` or `YOLO_MODEL=yolov8s` in `.env`.

---

## Stack

| Layer | Technology |
|---|---|
| Detection | [ultralytics YOLOv8](https://docs.ultralytics.com) |
| Tracking | [supervision ByteTracker](https://supervision.roboflow.com) |
| Video I/O | [OpenCV](https://opencv.org) |
| API | [FastAPI](https://fastapi.tiangolo.com) + WebSocket |
| Dashboard | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com) |
| Deep learning | [PyTorch](https://pytorch.org) (CPU + CUDA auto-detect) |
| Tests | [pytest](https://pytest.org) |
| Container | [Docker](https://docker.com) |

---

## License

MIT — free to use, modify, and distribute.
