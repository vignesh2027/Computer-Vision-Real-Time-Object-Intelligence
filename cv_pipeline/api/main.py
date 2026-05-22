"""
FastAPI server: WebSocket live stream + REST /analyze endpoint.
Run: uvicorn cv_pipeline.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cv_pipeline.analytics.counter import LineCounter, ZoneCounter
from cv_pipeline.analytics.heatmap import HeatmapGenerator
from cv_pipeline.detector.yolo_detector import Detection, get_detector
from cv_pipeline.stream.video_source import VideoSource
from cv_pipeline.tracker.bytetrack import ObjectTracker
from cv_pipeline.utils.drawing import Annotator

load_dotenv()

app = FastAPI(
    title="CV Pipeline API",
    description="Real-Time Object Intelligence — detection, tracking, analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Shared state ---
_detector = None
_tracker = ObjectTracker()
_annotator = Annotator()
_fps_history: deque = deque(maxlen=60)
_class_counts: Dict[str, int] = defaultdict(int)
_total_frames = 0
_alert_threshold: Dict[str, int] = {}
_alert_webhooks: List[str] = []


def _get_detector():
    global _detector
    if _detector is None:
        model = os.getenv("YOLO_MODEL", "yolov8n")
        conf = float(os.getenv("CONFIDENCE", "0.4"))
        _detector = get_detector(model_name=model, confidence=conf)
    return _detector


# --- WebSocket live stream ---

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    source_str = ws.query_params.get("source", "0")
    show_heatmap = ws.query_params.get("heatmap", "false").lower() == "true"
    show_zones = ws.query_params.get("zones", "false").lower() == "true"

    source = VideoSource.from_string(source_str)
    source.open()
    w, h = source.frame_size
    heatmap_gen = HeatmapGenerator((h, w))
    line_counter = LineCounter((0, h // 2), (w, h // 2))
    zone_counter = ZoneCounter()

    try:
        for frame in source.frames():
            t0 = time.monotonic()
            det = _get_detector()
            detections = det.detect(frame)
            tracked = _tracker.update(detections, frame.shape)

            heatmap_gen.update(tracked)
            in_c, out_c = line_counter.update(tracked)

            annotated = frame.copy()
            if show_heatmap:
                annotated = heatmap_gen.overlay(annotated)
            annotated = _annotator.draw_detections(annotated, tracked)
            annotated = _annotator.draw_line(
                annotated,
                line_counter.start.astype(int).tolist(),
                line_counter.end.astype(int).tolist(),
                in_c,
                out_c,
            )

            fps = 1.0 / max(time.monotonic() - t0, 1e-6)
            _fps_history.append(fps)
            cls_cnt: Dict[str, int] = defaultdict(int)
            for d in tracked:
                cls_cnt[d.class_name] += 1
            annotated = _annotator.draw_stats(annotated, fps, len(tracked), cls_cnt)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode()

            payload = {
                "frame": b64,
                "fps": round(fps, 1),
                "detections": len(tracked),
                "in_count": in_c,
                "out_count": out_c,
                "class_counts": dict(cls_cnt),
            }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        source.close()


# --- REST endpoints ---

class AnalyzeRequest(BaseModel):
    video_url: str
    max_frames: int = 100
    model: str = "yolov8n"
    confidence: float = 0.4


class AlertConfig(BaseModel):
    class_name: str
    threshold: int
    webhook_url: Optional[str] = None


@app.post("/analyze")
async def analyze_video(req: AnalyzeRequest) -> JSONResponse:
    detector = get_detector(model_name=req.model, confidence=req.confidence)
    cap = cv2.VideoCapture(req.video_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video URL")

    results: List[Dict[str, Any]] = []
    frame_idx = 0
    try:
        while frame_idx < req.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            results.append({
                "frame": frame_idx,
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "bbox": d.bbox.tolist(),
                    }
                    for d in detections
                ],
            })
            frame_idx += 1
    finally:
        cap.release()

    return JSONResponse({"frames_analyzed": frame_idx, "results": results})


@app.post("/alerts/configure")
async def configure_alert(cfg: AlertConfig):
    _alert_threshold[cfg.class_name] = cfg.threshold
    if cfg.webhook_url:
        _alert_webhooks.append(cfg.webhook_url)
    return {"status": "configured", "class": cfg.class_name, "threshold": cfg.threshold}


@app.get("/stats")
async def get_stats():
    avg_fps = sum(_fps_history) / len(_fps_history) if _fps_history else 0.0
    return {
        "avg_fps": round(avg_fps, 2),
        "total_frames": _total_frames,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
