"""
Swap any detection model by subclassing BaseDetector and updating this file only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    @property
    def class_names(self) -> List[str]:
        raise NotImplementedError


class YOLODetector(BaseDetector):
    """
    YOLOv8 wrapper. To swap models, subclass BaseDetector and return a different
    implementation from a factory — no other file needs changing.
    """

    MODEL_DIR = Path.home() / ".cv_pipeline" / "models"
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or str(self.MODEL_DIR / self.DEFAULT_MODEL)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device or self._auto_device()
        self._model = None

    def _auto_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self):
        if self._model is not None:
            return
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._ensure_model()
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return detections

    @property
    def class_names(self) -> List[str]:
        self._ensure_model()
        return list(self._model.names.values())


def get_detector(
    model_name: str = "yolov8n",
    **kwargs,
) -> BaseDetector:
    """Factory — change model_name to switch backends without touching other files."""
    model_map = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
    }
    model_file = model_map.get(model_name, model_name)
    model_path = str(YOLODetector.MODEL_DIR / model_file)
    return YOLODetector(model_path=model_path, **kwargs)
