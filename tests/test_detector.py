"""Tests for YOLODetector — mock the model to avoid downloading weights."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cv_pipeline.detector.yolo_detector import Detection, YOLODetector, get_detector


def _tensor_mock(value):
    """Return a MagicMock that behaves like a PyTorch tensor: .cpu().numpy() -> value."""
    t = MagicMock()
    t.cpu.return_value.numpy.return_value = value
    return t


def _make_mock_model(class_id: int = 0, class_name: str = "person", conf: float = 0.85):
    box = MagicMock()
    box.xyxy = [_tensor_mock(np.array([10.0, 20.0, 100.0, 200.0]))]
    box.conf = [_tensor_mock(np.array(conf))]
    box.cls  = [_tensor_mock(np.array(class_id))]

    result = MagicMock()
    result.boxes = [box]
    result.names = {class_id: class_name}

    model = MagicMock()
    model.predict.return_value = [result]
    model.names = {class_id: class_name}
    return model


class TestYOLODetector:
    def test_detect_returns_detections(self):
        detector = YOLODetector(model_path="yolov8n.pt", confidence=0.4)
        detector._model = _make_mock_model()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        d = detections[0]
        assert d.class_name == "person"
        assert d.confidence == pytest.approx(0.85)
        assert d.bbox.shape == (4,)

    def test_detect_empty_when_no_boxes(self):
        detector = YOLODetector()
        result = MagicMock()
        result.boxes = None
        result.names = {}
        detector._model = MagicMock()
        detector._model.predict.return_value = [result]
        detector._model.names = {}

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_class_names_property(self):
        detector = YOLODetector()
        detector._model = _make_mock_model(class_id=0, class_name="person")
        assert "person" in detector.class_names

    def test_get_detector_factory(self):
        # YOLO is imported lazily inside _ensure_model, so patch at the source
        with patch("ultralytics.YOLO", return_value=_make_mock_model()):
            det = get_detector("yolov8n", confidence=0.5)
            assert isinstance(det, YOLODetector)
            assert det.confidence == 0.5

    def test_auto_device_cpu_fallback(self):
        detector = YOLODetector()
        assert detector.device in ("cpu", "cuda", "mps")
