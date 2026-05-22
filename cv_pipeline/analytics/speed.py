"""
Speed estimator: pixel displacement per frame -> km/h.
Requires a calibration value: how many pixels = 1 metre at the camera's reference plane.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from cv_pipeline.detector.yolo_detector import Detection


class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 50.0, fps: float = 30.0):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self._prev_centroids: Dict[int, np.ndarray] = {}
        self._speeds: Dict[int, float] = {}  # track_id -> km/h

    def update(self, detections: list) -> Dict[int, float]:
        current: Dict[int, np.ndarray] = {}
        for det in detections:
            if det.track_id is None:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            current[det.track_id] = np.array([cx, cy])

        for tid, pos in current.items():
            if tid in self._prev_centroids:
                pixel_dist = float(np.linalg.norm(pos - self._prev_centroids[tid]))
                meters_per_sec = (pixel_dist / self.pixels_per_meter) * self.fps
                self._speeds[tid] = meters_per_sec * 3.6  # -> km/h
            else:
                self._speeds[tid] = 0.0

        self._prev_centroids = current
        return {tid: self._speeds.get(tid, 0.0) for tid in current}

    def get_speed(self, track_id: int) -> float:
        return self._speeds.get(track_id, 0.0)

    def reset(self):
        self._prev_centroids.clear()
        self._speeds.clear()
