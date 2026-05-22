"""
Accumulates object centroid positions into a heatmap overlay.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from cv_pipeline.detector.yolo_detector import Detection


class HeatmapGenerator:
    def __init__(
        self,
        frame_shape: Tuple[int, int],
        decay: float = 0.98,
        blur_radius: int = 21,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ):
        h, w = frame_shape[:2]
        self._accumulator = np.zeros((h, w), dtype=np.float32)
        self.decay = decay
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.colormap = colormap
        self.alpha = alpha

    def update(self, detections: List[Detection]):
        self._accumulator *= self.decay
        for det in detections:
            cx = int((det.bbox[0] + det.bbox[2]) / 2)
            cy = int((det.bbox[1] + det.bbox[3]) / 2)
            h, w = self._accumulator.shape
            if 0 <= cx < w and 0 <= cy < h:
                self._accumulator[cy, cx] += 1.0

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(self._accumulator, (self.blur_radius, self.blur_radius), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, self.colormap)
        mask = norm_u8 > 5
        out = frame.copy()
        out[mask] = cv2.addWeighted(frame, 1 - self.alpha, colored, self.alpha, 0)[mask]
        return out

    def reset(self):
        self._accumulator[:] = 0
