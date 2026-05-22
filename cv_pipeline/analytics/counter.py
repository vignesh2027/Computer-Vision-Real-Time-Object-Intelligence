"""
Line crossing counter and polygon zone counter.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cv_pipeline.detector.yolo_detector import Detection


class LineCounter:
    """
    Counts objects crossing a virtual line.
    The line is defined by two endpoints (x1,y1) -> (x2,y2).
    Direction is determined by which side of the line the object centroid moves from/to.
    """

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.in_count: int = 0
        self.out_count: int = 0
        self._prev_side: Dict[int, int] = {}  # track_id -> side

    def _side(self, point: np.ndarray) -> int:
        """Return +1 or -1 for which side of the line the point is on."""
        d = self.end - self.start
        v = point - self.start
        cross = d[0] * v[1] - d[1] * v[0]
        return 1 if cross >= 0 else -1

    def update(self, detections: List[Detection]) -> Tuple[int, int]:
        for det in detections:
            if det.track_id is None:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            current_side = self._side(np.array([cx, cy]))
            prev = self._prev_side.get(det.track_id)
            if prev is not None and prev != current_side:
                if current_side == 1:
                    self.in_count += 1
                else:
                    self.out_count += 1
            self._prev_side[det.track_id] = current_side
        return self.in_count, self.out_count

    def reset(self):
        self.in_count = 0
        self.out_count = 0
        self._prev_side.clear()


class ZoneCounter:
    """
    Counts objects inside user-defined polygon zones.
    zones: dict of zone_name -> list of (x, y) vertices
    """

    def __init__(self, zones: Optional[Dict[str, List[Tuple[int, int]]]] = None):
        self.zones: Dict[str, np.ndarray] = {}
        if zones:
            for name, pts in zones.items():
                self.add_zone(name, pts)

    def add_zone(self, name: str, points: List[Tuple[int, int]]):
        self.zones[name] = np.array(points, dtype=np.float32)

    def remove_zone(self, name: str):
        self.zones.pop(name, None)

    def count(self, detections: List[Detection]) -> Dict[str, int]:
        counts = {name: 0 for name in self.zones}
        for det in detections:
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            for name, poly in self.zones.items():
                if self._point_in_polygon(cx, cy, poly):
                    counts[name] += 1
        return counts

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        import cv2
        result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
        return result >= 0
