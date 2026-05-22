"""Tests for LineCounter and ZoneCounter."""
from __future__ import annotations

import numpy as np
import pytest

from cv_pipeline.analytics.counter import LineCounter, ZoneCounter
from cv_pipeline.detector.yolo_detector import Detection


def _det(x1, y1, x2, y2, track_id=1, cls="person") -> Detection:
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=float),
        confidence=0.9,
        class_id=0,
        class_name=cls,
        track_id=track_id,
    )


class TestLineCounter:
    def test_initial_counts_zero(self):
        lc = LineCounter((0, 240), (640, 240))
        assert lc.in_count == 0
        assert lc.out_count == 0

    def test_crossing_increments_count(self):
        lc = LineCounter((0, 240), (640, 240))
        # Object above the line (y < 240)
        lc.update([_det(100, 100, 200, 200, track_id=1)])  # centroid y=150
        # Object below the line (y > 240)
        lc.update([_det(100, 300, 200, 400, track_id=1)])  # centroid y=350
        assert lc.in_count + lc.out_count == 1

    def test_no_crossing_no_count(self):
        lc = LineCounter((0, 240), (640, 240))
        lc.update([_det(100, 100, 200, 200, track_id=1)])
        lc.update([_det(100, 120, 200, 220, track_id=1)])  # still above
        assert lc.in_count == 0
        assert lc.out_count == 0

    def test_reset(self):
        lc = LineCounter((0, 240), (640, 240))
        lc.update([_det(100, 100, 200, 200, track_id=1)])
        lc.update([_det(100, 300, 200, 400, track_id=1)])
        lc.reset()
        assert lc.in_count == 0
        assert lc.out_count == 0

    def test_multiple_objects(self):
        lc = LineCounter((0, 240), (640, 240))
        lc.update([
            _det(100, 100, 200, 200, track_id=1),
            _det(300, 100, 400, 200, track_id=2),
        ])
        lc.update([
            _det(100, 300, 200, 400, track_id=1),
            _det(300, 300, 400, 400, track_id=2),
        ])
        assert lc.in_count + lc.out_count == 2


class TestZoneCounter:
    def _square_zone(self):
        return [(100, 100), (300, 100), (300, 300), (100, 300)]

    def test_object_inside_zone(self):
        zc = ZoneCounter({"zone_a": self._square_zone()})
        counts = zc.count([_det(150, 150, 250, 250, track_id=1)])
        assert counts["zone_a"] == 1

    def test_object_outside_zone(self):
        zc = ZoneCounter({"zone_a": self._square_zone()})
        counts = zc.count([_det(400, 400, 500, 500, track_id=1)])
        assert counts["zone_a"] == 0

    def test_multiple_zones(self):
        zc = ZoneCounter({
            "zone_a": self._square_zone(),
            "zone_b": [(400, 400), (600, 400), (600, 600), (400, 600)],
        })
        counts = zc.count([
            _det(150, 150, 250, 250, track_id=1),
            _det(450, 450, 550, 550, track_id=2),
        ])
        assert counts["zone_a"] == 1
        assert counts["zone_b"] == 1

    def test_add_remove_zone(self):
        zc = ZoneCounter()
        zc.add_zone("zone_a", self._square_zone())
        assert "zone_a" in zc.zones
        zc.remove_zone("zone_a")
        assert "zone_a" not in zc.zones

    def test_empty_detections(self):
        zc = ZoneCounter({"zone_a": self._square_zone()})
        counts = zc.count([])
        assert counts["zone_a"] == 0
