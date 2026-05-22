"""
Annotation utilities — bounding boxes, labels, lines, polygons, stats overlay.
"""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cv_pipeline.detector.yolo_detector import Detection

# Warm-white palette constants
WARM_WHITE = (245, 245, 240)
DARK_CHARCOAL = (30, 30, 30)
ACCENT_AMBER = (30, 165, 255)   # BGR: amber/orange
ACCENT_GREEN = (80, 200, 80)
ACCENT_RED = (60, 60, 220)


def _class_color(class_id: int) -> Tuple[int, int, int]:
    """Deterministic per-class color (warm palette, not neon)."""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


class Annotator:
    def __init__(self, thickness: int = 2, font_scale: float = 0.55):
        self.thickness = thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            color = _class_color(det.class_id)
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.thickness)
            label = det.class_name
            if det.track_id is not None:
                label = f"#{det.track_id} {label}"
            label += f" {det.confidence:.2f}"
            self._put_label(out, label, (x1, y1), color)
        return out

    def _put_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        x, y = origin
        (tw, th), baseline = cv2.getTextSize(text, self.font, self.font_scale, 1)
        pad = 3
        bg_y1 = max(0, y - th - pad * 2)
        bg_y2 = y
        bg_x2 = x + tw + pad * 2
        cv2.rectangle(frame, (x, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x + pad, y - pad),
            self.font,
            self.font_scale,
            WARM_WHITE,
            1,
            cv2.LINE_AA,
        )

    def draw_line(
        self,
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        in_count: int,
        out_count: int,
    ) -> np.ndarray:
        out = frame.copy()
        cv2.line(out, start, end, ACCENT_AMBER, 2, cv2.LINE_AA)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(out, f"IN:{in_count}  OUT:{out_count}", mid, self.font, 0.7, ACCENT_AMBER, 2, cv2.LINE_AA)
        return out

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: Dict[str, np.ndarray],
        counts: Dict[str, int],
    ) -> np.ndarray:
        out = frame.copy()
        for i, (name, poly) in enumerate(zones.items()):
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            color = _class_color(i + 10)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
            cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            label = f"{name}: {counts.get(name, 0)}"
            cv2.putText(out, label, (cx - 30, cy), self.font, 0.65, WARM_WHITE, 2, cv2.LINE_AA)
        return out

    def draw_speed(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        speeds: Dict[int, float],
    ) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            if det.track_id is None:
                continue
            kmh = speeds.get(det.track_id, 0.0)
            if kmh < 0.5:
                continue
            x1, y1 = int(det.bbox[0]), int(det.bbox[3])
            cv2.putText(out, f"{kmh:.1f} km/h", (x1, y1 + 16), self.font, 0.5, ACCENT_GREEN, 1, cv2.LINE_AA)
        return out

    def draw_stats(
        self,
        frame: np.ndarray,
        fps: float,
        total_detections: int,
        class_counts: Dict[str, int],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        panel_w = 220
        overlay = out.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, min(h, 30 + 22 * (len(class_counts) + 2))), DARK_CHARCOAL, -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        y = 20
        cv2.putText(out, f"FPS: {fps:.1f}", (w - panel_w + 8, y), self.font, 0.55, ACCENT_AMBER, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(out, f"Objects: {total_detections}", (w - panel_w + 8, y), self.font, 0.55, WARM_WHITE, 1, cv2.LINE_AA)
        y += 22
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:8]:
            cv2.putText(out, f"  {cls}: {cnt}", (w - panel_w + 8, y), self.font, 0.48, WARM_WHITE, 1, cv2.LINE_AA)
            y += 18
        return out
