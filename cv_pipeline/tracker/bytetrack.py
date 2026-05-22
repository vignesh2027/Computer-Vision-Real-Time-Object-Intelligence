"""
ByteTrack multi-object tracker wrapping supervision's implementation.
Assigns persistent track IDs across frames.
"""
from __future__ import annotations

from typing import List

import numpy as np
import supervision as sv

from cv_pipeline.detector.yolo_detector import Detection


class ObjectTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self._tracker = sv.ByteTracker(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        if not detections:
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )

        tracked = self._tracker.update_with_detections(sv_detections)

        result: List[Detection] = []
        for i, (bbox, conf, cls_id, track_id) in enumerate(
            zip(tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id)
        ):
            cls_name = detections[0].class_name if i >= len(detections) else detections[i].class_name
            for d in detections:
                if d.class_id == cls_id:
                    cls_name = d.class_name
                    break
            result.append(
                Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=cls_name,
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return result

    def reset(self):
        self._tracker.reset()
