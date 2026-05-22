"""
ByteTrack multi-object tracker wrapping supervision's implementation.
Assigns persistent track IDs across frames.
"""
from __future__ import annotations

from typing import List

import numpy as np
import warnings

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack()

    def update(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        if not detections:
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        # Build class_id -> class_name lookup from input detections
        cls_name_map = {d.class_id: d.class_name for d in detections}

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tracked = self._tracker.update_with_detections(sv_detections)

        result: List[Detection] = []
        for bbox, conf, cls_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            cls_name = cls_name_map.get(int(cls_id), str(cls_id))
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
