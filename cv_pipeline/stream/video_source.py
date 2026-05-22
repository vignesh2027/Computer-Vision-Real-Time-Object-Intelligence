"""
Unified video source: webcam index, local file path, or RTSP URL.
"""
from __future__ import annotations

import time
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource:
    def __init__(
        self,
        source: Union[int, str] = 0,
        target_fps: Optional[float] = None,
        buffer_size: int = 1,
    ):
        self.source = source
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self.buffer_size = buffer_size

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def close(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 640, 480

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            self.open()
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self._cap is None:
            self.open()
        interval = 1.0 / self.target_fps if self.target_fps else None
        try:
            while True:
                t0 = time.monotonic()
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield frame
                if interval:
                    elapsed = time.monotonic() - t0
                    wait = interval - elapsed
                    if wait > 0:
                        time.sleep(wait)
        finally:
            self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def from_string(source: str) -> "VideoSource":
        if source.isdigit():
            return VideoSource(int(source))
        return VideoSource(source)
