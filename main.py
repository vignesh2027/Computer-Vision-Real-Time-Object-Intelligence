"""
Entry point: run the full CV pipeline from the command line.

Usage:
  python main.py --source 0                        # webcam
  python main.py --source video.mp4 --record
  python main.py --source rtsp://... --heatmap
  python main.py --api                             # FastAPI server only
  python main.py --dashboard                       # Streamlit dashboard only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Dict

import cv2

from cv_pipeline.analytics.counter import LineCounter, ZoneCounter
from cv_pipeline.analytics.heatmap import HeatmapGenerator
from cv_pipeline.detector.yolo_detector import get_detector
from cv_pipeline.stream.video_source import VideoSource
from cv_pipeline.tracker.bytetrack import ObjectTracker
from cv_pipeline.utils.drawing import Annotator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV Pipeline — Real-Time Object Intelligence")
    p.add_argument("--source", default="0", help="Webcam index, file path, or RTSP URL")
    p.add_argument("--model", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--api", action="store_true", help="Start FastAPI server")
    p.add_argument("--dashboard", action="store_true", help="Start Streamlit dashboard")
    p.add_argument("--alert-class", default="person")
    p.add_argument("--alert-threshold", type=int, default=10)
    return p.parse_args()


def run_pipeline(args: argparse.Namespace):
    detector = get_detector(model_name=args.model, confidence=args.confidence)
    tracker = ObjectTracker()
    annotator = Annotator()

    video_src = VideoSource.from_string(args.source)
    video_src.open()
    w, h = video_src.frame_size
    fps = video_src.fps

    heatmap_gen = HeatmapGenerator((h, w))
    line_counter = LineCounter((0, h // 2), (w, h // 2))

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Recording to {args.output}")

    fps_hist: deque = deque(maxlen=30)
    print("Press Q to quit.")

    try:
        for frame in video_src.frames():
            t0 = time.monotonic()
            detections = detector.detect(frame)
            tracked = tracker.update(detections, frame.shape)

            heatmap_gen.update(tracked)
            in_c, out_c = line_counter.update(tracked)

            annotated = frame.copy()
            if args.heatmap:
                annotated = heatmap_gen.overlay(annotated)
            annotated = annotator.draw_detections(annotated, tracked)
            annotated = annotator.draw_line(annotated, (0, h // 2), (w, h // 2), in_c, out_c)

            elapsed = time.monotonic() - t0
            cur_fps = 1.0 / max(elapsed, 1e-6)
            fps_hist.append(cur_fps)
            avg_fps = sum(fps_hist) / len(fps_hist)

            cls_cnt: Dict[str, int] = defaultdict(int)
            for d in tracked:
                cls_cnt[d.class_name] += 1

            annotated = annotator.draw_stats(annotated, avg_fps, len(tracked), cls_cnt)

            if args.alert_class and cls_cnt.get(args.alert_class, 0) >= args.alert_threshold:
                print(f"[ALERT] {cls_cnt[args.alert_class]} '{args.alert_class}' detected!")

            if writer:
                writer.write(annotated)

            cv2.imshow("CV Pipeline — Object Intelligence", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_src.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.api:
        subprocess.run(
            ["uvicorn", "cv_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            check=True,
        )
        return

    if args.dashboard:
        subprocess.run(
            ["streamlit", "run", "cv_pipeline/dashboard/app.py"],
            check=True,
        )
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
