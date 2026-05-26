#!/usr/bin/env python3
"""
face_detector.py — Multi-camera face detection service
--------------------------------------------------------
Reads cameras from the Node.js /api/cameras endpoint, spawns one
RTSPReader + FaceNPU worker per camera, and POSTs face-event JSON
(including a base64-encoded face crop) to /api/face-events.

Usage:
  python3 face_detector.py \
      --library libnn_yolo11n-face.so \
      --model   yolo11n-face.nb \
      --server  http://localhost:3000 \
      [--interval 2.0]   # min seconds between events per camera
      [--tiles 1]
      [--face-thresh 0.50]

The NPU model is shared across all cameras via a single FaceNPU instance
and per-camera InferenceWorker threads.  Each worker reads its stream
independently; events are serialised through a single HTTP poster thread.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time
from typing import Dict, Optional, Tuple

# ── env tweaks (same as cap.py) ───────────────────────────────────────────────
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed

import cv2 as cv
import numpy as np
import urllib.request

# ── re-use all detection logic from cap.py ────────────────────────────────────
# (imported as a module; cap.py must be in the same directory)
sys.path.insert(0, os.path.dirname(__file__))
from cap import (
    FaceNPU, InferenceWorker, RTSPReader,
    _cfg, letterbox, unletterbox_boxes,
    yolo_post_process, filter_face_geometry, reshape_heads,
    DETECT_SIZE,
)


# ══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════════════════════════════════════════════

def fetch_cameras(server: str) -> list:
    url = f"{server.rstrip('/')}/api/cameras"
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read())


def post_event(server: str, payload: dict) -> None:
    url = f"{server.rstrip('/')}/api/face-events"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            pass
    except Exception as e:
        print(f"[poster] POST failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Crop helper
# ══════════════════════════════════════════════════════════════════════════════

def crop_face_jpeg(bgr: np.ndarray, box: np.ndarray, pad: float = 0.20) -> str:
    """Crop face with padding, encode to JPEG, return base64 string."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    x1c = max(0, x1 - px);  y1c = max(0, y1 - py)
    x2c = min(w, x2 + px);  y2c = min(h, y2 + py)
    crop = bgr[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return ""
    ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ""
    return base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera worker
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    """Reads one RTSP stream, runs inference, queues events."""

    def __init__(self, cam: dict, detector: FaceNPU,
                 event_queue: queue.Queue, min_interval: float):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam          = cam
        self.detector     = detector
        self.event_queue  = event_queue
        self.min_interval = min_interval
        self._run         = True
        self._last_event  = 0.0

    def stop(self) -> None:
        self._run = False

    def run(self) -> None:
        cam = self.cam
        print(f"[{cam['name']}] opening RTSP {cam['url'][:60]}…", flush=True)

        reader = RTSPReader(cam["url"], grab_drain=1)

        # Wait up to 15 s for first frame
        t0 = time.time()
        while time.time() - t0 < 15.0:
            ok, fr, _ = reader.read()
            if ok and fr is not None:
                break
            time.sleep(0.05)
        else:
            print(f"[{cam['name']}] ✗ no frame in 15s, exiting", flush=True)
            reader.release()
            return

        worker = InferenceWorker(self.detector)
        worker.start()
        last_seq = -1

        try:
            while self._run:
                ok, frame, seq = reader.read_copy()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                if seq != last_seq:
                    worker.submit(frame)
                    last_seq = seq

                det = worker.get_det()
                if det is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if now - self._last_event < self.min_interval:
                    time.sleep(0.01)
                    continue

                boxes_px, scores, classes = det
                self._last_event = now

                # Emit one event per detected face
                for box, score in zip(boxes_px, scores):
                    crop_b64 = crop_face_jpeg(frame, box)
                    self.event_queue.put({
                        "camId":   cam["id"],
                        "camName": cam["name"],
                        "score":   round(float(score), 3),
                        "ts":      int(now * 1000),
                        "crop":    crop_b64,
                    })

        finally:
            worker.stop()
            reader.release()
            print(f"[{cam['name']}] stopped", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Poster thread  (serialises HTTP calls so we don't DDoS ourselves)
# ══════════════════════════════════════════════════════════════════════════════

class PosterThread(threading.Thread):
    def __init__(self, server: str, q: queue.Queue):
        super().__init__(daemon=True, name="poster")
        self.server = server
        self.q      = q

    def run(self) -> None:
        while True:
            payload = self.q.get()
            post_event(self.server, payload)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-camera face detection service")
    ap.add_argument("--library",      required=True, help="Path to libnn_*.so")
    ap.add_argument("--model",        required=True, help="Path to *.nb model")
    ap.add_argument("--server",       default="http://localhost:3000")
    ap.add_argument("--interval",     type=float, default=2.0,
                    help="Min seconds between face events per camera")
    ap.add_argument("--face-thresh",  type=float, default=0.50)
    ap.add_argument("--nms",          type=float, default=0.45)
    ap.add_argument("--tiles",        type=int,   default=1, choices=[0,1,2,4])
    ap.add_argument("--level",        type=int,   default=0)
    ap.add_argument("--poll",         type=float, default=30.0,
                    help="Seconds between camera-list refresh")
    args = ap.parse_args()

    _cfg.face_thresh = args.face_thresh
    _cfg.nms         = args.nms

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    print("Initialising NPU model…", flush=True)
    detector        = FaceNPU(args.library, args.model, args.level)
    detector.tiles  = 0 if args.tiles <= 1 else args.tiles

    event_queue: queue.Queue = queue.Queue(maxsize=500)
    PosterThread(args.server, event_queue).start()

    workers: Dict[str, CameraWorker] = {}

    def sync_cameras() -> None:
        try:
            cams = fetch_cameras(args.server)
        except Exception as e:
            print(f"[sync] fetch cameras failed: {e}", flush=True)
            return

        current_ids = {c["id"] for c in cams}
        running_ids = set(workers.keys())

        # Stop removed cameras
        for rid in running_ids - current_ids:
            print(f"[sync] stopping removed camera {rid}", flush=True)
            workers[rid].stop()
            del workers[rid]

        # Start new cameras
        for cam in cams:
            if cam["id"] not in workers:
                print(f"[sync] starting camera {cam['name']}", flush=True)
                w = CameraWorker(cam, detector, event_queue, args.interval)
                workers[cam["id"]] = w
                w.start()

    sync_cameras()

    try:
        while True:
            time.sleep(args.poll)
            sync_cameras()
    except KeyboardInterrupt:
        print("\nShutting down…")
        for w in workers.values():
            w.stop()


if __name__ == "__main__":
    main()
