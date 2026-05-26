#!/usr/bin/env python3
"""
face_detector.py — Multi-camera face detection via MediaMTX re-stream
-----------------------------------------------------------------------
Instead of opening the original RTSP source again (which doubles network
load, auth, and decode CPU), this service reads from the MediaMTX LOCAL
re-stream:  rtsp://localhost:8554/<camId>

MediaMTX already holds the upstream connection and does the decode; we
just attach as a second local consumer.  All NPU inference runs on the
Khadas NPU via asnn/Electron — zero CPU for inference.

CPU savings vs original face_detector.py
  • No second upstream RTSP connection
  • Local loopback TCP — no network decode overhead
  • Frame throttling: grab() drops frames, retrieve() only every N seconds
  • FFmpeg stderr suppressed for offline cameras
  • Exponential backoff for dead cameras (no hammer-retry)

Usage:
  python3 face_detector.py \\
      --library ./yolo11n-face/libnn_yolo11n-face.so \\
      --model   ./yolo11n-face/yolo11n-face.nb \\
      [--server  http://localhost:3000] \\
      [--mtx-rtsp rtsp://localhost:8554] \\
      [--fps     2]       # inference frames per second (default 2)
      [--interval 2.0]    # min seconds between posted events per camera
      [--tiles 1]
      [--face-thresh 0.50]
      [--poll 30]         # seconds between camera-list refresh
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from typing import Dict, Optional, Tuple

# ── silence FFmpeg / OpenCV stderr for offline cameras ────────────────────────
# We redirect fd=2 per-thread is not possible, so we suppress at lib level
os.environ["OPENCV_LOG_LEVEL"]   = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"   # AV_LOG_QUIET

for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
# Low-latency local loopback — no need for big buffers
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|max_delay;0"
    "|reorder_queue_size;0"
    "|loglevel;quiet"           # suppress FFmpeg output per-stream
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cap import (
    FaceNPU, InferenceWorker,
    _cfg,
)


# ══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════════════════════════════════════════════

def fetch_cameras(server: str) -> list:
    url = f"{server.rstrip('/')}/api/cameras"
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read())


def post_event(server: str, payload: dict) -> None:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{server.rstrip('/')}/api/face-events",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=6):
            pass
    except Exception as e:
        print(f"[poster] POST failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Throttled RTSP reader  (grab-drain + interval gate)
# ══════════════════════════════════════════════════════════════════════════════

class ThrottledRTSPReader:
    """
    Opens rtsp://localhost:8554/<camId> (MediaMTX local re-stream).
    Uses cap.grab() to drain the decoder queue, only calling retrieve()
    when at least `min_interval` seconds have elapsed.  This keeps CPU
    near zero between inference cycles.
    """

    OPEN_TIMEOUT = 12.0   # seconds to wait for first frame
    MAX_FAILS    = 5      # consecutive read failures before reconnect

    def __init__(self, url: str, min_interval: float = 0.5):
        self.url          = url
        self.min_interval = max(min_interval, 0.1)
        self._cap: Optional[cv.VideoCapture] = None
        self._lock        = threading.Lock()
        self._run         = True
        self._frame: Optional[np.ndarray] = None
        self._seq         = 0
        self._thread      = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── internal ──────────────────────────────────────────────────────────────

    def _open(self) -> cv.VideoCapture:
        cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        backoff = 2.0
        while self._run:
            cap = self._open()
            # Wait for stream to become available
            t0 = time.time()
            ready = False
            while self._run and (time.time() - t0) < self.OPEN_TIMEOUT:
                ok, fr = cap.read()
                if ok and fr is not None:
                    with self._lock:
                        self._frame = fr
                        self._seq  += 1
                    ready = True
                    backoff = 2.0
                    break
                time.sleep(0.2)

            if not ready:
                cap.release()
                if self._run:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                continue

            # Steady-state: grab-drain loop
            fails = 0
            next_retrieve = time.perf_counter()
            while self._run:
                now = time.perf_counter()
                if now < next_retrieve:
                    # Just drain decoder — no retrieve, no copy
                    if not cap.grab():
                        fails += 1
                        if fails >= self.MAX_FAILS:
                            break
                    else:
                        fails = 0
                    time.sleep(0.005)
                    continue

                ok, fr = cap.retrieve()
                if not ok or fr is None:
                    fails += 1
                    if fails >= self.MAX_FAILS:
                        break
                    time.sleep(0.01)
                    continue

                fails = 0
                next_retrieve = now + self.min_interval
                with self._lock:
                    self._frame = fr
                    self._seq  += 1

            cap.release()
            if self._run:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ── public ────────────────────────────────────────────────────────────────

    def read_copy(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame.copy(), self._seq

    def release(self) -> None:
        self._run = False


# ══════════════════════════════════════════════════════════════════════════════
# Face crop helper
# ══════════════════════════════════════════════════════════════════════════════

def crop_face_jpeg(bgr: np.ndarray, box: np.ndarray, pad: float = 0.20) -> str:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    px, py  = int(bw * pad), int(bh * pad)
    x1c = max(0, x1 - px);  y1c = max(0, y1 - py)
    x2c = min(w, x2 + px);  y2c = min(h, y2 + py)
    crop = bgr[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return ""
    ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode() if ok else ""


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera worker
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    """
    • Reads local MediaMTX re-stream (rtsp://localhost:8554/<camId>)
    • Submits frames to NPU InferenceWorker
    • Posts face-event JSON to Node server
    """

    def __init__(
        self,
        cam: dict,
        mtx_base: str,
        detector: FaceNPU,
        event_queue: queue.Queue,
        fps: float,
        min_interval: float,
    ):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam          = cam
        self.local_url    = f"{mtx_base.rstrip('/')}/{cam['id']}"
        self.detector     = detector
        self.event_queue  = event_queue
        self.frame_interval  = 1.0 / max(fps, 0.1)   # min seconds between frames fed to NPU
        self.min_interval    = min_interval            # min seconds between posted events
        self._run         = True
        self._last_event  = 0.0

    def stop(self) -> None:
        self._run = False

    def run(self) -> None:
        cam = self.cam
        print(f"[{cam['name']}] → local stream {self.local_url}", flush=True)

        reader = ThrottledRTSPReader(self.local_url, min_interval=self.frame_interval)
        npu_worker = InferenceWorker(self.detector)
        npu_worker.start()

        last_seq       = -1
        last_submit_t  = 0.0

        try:
            while self._run:
                ok, frame, seq = reader.read_copy()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                now = time.perf_counter()

                # Submit new frame to NPU only when seq changed
                if seq != last_seq and (now - last_submit_t) >= self.frame_interval:
                    npu_worker.submit(frame)
                    last_seq      = seq
                    last_submit_t = now

                det = npu_worker.get_det()
                if det is None:
                    time.sleep(0.01)
                    continue

                wall = time.time()
                if wall - self._last_event < self.min_interval:
                    time.sleep(0.01)
                    continue

                boxes_px, scores, _ = det
                self._last_event = wall

                for box, score in zip(boxes_px, scores):
                    crop_b64 = crop_face_jpeg(frame, box)
                    self.event_queue.put({
                        "camId":   cam["id"],
                        "camName": cam["name"],
                        "score":   round(float(score), 3),
                        "ts":      int(wall * 1000),
                        "crop":    crop_b64,
                    })

        finally:
            npu_worker.stop()
            reader.release()
            print(f"[{cam['name']}] stopped", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Poster thread
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
    ap = argparse.ArgumentParser(description="NPU face detection via MediaMTX local re-stream")
    ap.add_argument("--library",     required=True)
    ap.add_argument("--model",       required=True)
    ap.add_argument("--server",      default="http://localhost:3000",
                    help="Node.js server base URL")
    ap.add_argument("--mtx-rtsp",    default="rtsp://localhost:8554",
                    help="MediaMTX local RTSP base (default rtsp://localhost:8554)")
    ap.add_argument("--fps",         type=float, default=2.0,
                    help="Inference frames per second per camera (default 2)")
    ap.add_argument("--interval",    type=float, default=2.0,
                    help="Min seconds between posted face events per camera")
    ap.add_argument("--face-thresh", type=float, default=0.50)
    ap.add_argument("--nms",         type=float, default=0.45)
    ap.add_argument("--tiles",       type=int,   default=1, choices=[0,1,2,4])
    ap.add_argument("--level",       type=int,   default=0)
    ap.add_argument("--poll",        type=float, default=30.0,
                    help="Seconds between camera-list refresh")
    args = ap.parse_args()

    _cfg.face_thresh = args.face_thresh
    _cfg.nms         = args.nms

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    print("Initialising NPU model…", flush=True)
    detector       = FaceNPU(args.library, args.model, args.level)
    detector.tiles = 0 if args.tiles <= 1 else args.tiles

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

        for rid in running_ids - current_ids:
            print(f"[sync] stopping removed camera {rid}", flush=True)
            workers[rid].stop()
            del workers[rid]

        for cam in cams:
            if cam["id"] not in workers:
                print(f"[sync] starting camera '{cam['name']}'", flush=True)
                w = CameraWorker(
                    cam,
                    args.mtx_rtsp,
                    detector,
                    event_queue,
                    fps=args.fps,
                    min_interval=args.interval,
                )
                workers[cam["id"]] = w
                w.start()

    sync_cameras()

    try:
        while True:
            time.sleep(args.poll)
            sync_cameras()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
        for w in workers.values():
            w.stop()


if __name__ == "__main__":
    main()
