#!/usr/bin/env python3
"""
face_server.py — Multi-camera face detection bridge
────────────────────────────────────────────────────
Reads cameras.json, opens each RTSP stream, runs NPU inference
(same pipeline as cap.py), then POSTs face-crop events as JSON to
the Node.js server at http://localhost:3000/api/face-event

Each event payload:
  {
    "camera_id":  "front_door_xyz",
    "camera_name": "Front Door",
    "ts":          1712345678.123,       # Unix timestamp float
    "score":       0.93,
    "bbox":        [x1,y1,x2,y2],       # pixels in original frame
    "crop_b64":    "<base64 JPEG>"       # 96×96 face crop
  }

Usage:
  python3 face_server.py \
      --library libnn_yolo11n-face.so \
      --model   yolo11n-face.nb \
      [--face-thresh 0.50] [--nms 0.45] [--tiles 1] [--cooldown 2.0]

The script auto-reloads cameras.json every 10 s so you can add/remove
cameras from the dashboard without restarting.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# ── thread-count caps before any import ──────────────────────────────────────
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

import cv2 as cv
import numpy as np
import urllib.request

# ── re-use all inference code from cap.py verbatim ───────────────────────────
# We import the module-level symbols we need; cap.py must be in the same dir.

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from cap import (   # noqa: E402
    FaceNPU, InferenceWorker, RTSPReader,
    _cfg, draw,
)

# ─────────────────────────────────────────────────────────────────────────────

CAMERAS_DB   = os.path.join(_here, "cameras.json")
NODE_EVENT   = "http://localhost:3000/api/face-event"

DEFAULT_COOLDOWN = 2.0   # seconds between events for the same camera


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cameras() -> List[Dict]:
    try:
        with open(CAMERAS_DB) as f:
            return json.load(f)
    except Exception:
        return []


def crop_b64(frame_bgr: np.ndarray, box: np.ndarray, pad: float = 0.18) -> str:
    """Crop face region (with padding), resize to 96×96, encode as JPEG base64."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    crop = cv.resize(crop, (96, 96), interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def post_event(payload: dict):
    """Fire-and-forget POST to Node.js."""
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            NODE_EVENT,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[post_event] {e}", flush=True)


# ── per-camera worker ─────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """Runs one RTSP stream + inference thread for a single camera."""

    def __init__(self, cam: dict, detector: FaceNPU, cooldown: float):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam       = cam
        self.detector  = detector
        self.cooldown  = cooldown
        self._run      = True
        self._last_ts: float = 0.0

    def stop(self):
        self._run = False

    def run(self):
        cam_id   = self.cam["id"]
        cam_name = self.cam["name"]
        url      = self.cam["url"]

        print(f"[{cam_name}] Opening {url[:60]}…", flush=True)

        try:
            reader = RTSPReader(url, grab_drain=1)
        except Exception as e:
            print(f"[{cam_name}] RTSPReader error: {e}", flush=True)
            return

        # Wait up to 15s for first frame
        t0 = time.time()
        while time.time() - t0 < 15.0:
            ok, fr, _ = reader.read()
            if ok and fr is not None:
                break
            time.sleep(0.05)
        else:
            print(f"[{cam_name}] No frame within 15s — aborting.", flush=True)
            reader.release()
            return

        print(f"[{cam_name}] Stream open. Starting inference…", flush=True)

        worker = InferenceWorker(self.detector)
        worker.start()

        last_seq = -1

        try:
            while self._run:
                ok, frame, seq = reader.read_copy()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue

                if seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq

                worker.submit(frame)
                det = worker.get_det()

                if det is None:
                    continue

                boxes_px, scores, _ = det
                now = time.time()

                # Cooldown: emit at most 1 event per camera per `cooldown` seconds
                if now - self._last_ts < self.cooldown:
                    continue
                self._last_ts = now

                # Pick highest-confidence face
                best = int(np.argmax(scores))
                box  = boxes_px[best]
                score = float(scores[best])
                crop  = crop_b64(frame, box)

                payload = {
                    "camera_id":   cam_id,
                    "camera_name": cam_name,
                    "ts":          now,
                    "score":       round(score, 3),
                    "bbox":        [int(v) for v in box],
                    "crop_b64":    crop,
                    "face_count":  len(scores),
                }

                print(
                    f"[{cam_name}] Face detected  score={score:.2f}  "
                    f"faces={len(scores)}  t={time.strftime('%H:%M:%S')}",
                    flush=True,
                )
                threading.Thread(target=post_event, args=(payload,), daemon=True).start()

        finally:
            worker.stop()
            reader.release()
            print(f"[{cam_name}] Worker stopped.", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-camera face detection → NVR dashboard")
    ap.add_argument("--library",     required=True, help="Path to libnn_*.so")
    ap.add_argument("--model",       required=True, help="Path to *.nb model")
    ap.add_argument("--level",       type=int,   default=0)
    ap.add_argument("--face-thresh", type=float, default=0.50)
    ap.add_argument("--nms",         type=float, default=0.45)
    ap.add_argument("--tiles",       type=int,   default=1, choices=[0,1,2,4])
    ap.add_argument("--tile-overlap",type=float, default=0.25)
    ap.add_argument("--cooldown",    type=float, default=DEFAULT_COOLDOWN,
                    help="Min seconds between face events per camera (default 2)")
    args = ap.parse_args()

    if not os.path.isfile(args.library):
        sys.exit(f"Library not found: {args.library}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    _cfg.face_thresh = args.face_thresh
    _cfg.nms         = args.nms

    print("Loading NPU model…", flush=True)
    detector = FaceNPU(args.library, args.model, args.level)
    detector.tiles        = 0 if args.tiles <= 1 else args.tiles
    detector.tile_overlap = args.tile_overlap
    print("NPU ready.", flush=True)

    active: Dict[str, CameraWorker] = {}   # cam_id → worker

    def sync_cameras():
        """Start workers for new cameras; stop workers for removed ones."""
        cams     = load_cameras()
        new_ids  = {c["id"] for c in cams}
        cur_ids  = set(active.keys())

        # Remove stale
        for cid in cur_ids - new_ids:
            print(f"[sync] Removing camera {cid}", flush=True)
            active[cid].stop()
            del active[cid]

        # Add new
        for cam in cams:
            if cam["id"] not in active:
                print(f"[sync] Starting camera {cam['name']}", flush=True)
                w = CameraWorker(cam, detector, args.cooldown)
                active[cam["id"]] = w
                w.start()

    # Initial sync
    sync_cameras()

    print(f"Running — watching {CAMERAS_DB} for changes every 10s", flush=True)
    print(f"Posting face events → {NODE_EVENT}", flush=True)

    try:
        while True:
            time.sleep(10)
            sync_cameras()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
        for w in active.values():
            w.stop()


if __name__ == "__main__":
    main()
