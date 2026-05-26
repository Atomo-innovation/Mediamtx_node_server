#!/usr/bin/env python3
"""
face_detector.py — NPU Face Detection + Recognition via MediaMTX local re-stream
----------------------------------------------------------------------------------
Architecture (mirrors app.py pipeline exactly):
  1. ThrottledRTSPReader  → reads rtsp://localhost:8554/<camId> (MediaMTX re-stream)
                            grab() drains buffer, retrieve() only at --fps rate
  2. NPU InferenceWorker (from cap.py)  → YOLO11n-face on Khadas NPU (asnn)
  3. RecogWorker         → aligner.py align_face() + NPUFaceRecognizer (MobileFaceNet)
                           centroid matching + multi-frame voting (from app.py)
  4. PosterThread        → HTTP POST to Node.js /api/face-events with crop + identity

Face DB (face_db.json) is shared with app.py centroid format:
  { "centroids": { "name": [512-d float] }, "counts": { "__count_name": N } }

Usage:
  python3 face_detector.py \\
      --library  ./yolo11n-face/libnn_yolo11n-face.so \\
      --model    ./yolo11n-face/yolo11n-face.nb \\
      --recog_model models/mobilefacenet.onnx \\
      [--server     http://localhost:3000] \\
      [--mtx_rtsp   rtsp://localhost:8554] \\
      [--fps        2]        # inference frames/sec per camera
      [--interval   3.0]      # min seconds between posted events per camera
      [--tolerance  0.45]     # cosine distance threshold for identity match
      [--vote_frames 4]       # frames needed to confirm identity
      [--use_npu    true]     # MobileFaceNet on NPU (TIM-VX) or CPU
      [--face_thresh 0.50]
      [--poll       30]       # seconds between camera-list refresh

DB management (same endpoint format as app.py):
  Enroll a face:
    POST /api/enroll  { "name": "Alice", "crop_b64": "<base64 jpeg>" }
  List known people:
    GET  /api/known
  Delete a person:
    DELETE /api/known/<name>
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

# ── env tweaks ────────────────────────────────────────────────────────────────
os.environ["OPENCV_LOG_LEVEL"]       = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    "|max_delay;0|reorder_queue_size;0|loglevel;quiet"
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

# ── local imports (same directory as cap.py, aligner.py, recognizer.py) ──────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cap        import FaceNPU, InferenceWorker, _cfg
from aligner    import align_face
from recognizer import NPUFaceRecognizer


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
def _str2bool(v): return str(v).lower() in ("true","yes","1","y","t")

ap = argparse.ArgumentParser(description="NPU face detection + recognition service")
ap.add_argument("--library",      required=True,  help="Path to libnn_yolo11n-face.so")
ap.add_argument("--model",        required=True,  help="Path to yolo11n-face.nb")
ap.add_argument("--recog_model",  required=True,  help="Path to mobilefacenet.onnx")
ap.add_argument("--recog_type",   default="mobilefacenet",
                choices=["mobilefacenet","arcface_r50"])
ap.add_argument("--use_npu",      default=True,   type=_str2bool,
                help="Run MobileFaceNet on NPU TIM-VX (default true)")
ap.add_argument("--server",       default="http://localhost:3000")
ap.add_argument("--mtx_rtsp",     default="rtsp://localhost:8554",
                help="MediaMTX local RTSP base URL")
ap.add_argument("--fps",          type=float, default=2.0,
                help="Detection frames/sec per camera (default 2)")
ap.add_argument("--interval",     type=float, default=3.0,
                help="Min seconds between posted events per camera")
ap.add_argument("--tolerance",    type=float, default=0.45,
                help="Cosine distance threshold for identity match")
ap.add_argument("--vote_frames",  type=int,   default=4,
                help="Frames to confirm identity (same as app.py)")
ap.add_argument("--min_quality",  type=float, default=20.0,
                help="Laplacian blur threshold — skip blurry crops")
ap.add_argument("--min_face_px",  type=int,   default=40,
                help="Minimum face side in pixels")
ap.add_argument("--face_thresh",  type=float, default=0.50)
ap.add_argument("--nms",          type=float, default=0.45)
ap.add_argument("--tiles",        type=int,   default=1, choices=[0,1,2,4])
ap.add_argument("--level",        type=int,   default=0)
ap.add_argument("--poll",         type=float, default=30.0,
                help="Seconds between camera-list refresh from Node server")
ap.add_argument("--db",           default="face_db.json",
                help="Path to centroid DB (shared with app.py format)")
ap.add_argument("--mgmt_port",    type=int,   default=5051,
                help="HTTP port for /api/enroll and /api/known endpoints")
ARGS = ap.parse_args()

_cfg.face_thresh = ARGS.face_thresh
_cfg.nms         = ARGS.nms


# ══════════════════════════════════════════════════════════════════════════════
# FACE DATABASE  — centroid format (identical to app.py)
# ══════════════════════════════════════════════════════════════════════════════
_db_lock         = threading.Lock()
known_centroids: Dict[str, np.ndarray] = {}   # name → 512-d L2 unit vector
                                               # "__count_name" → int

DB_PATH = ARGS.db


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def load_db() -> None:
    global known_centroids
    if not os.path.exists(DB_PATH):
        print(f"[DB] No database at {DB_PATH} — starting empty", flush=True)
        return
    try:
        with open(DB_PATH) as f:
            d = json.load(f)
        with _db_lock:
            known_centroids = {}
            if "centroids" in d:
                for k, v in d["centroids"].items():
                    known_centroids[k] = np.array(v, dtype=np.float32)
                for k, v in d.get("counts", {}).items():
                    known_centroids[k] = int(v)
            elif "names" in d and "encodings" in d:
                # migrate old list-of-embeddings format
                from collections import defaultdict as dd
                groups = dd(list)
                for n, e in zip(d["names"], d["encodings"]):
                    groups[n].append(np.array(e, dtype=np.float32))
                for n, embs in groups.items():
                    c    = np.mean(embs, axis=0).astype(np.float32)
                    norm = np.linalg.norm(c)
                    known_centroids[n] = c / norm if norm > 1e-9 else c
                    known_centroids[f"__count_{n}"] = len(embs)
                _save_db_locked()
        real = {k for k in known_centroids if not k.startswith("__count_")}
        print(f"[DB] Loaded {len(real)} known people: {sorted(real)}", flush=True)
    except Exception as e:
        print(f"[DB] Load error: {e}", flush=True)


def _save_db_locked() -> None:
    """Must be called with _db_lock held."""
    centroids, counts = {}, {}
    for k, v in known_centroids.items():
        if k.startswith("__count_"):
            counts[k] = int(v)
        else:
            centroids[k] = np.array(v, dtype=np.float32).tolist()
    with open(DB_PATH, "w") as f:
        json.dump({"centroids": centroids, "counts": counts,
                   "names": [k for k in centroids]}, f)


def _add_to_centroid(name: str, emb: np.ndarray) -> None:
    with _db_lock:
        if name in known_centroids and isinstance(known_centroids[name], np.ndarray):
            count   = known_centroids.get(f"__count_{name}", 1)
            new_c   = (known_centroids[name] * count + emb) / (count + 1)
            norm    = np.linalg.norm(new_c)
            known_centroids[name]              = new_c / norm if norm > 1e-9 else new_c
            known_centroids[f"__count_{name}"] = count + 1
        else:
            norm = np.linalg.norm(emb)
            known_centroids[name]              = emb / norm if norm > 1e-9 else emb.copy()
            known_centroids[f"__count_{name}"] = 1
        _save_db_locked()


def _match(emb: np.ndarray) -> Tuple[str, float]:
    with _db_lock:
        real = {k: v for k, v in known_centroids.items()
                if not k.startswith("__count_") and isinstance(v, np.ndarray)}
    if not real:
        return "Unknown", 0.0
    best_name, best_dist = "Unknown", 1.0
    for name, centroid in real.items():
        d = _cosine(emb, centroid)
        if d < best_dist:
            best_dist, best_name = d, name
    conf = round(1.0 - best_dist, 3)
    if best_dist <= ARGS.tolerance:
        return best_name, conf
    return "Unknown", conf


def _known_names() -> List[str]:
    with _db_lock:
        return sorted(k for k in known_centroids if not k.startswith("__count_"))


def _delete_person(name: str) -> bool:
    with _db_lock:
        if name not in known_centroids:
            return False
        del known_centroids[name]
        known_centroids.pop(f"__count_{name}", None)
        _save_db_locked()
    return True


def _quality(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (sequential with 1.5 s NPU cooldown — same as app.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*56, flush=True)
print("  face_detector — NPU Detect + Recognise", flush=True)
print("="*56, flush=True)
print(f"  Detection  : NPU (asnn / Electron)", flush=True)
print(f"  Recognition: {'NPU (TIM-VX)' if ARGS.use_npu else 'CPU (onnxruntime)'}", flush=True)
print(f"  Tolerance  : {ARGS.tolerance}   Vote frames: {ARGS.vote_frames}", flush=True)
print(f"  Inference  : {ARGS.fps} fps/camera   Min face: {ARGS.min_face_px}px", flush=True)
print("="*56 + "\n", flush=True)

print("[Boot] 1/2  Loading YOLO11n-face (NPU) …", flush=True)
_det_model            = FaceNPU(ARGS.library, ARGS.model, ARGS.level)
_det_model.tiles      = 0 if ARGS.tiles <= 1 else ARGS.tiles
_det_model.tile_overlap = 0.25
print("[Boot]      YOLO11n-face OK", flush=True)

print("[Boot] NPU cooldown 1.5 s …", flush=True)
gc.collect()
time.sleep(1.5)

print("[Boot] 2/2  Loading MobileFaceNet …", flush=True)
_recog_mutex = threading.Lock()
try:
    _recognizer = NPUFaceRecognizer(ARGS.recog_model, ARGS.recog_type, ARGS.use_npu)
    print(f"[Boot]      MobileFaceNet OK ({'NPU' if ARGS.use_npu else 'CPU'})", flush=True)
except Exception as e:
    print(f"[Boot]      NPU recognizer failed ({e}), trying CPU …", flush=True)
    _recognizer = NPUFaceRecognizer(ARGS.recog_model, ARGS.recog_type, False)
    print("[Boot]      MobileFaceNet OK (CPU fallback)", flush=True)

gc.collect()
time.sleep(0.5)
print("[Boot] All models loaded\n", flush=True)

load_db()


def _get_embedding(aligned: np.ndarray) -> Optional[np.ndarray]:
    try:
        with _recog_mutex:
            return _recognizer.get_embedding(aligned)
    except Exception as e:
        print(f"[Recog] Error: {e}", flush=True)
        time.sleep(0.3)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# THROTTLED RTSP READER  (local MediaMTX re-stream — low CPU)
# ══════════════════════════════════════════════════════════════════════════════
class ThrottledRTSPReader:
    """
    Reads from rtsp://localhost:8554/<camId>.
    grab() drains the decoder queue cheaply; retrieve() only at min_interval.
    Exponential backoff reconnect for offline streams.
    """
    MAX_FAILS    = 5
    OPEN_TIMEOUT = 12.0

    def __init__(self, url: str, min_interval: float = 0.5):
        self.url          = url
        self.min_interval = max(min_interval, 0.05)
        self._frame: Optional[np.ndarray] = None
        self._seq         = 0
        self._lock        = threading.Lock()
        self._run         = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        backoff = 2.0
        while self._run:
            cap    = self._open()
            t0     = time.time()
            ready  = False
            while self._run and (time.time() - t0) < self.OPEN_TIMEOUT:
                ok, fr = cap.read()
                if ok and fr is not None:
                    with self._lock:
                        self._frame = fr
                        self._seq  += 1
                    ready   = True
                    backoff = 2.0
                    break
                time.sleep(0.2)

            if not ready:
                cap.release()
                if self._run:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                continue

            fails          = 0
            next_retrieve  = time.perf_counter()
            while self._run:
                now = time.perf_counter()
                if now < next_retrieve:
                    if not cap.grab():
                        fails += 1
                        if fails >= self.MAX_FAILS:
                            break
                    else:
                        fails = 0
                    time.sleep(0.003)
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

    def read_copy(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame.copy(), self._seq

    def release(self) -> None:
        self._run = False


# ══════════════════════════════════════════════════════════════════════════════
# FACE CROP HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _crop_padded(frame: np.ndarray, box: np.ndarray, pad: float = 0.15
                 ) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    if bw < ARGS.min_face_px or bh < ARGS.min_face_px:
        return None
    px, py = int(bw * pad), int(bh * pad)
    x1c = max(0, x1 - px);  y1c = max(0, y1 - py)
    x2c = min(w, x2 + px);  y2c = min(h, y2 + py)
    crop = frame[y1c:y2c, x1c:x2c]
    return crop if crop.size > 0 else None


def _to_b64(img: np.ndarray, quality: int = 75) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else ""


# ══════════════════════════════════════════════════════════════════════════════
# RECOGNITION WORKER  (per camera — mirrors app.py RecogWorker exactly)
# ══════════════════════════════════════════════════════════════════════════════
class RecogWorker(threading.Thread):
    """
    Consumes (pos_key, aligned_112, crop_bgr) from its queue.
    Extracts MobileFaceNet embedding, does centroid matching + vote buffer.
    Confirmed detections → event_queue for posting to Node server.
    """

    def __init__(self, cam_id: int, cam_name: str,
                 recog_q: queue.Queue, event_q: queue.Queue,
                 min_interval: float):
        super().__init__(daemon=True, name=f"RecogWorker-{cam_id}")
        self.cam_id       = cam_id
        self.cam_name     = cam_name
        self.recog_q      = recog_q
        self.event_q      = event_q
        self.min_interval = min_interval
        self._running     = True
        self._vote_buf:  Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=ARGS.vote_frames))
        self._track_lbl: Dict[str, Tuple[str, float]] = {}
        self._last_event: Dict[str, float] = {}   # pos_key → last post time

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        print(f"[RecogWorker-{self.cam_id}] Started", flush=True)
        while self._running:
            try:
                pos_key, aligned, crop_bgr = self.recog_q.get(timeout=0.5)
            except queue.Empty:
                continue

            emb = _get_embedding(aligned)
            if emb is None:
                continue

            name, conf = _match(emb)

            # ── voting (identical to app.py) ──────────────────────────────────
            self._vote_buf[pos_key].append((name, conf))
            votes = list(self._vote_buf[pos_key])
            if len(votes) < ARGS.vote_frames:
                continue

            vote_names  = [v[0] for v in votes]
            most_common = max(set(vote_names), key=vote_names.count)
            vote_count  = vote_names.count(most_common)
            if vote_count < max(2, ARGS.vote_frames // 2 + 1):
                continue

            avg_conf = float(np.mean([v[1] for v in votes
                                      if v[0] == most_common]))

            # ── hold logic: don't flip known → unknown on a single bad frame ──
            prev = self._track_lbl.get(pos_key)
            if (most_common == "Unknown" and prev
                    and prev[0] != "Unknown"):
                continue   # wait for cleaner frames before overriding known

            self._track_lbl[pos_key] = (most_common, avg_conf)

            # ── min interval gate ─────────────────────────────────────────────
            now = time.time()
            if now - self._last_event.get(pos_key, 0) < self.min_interval:
                continue
            self._last_event[pos_key] = now

            # ── post event ────────────────────────────────────────────────────
            crop_b64 = _to_b64(crop_bgr)
            self.event_q.put({
                "camId":    f"cam_{self.cam_id}",
                "camName":  self.cam_name,
                "name":     most_common,
                "known":    most_common != "Unknown",
                "score":    round(avg_conf, 3),
                "ts":       int(now * 1000),
                "crop":     crop_b64,
            })

        print(f"[RecogWorker-{self.cam_id}] Stopped", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# PER-CAMERA WORKER
# ══════════════════════════════════════════════════════════════════════════════
class CameraWorker(threading.Thread):
    """
    Reads from MediaMTX local RTSP → YOLO11n-face NPU detect →
    align_face() → RecogWorker queue.
    """

    def __init__(self, cam: dict, event_q: queue.Queue):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam       = cam
        self.local_url = f"{ARGS.mtx_rtsp.rstrip('/')}/{cam['id']}"
        self._run      = True

        frame_interval = 1.0 / max(ARGS.fps, 0.1)
        self._reader   = ThrottledRTSPReader(self.local_url,
                                             min_interval=frame_interval)
        self._recog_q  = queue.Queue(maxsize=8)
        self._recog_w  = RecogWorker(
            cam_id       = 0,          # slot index within this camera
            cam_name     = cam["name"],
            recog_q      = self._recog_q,
            event_q      = event_q,
            min_interval = ARGS.interval,
        )
        self._npu_w    = InferenceWorker(_det_model)
        self._last_seq = -1

    def stop(self) -> None:
        self._run = False
        self._recog_w.stop()

    def run(self) -> None:
        cam = self.cam
        print(f"[{cam['name']}] → {self.local_url}", flush=True)
        self._npu_w.start()
        self._recog_w.start()

        try:
            while self._run:
                ok, frame, seq = self._reader.read_copy()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                if seq == self._last_seq:
                    time.sleep(0.01)
                    continue
                self._last_seq = seq

                # submit to NPU detection worker (always processes newest frame)
                self._npu_w.submit(frame)
                det = self._npu_w.get_det()
                if det is None:
                    time.sleep(0.01)
                    continue

                boxes_px, scores, _ = det
                h, w = frame.shape[:2]

                for i, (box, score) in enumerate(zip(boxes_px, scores)):
                    x1, y1, x2, y2 = box.astype(int)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < ARGS.min_face_px or bh < ARGS.min_face_px:
                        continue

                    # crop for event thumbnail
                    crop = _crop_padded(frame, box, pad=0.15)
                    if crop is None:
                        continue

                    # quality gate
                    if _quality(crop) < ARGS.min_quality:
                        continue

                    # The YOLO11-face model does NOT output 5 landmarks like
                    # YuNet — we get a box only.  Use a centre-crop align.
                    # align_face expects (5,2) landmarks; we fabricate them
                    # from box geometry so aligner.py works unchanged.
                    lms = _estimate_landmarks(x1, y1, x2, y2)
                    aligned = align_face(frame, lms, size=112)

                    pos_key = f"f{i}"
                    item    = (pos_key, aligned, crop)
                    try:
                        self._recog_q.put_nowait(item)
                    except queue.Full:
                        try:    self._recog_q.get_nowait()
                        except queue.Empty: pass
                        try:    self._recog_q.put_nowait(item)
                        except queue.Full:  pass

        finally:
            self._npu_w.stop()
            self._recog_w.stop()
            self._reader.release()
            print(f"[{cam['name']}] stopped", flush=True)


def _estimate_landmarks(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Estimate ArcFace-style 5-point landmarks from a bounding box.
    YOLO11-face outputs boxes only (no keypoints in the asnn wrapper).
    These are a good-enough approximation for the affine warp.
    """
    w = x2 - x1
    h = y2 - y1
    # Proportions from mean face geometry (empirically derived)
    le  = [x1 + w * 0.30, y1 + h * 0.37]   # left eye
    re  = [x1 + w * 0.70, y1 + h * 0.37]   # right eye
    nos = [x1 + w * 0.50, y1 + h * 0.56]   # nose tip
    lm  = [x1 + w * 0.35, y1 + h * 0.75]   # left mouth
    rm  = [x1 + w * 0.65, y1 + h * 0.75]   # right mouth
    return np.array([le, re, nos, lm, rm], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# HTTP POSTER  (serialises POSTs to avoid overwhelming Node server)
# ══════════════════════════════════════════════════════════════════════════════
class PosterThread(threading.Thread):
    def __init__(self, server: str, q: queue.Queue):
        super().__init__(daemon=True, name="poster")
        self.server = server.rstrip("/")
        self.q      = q

    def run(self) -> None:
        while True:
            payload = self.q.get()
            try:
                data = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    f"{self.server}/api/face-events",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=6):
                    pass
            except Exception as e:
                print(f"[poster] POST failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA LIST SYNC
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_cameras() -> list:
    url = f"{ARGS.server.rstrip('/')}/api/cameras"
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read())


# ══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT HTTP SERVER  (/api/enroll  /api/known  DELETE /api/known/<name>)
# ══════════════════════════════════════════════════════════════════════════════
class _MgmtHandler(BaseHTTPRequestHandler):
    """
    Tiny management API so you can enroll faces without running full app.py.
    POST /api/enroll    { "name": "Alice", "crop_b64": "<base64 jpeg>" }
    GET  /api/known     → { "names": ["Alice", ...] }
    DELETE /api/known/<name>
    """

    def log_message(self, fmt, *args):   # silence default logging
        pass

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/known":
            self._send_json(200, {"names": _known_names()})
        elif self.path == "/api/db_reload":
            load_db()
            self._send_json(200, {"ok": True, "names": _known_names()})
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if self.path.startswith("/api/known/"):
            name = self.path[len("/api/known/"):]
            ok   = _delete_person(name)
            self._send_json(200 if ok else 404,
                            {"ok": ok, "msg": f"{'Deleted' if ok else 'Not found'}: {name}"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/enroll":
            self._send_json(404, {"error": "not found"})
            return
        length  = int(self.headers.get("Content-Length", 0))
        body    = json.loads(self.rfile.read(length))
        name    = str(body.get("name","")).strip()
        b64     = body.get("crop_b64","")
        if not name or not b64:
            self._send_json(400, {"error": "name and crop_b64 required"})
            return
        try:
            img_data = base64.b64decode(b64)
            arr      = np.frombuffer(img_data, dtype=np.uint8)
            img      = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            self._send_json(400, {"error": f"Image decode failed: {e}"})
            return

        # Align + enroll with augmentation (mirrors app.py _enroll_face)
        h2, w2 = img.shape[:2]
        lms     = _estimate_landmarks(0, 0, w2, h2)
        aligned = align_face(img, lms, size=112)

        # 4 augmentations: original + flip + brightness ±20
        variants = [
            aligned,
            cv2.flip(aligned, 1),
            cv2.convertScaleAbs(aligned, alpha=1.0, beta=20),
            cv2.convertScaleAbs(aligned, alpha=1.0, beta=-20),
        ]
        count = 0
        for v in variants:
            emb = _get_embedding(v)
            if emb is not None:
                _add_to_centroid(name, emb)
                count += 1
            time.sleep(0.02)

        real_count = known_centroids.get(f"__count_{name}", 0)
        self._send_json(200, {
            "ok":      True,
            "name":    name,
            "added":   count,
            "total":   int(real_count),
            "message": f"Enrolled {count} embeddings for '{name}' (total: {int(real_count)})",
        })


def _start_mgmt_server() -> None:
    srv = HTTPServer(("0.0.0.0", ARGS.mgmt_port), _MgmtHandler)
    t   = threading.Thread(target=srv.serve_forever, daemon=True, name="MgmtHTTP")
    t.start()
    print(f"[Mgmt] Enroll API → http://0.0.0.0:{ARGS.mgmt_port}/api/enroll", flush=True)
    print(f"[Mgmt] Known list → http://0.0.0.0:{ARGS.mgmt_port}/api/known",  flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _start_mgmt_server()

    event_queue: queue.Queue = queue.Queue(maxsize=500)
    PosterThread(ARGS.server, event_queue).start()

    workers: Dict[str, CameraWorker] = {}

    def sync_cameras() -> None:
        try:
            cams = _fetch_cameras()
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
                w = CameraWorker(cam, event_queue)
                workers[cam["id"]] = w
                w.start()

    sync_cameras()

    try:
        while True:
            time.sleep(ARGS.poll)
            sync_cameras()
    except KeyboardInterrupt:
        print("\nShutting down …", flush=True)
        for w in list(workers.values()):
            w.stop()


if __name__ == "__main__":
    main()
