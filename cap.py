#!/usr/bin/env python3
"""
Face detection on USB / MIPI / RTSP using Khadas NPU (asnn / Electron).

Pipeline per frame:
  1. Capture (RTSP uses low-latency FFmpeg + background reader)
  2. Pre: BGR -> RGB, letterbox 640x640, /255, CHW float32
  3. NPU: nn_inference only (no CPU fallback for the model)
  4. Post: YOLO11-face heads decode + NMS + geometry filter on CPU
  5. Optional 2x2 tiles (--tiles 4) for small / off-center faces on 1080p

Realtime mode (default): display loop never waits on the NPU; a worker thread
always infers the latest frame. Use --tiles 4 only when you need max recall.

Examples:
  python3 cap.py --library libnn_yolo11n-face.so --model yolo11n-face.nb \\
      --type rtsp --device 'rtsp://user:pass@192.168.1.10:554/stream'

  python3 cap.py --library libnn_yolo11n-face.so --model yolo11n-face.nb \\
      --type usb --device 0 --tiles 4 --face-thresh 0.50
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2 as cv
import numpy as np
from asnn.api import asnn
from asnn.types import output_format

GRID0, GRID1, GRID2 = 20, 40, 80
LISTSIZE = 65
SPAN = 1
NUM_CLS = 1
DETECT_SIZE = 640
DEFAULT_FACE_THRESH = 0.50
DEFAULT_NMS = 0.45

FACE_MIN_ASPECT = 0.72
FACE_MAX_ASPECT = 1.38
FACE_MAX_REL_AREA = 0.10
FACE_MIN_REL_SIDE = 0.018

CLASSES = ("Face",)

constant_matrix = np.array(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.float32
).T

class _Cfg:
    face_thresh = DEFAULT_FACE_THRESH
    nms = DEFAULT_NMS


_cfg = _Cfg()


def letterbox(
    frame_rgb: np.ndarray,
    target: int = DETECT_SIZE,
    canvas: Optional[np.ndarray] = None,
):
    h, w = frame_rgb.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv.INTER_AREA if nh < h else cv.INTER_LINEAR
    resized = cv.resize(frame_rgb, (nw, nh), interpolation=interp)
    if canvas is None:
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    else:
        canvas.fill(114)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def unletterbox_boxes(boxes, scale, pad_x, pad_y, orig_w, orig_h, target=DETECT_SIZE):
    boxes = boxes.astype(np.float64, copy=True)
    boxes[:, [0, 2]] *= target
    boxes[:, [1, 3]] *= target
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= max(scale, 1e-6)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    return boxes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    xm = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - xm)
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-12)


def process(input_tensor):
    grid_h, grid_w = int(input_tensor.shape[0]), int(input_tensor.shape[1])
    box_class_probs = sigmoid(input_tensor[..., :NUM_CLS])

    box_0 = softmax(input_tensor[..., NUM_CLS : NUM_CLS + 16], -1)
    box_1 = softmax(input_tensor[..., NUM_CLS + 16 : NUM_CLS + 32], -1)
    box_2 = softmax(input_tensor[..., NUM_CLS + 32 : NUM_CLS + 48], -1)
    box_3 = softmax(input_tensor[..., NUM_CLS + 48 : NUM_CLS + 64], -1)

    result = np.zeros((grid_h, grid_w, 1, 4), dtype=np.float32)
    result[..., 0] = np.dot(box_0, constant_matrix)[..., 0]
    result[..., 1] = np.dot(box_1, constant_matrix)[..., 0]
    result[..., 2] = np.dot(box_2, constant_matrix)[..., 0]
    result[..., 3] = np.dot(box_3, constant_matrix)[..., 0]

    col = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w, 1, 1).astype(np.float32)
    row = np.tile(np.arange(grid_h).reshape(-1, 1), grid_w).reshape(grid_h, grid_w, 1, 1).astype(
        np.float32
    )
    grid = np.concatenate((col, row), axis=-1)
    gh, gw = float(grid_h), float(grid_w)
    result[..., 0:2] = (0.5 - result[..., 0:2] + grid) / (gw, gh)
    result[..., 2:4] = (0.5 + result[..., 2:4] + grid) / (gw, gh)
    return result, box_class_probs


def filter_boxes(boxes, box_class_probs):
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_class_scores >= _cfg.face_thresh)
    return boxes[pos], np.zeros(len(pos[0]), dtype=np.int32), box_class_scores[pos]


def nms_boxes(boxes, scores):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-12)
        order = rest[np.where(ovr <= _cfg.nms)[0]]
    return np.array(keep, dtype=np.int64)


def yolo_post_process(input_data):
    boxes_list, classes_list, scores_list = [], [], []
    for head in input_data:
        result, confidence = process(head)
        b, c, s = filter_boxes(result, confidence)
        if len(b) > 0:
            boxes_list.append(b.reshape(-1, 4))
            classes_list.append(c.reshape(-1))
            scores_list.append(s.reshape(-1))
    if not boxes_list:
        return None

    boxes = np.concatenate(boxes_list, axis=0)
    classes = np.concatenate(classes_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)

    keep = nms_boxes(boxes, scores)
    if len(keep) == 0:
        return None
    return boxes[keep], scores[keep], classes[keep]


def filter_face_geometry(boxes, scores, classes, orig_w: int, orig_h: int):
    if boxes is None or len(boxes) == 0:
        return None
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h / max(orig_w * orig_h, 1)
    aspect = w / np.maximum(h, 1e-6)
    rel_w = w / max(orig_w, 1)
    rel_h = h / max(orig_h, 1)
    keep = (
        (area >= 3e-5)
        & (area <= FACE_MAX_REL_AREA)
        & (aspect >= FACE_MIN_ASPECT)
        & (aspect <= FACE_MAX_ASPECT)
        & (rel_h >= FACE_MIN_REL_SIDE)
        & (rel_w >= FACE_MIN_REL_SIDE)
    )
    if not np.any(keep):
        return None
    return boxes[keep], scores[keep], classes[keep]


def reshape_heads(data):
    d0 = np.asarray(data[0], dtype=np.float32)
    d1 = np.asarray(data[1], dtype=np.float32)
    d2 = np.asarray(data[2], dtype=np.float32)
    return [
        np.transpose(d2.reshape(SPAN, LISTSIZE, GRID0, GRID0), (2, 3, 0, 1)),
        np.transpose(d1.reshape(SPAN, LISTSIZE, GRID1, GRID1), (2, 3, 0, 1)),
        np.transpose(d0.reshape(SPAN, LISTSIZE, GRID2, GRID2), (2, 3, 0, 1)),
    ]


def draw(image, boxes_px, scores, classes, verbose: bool = False):
    for box, score, cl in zip(boxes_px, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        if verbose:
            print(f"class: {CLASSES[cl]}, score: {score:.3f}, box: [{x1}, {y1}, {x2}, {y2}]")
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{CLASSES[cl]} {score:.2f}"
        cv.putText(image, label, (x1, max(0, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class FaceNPU:
    """NPU inference + CPU post-process (decode/NMS only)."""

    def __init__(self, library: str, model: str, level: int):
        self.net = asnn("Electron")
        print(f" |---+ asnn Version: {self.net.get_nn_version()} +---| ")
        print("Start init neural network ...")
        self.net.nn_init(library=library, model=model, level=level)
        print("Done.")
        self.tiles = 0
        self.tile_overlap = 0.25
        self._chw = np.zeros((3, DETECT_SIZE, DETECT_SIZE), dtype=np.float32)
        self._lb_canvas = np.full((DETECT_SIZE, DETECT_SIZE, 3), 114, dtype=np.uint8)

    def _preprocess_rgb(self, rgb: np.ndarray) -> Tuple[np.ndarray, float, int, int, int, int]:
        orig_h, orig_w = rgb.shape[:2]
        lb, scale, pad_x, pad_y = letterbox(rgb, canvas=self._lb_canvas)
        self._chw[:] = np.transpose(lb.astype(np.float32) * (1.0 / 255.0), (2, 0, 1))
        return self._chw, scale, pad_x, pad_y, orig_w, orig_h

    def _infer_rgb_crop(self, rgb: np.ndarray) -> Optional[Tuple]:
        chw, scale, pad_x, pad_y, orig_w, orig_h = self._preprocess_rgb(rgb)
        data = self.net.nn_inference(
            [chw],
            platform="ONNX",
            reorder="2 1 0",
            output_tensor=3,
            output_format=output_format.OUT_FORMAT_FLOAT32,
        )
        heads = reshape_heads(data)
        det = yolo_post_process(heads)
        if det is None:
            return None
        boxes, scores, classes = det
        boxes_px = unletterbox_boxes(boxes, scale, pad_x, pad_y, orig_w, orig_h)
        return filter_face_geometry(boxes_px, scores, classes, orig_w, orig_h)

    def _infer_tiled(self, bgr: np.ndarray) -> Optional[Tuple]:
        h, w = bgr.shape[:2]
        cols = 2 if self.tiles >= 2 else 1
        rows = 2 if self.tiles >= 4 else 1
        base_tw, base_th = w / cols, h / rows
        pad_w = int(base_tw * self.tile_overlap)
        pad_h = int(base_th * self.tile_overlap)
        parts: List[Optional[Tuple]] = []

        for row in range(rows):
            for col in range(cols):
                x0 = int(col * base_tw)
                y0 = int(row * base_th)
                x1 = int((col + 1) * base_tw) if col < cols - 1 else w
                y1 = int((row + 1) * base_th) if row < rows - 1 else h
                if self.tile_overlap > 0:
                    x0 = max(0, x0 - pad_w)
                    y0 = max(0, y0 - pad_h)
                    x1 = min(w, x1 + pad_w)
                    y1 = min(h, y1 + pad_h)
                tile_bgr = bgr[y0:y1, x0:x1]
                tile_rgb = cv.cvtColor(tile_bgr, cv.COLOR_BGR2RGB)
                det = self._infer_rgb_crop(tile_rgb)
                if det is None:
                    continue
                b, s, c = det
                b = b.copy()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                parts.append((b, s, c))

        boxes_list, scores_list, classes_list = [], [], []
        for part in parts:
            if part is None:
                continue
            b, s, c = part
            boxes_list.append(b)
            scores_list.append(s)
            classes_list.append(c)
        if not boxes_list:
            return None
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        classes = np.concatenate(classes_list, axis=0)
        keep = nms_boxes(boxes, scores)
        if len(keep) == 0:
            return None
        return boxes[keep], scores[keep], classes[keep]

    def infer_bgr(self, bgr: np.ndarray) -> Optional[Tuple]:
        if self.tiles <= 1:
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            return self._infer_rgb_crop(rgb)
        return self._infer_tiled(bgr)


class InferenceWorker(threading.Thread):
    """Background NPU worker: always processes the newest frame, drops backlog."""

    def __init__(self, detector: FaceNPU):
        super().__init__(daemon=True)
        self.detector = detector
        self._lock = threading.Lock()
        self._pending: Optional[np.ndarray] = None
        self._wake = threading.Event()
        self._run = True
        self.det: Optional[Tuple] = None
        self.infer_count = 0
        self.infer_fps = 0.0

    def submit(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._pending = bgr
        self._wake.set()

    def stop(self) -> None:
        self._run = False
        self._wake.set()

    def run(self) -> None:
        fps_t = time.perf_counter()
        fps_n = 0
        while self._run:
            if not self._wake.wait(timeout=0.05):
                continue
            self._wake.clear()

            frame = None
            while True:
                with self._lock:
                    if self._pending is not None:
                        frame = self._pending
                        self._pending = None
                    else:
                        break

            if frame is None:
                continue

            det = self.detector.infer_bgr(frame)
            fps_n += 1
            with self._lock:
                self.det = det
                self.infer_count += 1

            now = time.perf_counter()
            if now - fps_t >= 1.0:
                self.infer_fps = fps_n / max(now - fps_t, 1e-6)
                fps_n = 0
                fps_t = now

    def get_det(self) -> Optional[Tuple]:
        with self._lock:
            return self.det


class RTSPReader:
    """Background RTSP decode; main thread runs NPU."""

    def __init__(self, url: str, grab_drain: int = 1):
        self.url = url
        self.grab_drain = max(0, int(grab_drain))
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._run = True
        self._cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv.VideoCapture(self.url, cv.CAP_FFMPEG)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_latest(self):
        if self.grab_drain <= 0:
            return self._cap.read()
        for _ in range(self.grab_drain):
            if not self._cap.grab():
                return False, None
        return self._cap.retrieve()

    def _loop(self):
        fails = 0
        while self._run:
            ret, frame = self._read_latest()
            if not ret or frame is None:
                fails += 1
                try:
                    self._cap.release()
                except Exception:
                    pass
                time.sleep(min(fails, 3))
                self._cap = self._open()
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame, self._seq

    def read_copy(self):
        ok, fr, seq = self.read()
        if not ok or fr is None:
            return False, None, -1
        return True, fr.copy(), seq

    def release(self):
        self._run = False
        try:
            self._cap.release()
        except Exception:
            pass


def open_capture(cap_type: str, device: str, grab_drain: int):
    if cap_type == "rtsp":
        url = device.strip()
        if not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            sys.exit(f"RTSP --device must be a URL, got: {device!r}")
        reader = RTSPReader(url, grab_drain=grab_drain)
        t0 = time.time()
        while time.time() - t0 < 15.0:
            ok, fr, _ = reader.read()
            if ok and fr is not None:
                return reader, "rtsp"
            time.sleep(0.05)
        reader.release()
        sys.exit(f"No frame from RTSP within 15s: {url[:80]}")

    if cap_type == "usb":
        cap = cv.VideoCapture(int(device), cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        return cap, "usb"

    if cap_type == "mipi":
        pipeline = (
            f"v4l2src device=/dev/video{device} io-mode=dmabuf ! "
            "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
            "queue ! videoconvert ! appsink"
        )
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        return cap, "mipi"

    sys.exit(f"Unsupported --type {cap_type!r}. Use: rtsp, usb, mipi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection (NPU) on RTSP / USB / MIPI")
    parser.add_argument("--library", required=True, help="Path to libnn_*.so")
    parser.add_argument("--model", required=True, help="Path to *.nb model")
    parser.add_argument(
        "--type",
        required=True,
        choices=["rtsp", "usb", "mipi"],
        help="Input: rtsp URL, usb index, or mipi video device number",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="rtsp://... URL, usb camera index (0), or mipi /dev/videoN index",
    )
    parser.add_argument("--level", type=int, default=0, help="asnn log level 0/1/2")
    parser.add_argument("--face-thresh", type=float, default=DEFAULT_FACE_THRESH)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS)
    parser.add_argument(
        "--tiles",
        type=int,
        choices=[0, 1, 2, 4],
        default=1,
        help="1=fast full-frame (default), 4=2x2 tiles slower but better for tiny faces",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument(
        "--rtsp-grab-drain",
        type=int,
        default=1,
        help="RTSP frames to skip per read (1=low latency, 3=smoother)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Block display on each NPU infer (slower, no pipelining)",
    )
    parser.add_argument(
        "--max-display-fps",
        type=float,
        default=30.0,
        help="Cap preview refresh rate (0=unlimited)",
    )
    parser.add_argument("--stats", action="store_true", help="Print infer FPS every 2s")
    parser.add_argument("--verbose", action="store_true", help="Print each detection")
    args = parser.parse_args()

    if not os.path.isfile(args.library):
        sys.exit(f"Library not found: {args.library}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    _cfg.face_thresh = float(args.face_thresh)
    _cfg.nms = float(args.nms)

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    detector = FaceNPU(args.library, args.model, args.level)
    detector.tiles = 0 if args.tiles <= 1 else int(args.tiles)
    detector.tile_overlap = float(args.tile_overlap)

    cap_or_reader, kind = open_capture(args.type, args.device, args.rtsp_grab_drain)

    worker: Optional[InferenceWorker] = None
    if not args.sync:
        worker = InferenceWorker(detector)
        worker.start()

    stats_t = time.time()
    disp_n = 0
    t_disp = time.perf_counter()
    last_submit_seq = -1
    display_interval = 1.0 / max(args.max_display_fps, 1.0) if args.max_display_fps > 0 else 0.0
    next_display_t = 0.0

    mode = "sync" if args.sync else "realtime-async"
    print(
        f"Running type={args.type} mode={mode} tiles={detector.tiles} "
        f"thresh={_cfg.face_thresh} nms={_cfg.nms} (NPU inference, RGB 640 letterbox)"
    )

    try:
        while True:
            if kind == "rtsp":
                ret, frame, seq = cap_or_reader.read_copy()
            else:
                ret, frame = cap_or_reader.read()
                seq = -1

            if not ret or frame is None:
                if kind == "rtsp":
                    time.sleep(0.005)
                    continue
                break

            if worker is not None:
                if seq >= 0:
                    if seq != last_submit_seq:
                        worker.submit(frame)
                        last_submit_seq = seq
                else:
                    worker.submit(frame.copy())
                det = worker.get_det()
                infer_fps = worker.infer_fps
            else:
                det = detector.infer_bgr(frame)
                infer_fps = 0.0

            vis = frame
            if det is not None:
                boxes_px, scores, classes = det
                vis = frame.copy()
                draw(vis, boxes_px, scores, classes, verbose=args.verbose)

            now = time.perf_counter()
            if display_interval > 0 and now < next_display_t:
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            next_display_t = now + display_interval

            if infer_fps > 0:
                cv.putText(
                    vis,
                    f"infer {infer_fps:.1f} fps",
                    (12, 28),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv.imshow("face", vis)
            disp_n += 1
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

            if args.stats and (time.time() - stats_t) >= 2.0:
                elapsed = time.perf_counter() - t_disp
                disp_fps = disp_n / max(elapsed, 1e-6)
                if worker is not None:
                    print(f"display_fps={disp_fps:.1f} infer_fps={worker.infer_fps:.1f}")
                else:
                    print(f"display_fps={disp_fps:.1f}")
                stats_t = time.time()
                disp_n = 0
                t_disp = time.perf_counter()

    finally:
        if worker is not None:
            worker.stop()
        if kind == "rtsp":
            cap_or_reader.release()
        else:
            cap_or_reader.release()
        cv.destroyAllWindows()
