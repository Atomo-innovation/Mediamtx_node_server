import json
import sys
import time
from pathlib import Path
from threading import Event, Thread
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_engine import FaceEngine, open_capture

DEFAULT_SERVER = "http://127.0.0.1:3000"
# Per-camera sampling interval for running detection/recognition.
FRAME_INTERVAL_SEC = 0.8
# Higher = smoother "LIVE" tiles (not true 30fps video).
PREVIEW_INTERVAL_SEC = 0.6
DETECTION_COOLDOWN_SEC = 2.5


def http_get_json(url: str):
    with request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class CameraProcessor:
    def __init__(self, server_url: str, camera: dict, stop_event: Event):
        # IMPORTANT: each camera gets its own OpenCV model instances.
        # Sharing FaceEngine across threads can crash / corrupt state.
        self.engine = FaceEngine()
        self.server_url = server_url.rstrip("/")
        self.camera = camera
        self.stop_event = stop_event
        self.last_detection_at = 0.0
        self.last_preview_at = 0.0

    def post_status(self, connected: bool):
        try:
            http_post_json(
                f"{self.server_url}/api/cameras/{self.camera['id']}/status",
                {"connected": connected},
            )
        except error.URLError:
            pass

    def upload_preview(self, frame):
        preview = self.engine.resize_for_preview(frame)
        preview_b64 = self.engine.encode_image_jpeg(preview, quality=70)
        if not preview_b64:
            return
        try:
            http_post_json(
                f"{self.server_url}/api/cameras/{self.camera['id']}/preview",
                {"previewB64": preview_b64},
            )
        except error.URLError as exc:
            print(f"[worker] failed to upload preview: {exc}", file=sys.stderr)

    def process_once(self, frame):
        faces = self.engine.detect_faces(frame)
        if faces.shape[0] == 0:
            return

        now = time.time()
        if now - self.last_detection_at < DETECTION_COOLDOWN_SEC:
            return

        for face_row in faces:
            embedding, _ = self.engine.extract_embedding(frame, face_row)
            if embedding is None:
                continue

            preview = self.engine.crop_face_preview(frame, face_row)
            if preview is None:
                continue
            preview_b64 = self.engine.encode_image_jpeg(preview)
            if not preview_b64:
                continue
            pose = self.engine.infer_pose(face_row)

            payload = {
                "cameraId": self.camera["id"],
                "cameraName": self.camera["name"],
                "embedding": self.engine.embedding_to_list(embedding),
                "snapshotB64": preview_b64,
                "pose": pose,
            }

            try:
                http_post_json(f"{self.server_url}/api/detections", payload)
                self.last_detection_at = now
            except error.URLError as exc:
                print(f"[worker] failed to post detection: {exc}", file=sys.stderr)
            break

    def run(self):
        source = self.camera["source"]
        force_tcp = bool(self.camera.get("forceTcp"))

        while not self.stop_event.is_set():
            cap = open_capture(source, force_tcp=force_tcp)
            if not cap.isOpened():
                print(f"[worker] cannot open camera {self.camera['name']}", file=sys.stderr)
                self.post_status(False)
                time.sleep(2)
                continue

            self.post_status(True)
            last_frame_at = 0.0

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.post_status(False)
                    break

                now = time.time()
                if now - self.last_preview_at >= PREVIEW_INTERVAL_SEC:
                    self.upload_preview(frame)
                    self.last_preview_at = now

                if now - last_frame_at < FRAME_INTERVAL_SEC:
                    continue
                last_frame_at = now

                try:
                    self.process_once(frame)
                except Exception as exc:
                    print(f"[worker] processing error on {self.camera['name']}: {exc}", file=sys.stderr)

            cap.release()
            if not self.stop_event.is_set():
                time.sleep(1)


class WorkerManager:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.stop_event = Event()
        self.threads = {}
        self.config_version = None

    def fetch_config(self):
        return http_get_json(f"{self.server_url}/api/internal/worker-config")

    def sync_cameras(self, cameras):
        active_ids = {camera["id"] for camera in cameras if camera.get("enabled")}

        for camera_id, thread in list(self.threads.items()):
            if camera_id not in active_ids:
                thread["stop"].set()
                thread["thread"].join(timeout=2)
                del self.threads[camera_id]

        for camera in cameras:
            if not camera.get("enabled"):
                continue
            if camera["id"] in self.threads:
                continue

            stop = Event()
            processor = CameraProcessor(self.server_url, camera, stop)
            thread = Thread(target=processor.run, daemon=True)
            thread.start()
            self.threads[camera["id"]] = {"thread": thread, "stop": stop}

    def run(self):
        print(f"[worker] connected to {self.server_url}", flush=True)
        print("[worker] using face_recognition_sface_2021dec.onnx", flush=True)
        while not self.stop_event.is_set():
            try:
                config = self.fetch_config()
                version = config.get("version")
                if version != self.config_version:
                    self.config_version = version
                    self.sync_cameras(config.get("cameras", []))
                    print(f"[worker] synced config version {version}", flush=True)
            except Exception as exc:
                print(f"[worker] waiting for server: {exc}", file=sys.stderr)

            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

        for entry in self.threads.values():
            entry["stop"].set()


def main():
    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER
    manager = WorkerManager(server_url)
    try:
        manager.run()
    except KeyboardInterrupt:
        manager.stop_event.set()


if __name__ == "__main__":
    main()
