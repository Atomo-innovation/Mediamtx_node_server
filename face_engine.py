import base64
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import cv2 as cv
import numpy as np
from huggingface_hub import hf_hub_download

COSINE_THRESHOLD = 0.70
NORML2_THRESHOLD = 1.128
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"


class YuNet:
    def __init__(
        self,
        model_path: str,
        input_size=(320, 320),
        conf_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = 0,
        target_id: int = 0,
    ):
        self._model = cv.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            conf_threshold,
            nms_threshold,
            top_k,
            backend_id,
            target_id,
        )

    def set_input_size(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        faces = self._model.detect(image)
        return np.empty((0, 15), dtype=np.float32) if faces[1] is None else faces[1]


class SFace:
    def __init__(self, model_path: str, dis_type: int = 0, backend_id: int = 0, target_id: int = 0):
        self._model = cv.FaceRecognizerSF.create(model_path, "", backend_id, target_id)
        self._dis_type = dis_type

    def infer(self, image, face_data):
        aligned = self._model.alignCrop(image, face_data.astype(np.float32))
        return self._model.feature(aligned)

    def score(self, feat1, feat2) -> float:
        return float(self._model.match(feat1, feat2, self._dis_type))


class FaceEngine:
    def __init__(self, conf_threshold: float = 0.9, dis_type: int = 0):
        yunet_path = hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")
        sface_path = hf_hub_download("opencv/face_recognition_sface", SFACE_MODEL)
        self.dis_type = dis_type
        self.threshold = COSINE_THRESHOLD if dis_type == 0 else NORML2_THRESHOLD
        self.detector = YuNet(yunet_path, conf_threshold=conf_threshold)
        self.recognizer = SFace(sface_path, dis_type=dis_type)

    def detect_faces(self, image):
        self.detector.set_input_size((image.shape[1], image.shape[0]))
        return self.detector.infer(image)

    def extract_embedding(self, image, face_row=None):
        if face_row is None:
            faces = self.detect_faces(image)
            if faces.shape[0] == 0:
                return None, None
            face_row = faces[np.argmax(faces[:, -1])]
        embedding = self.recognizer.infer(image, face_row[:-1])
        return embedding, face_row

    def embedding_to_list(self, embedding):
        return embedding.flatten().astype(float).tolist()

    def embedding_from_list(self, values):
        return np.array(values, dtype=np.float32).reshape(1, -1)

    def match_best(self, probe_embedding, enrolled_users):
        best_user = None
        best_score = None

        for user in enrolled_users:
            for photo in user.get("photos", []):
                ref = self.embedding_from_list(photo["embedding"])
                score = self.recognizer.score(probe_embedding, ref)
                if best_score is None:
                    best_score = score
                    best_user = user
                elif self.dis_type == 0 and score > best_score:
                    best_score = score
                    best_user = user
                elif self.dis_type != 0 and score < best_score:
                    best_score = score
                    best_user = user

        if best_score is None:
            return None, None, False

        if self.dis_type == 0:
            matched = best_score >= self.threshold
        else:
            matched = best_score <= self.threshold

        return best_user, best_score, matched

    def crop_face_preview(self, image, face_row, padding=0.15):
        if image is None or image.size == 0:
            return None

        raw = np.asarray(face_row[:4], dtype=np.float32)
        if raw.shape[0] != 4 or not np.isfinite(raw).all():
            return None

        x, y, w, h = [float(v) for v in raw]
        if w <= 1 or h <= 1:
            return None

        # Use Python ints to avoid int32 overflow.
        xi = int(round(x))
        yi = int(round(y))
        wi = int(round(w))
        hi = int(round(h))

        pad_x = int(round(wi * padding))
        pad_y = int(round(hi * padding))

        x1 = max(0, xi - pad_x)
        y1 = max(0, yi - pad_y)
        x2 = min(int(image.shape[1]), xi + wi + pad_x)
        y2 = min(int(image.shape[0]), yi + hi + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop.copy()

    def infer_pose(self, face_row):
        raw = np.asarray(face_row, dtype=np.float32)
        if raw.shape[0] < 9 or not np.isfinite(raw[:9]).all():
            return "front"

        x, y, w, h = [float(v) for v in raw[:4]]
        nose_x = float(raw[8])
        center_x = x + w / 2.0
        offset = abs(nose_x - center_x) / max(w, 1.0)
        if offset < 0.12:
            return "front"
        if nose_x < center_x:
            return "left"
        return "right"

    def resize_for_preview(self, frame, max_width=960):
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame
        scale = max_width / width
        return cv.resize(frame, (int(width * scale), int(height * scale)))

    def encode_image_jpeg(self, image, quality=80):
        if image is None or getattr(image, "size", 0) == 0:
            return None
        ok, encoded = cv.imencode(".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")


def ensure_rtsp_transport_tcp(rtsp_url: str) -> str:
    try:
        parsed = urlparse(rtsp_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "rtsp_transport" not in query:
            query["rtsp_transport"] = "tcp"
            parsed = parsed._replace(query=urlencode(query, doseq=True))
            return urlunparse(parsed)
    except Exception:
        pass
    return rtsp_url


def open_capture(source, force_tcp=False):
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv.VideoCapture(int(source))
    else:
        url = ensure_rtsp_transport_tcp(source) if force_tcp else source
        cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_image(path: str):
    image = cv.imread(path)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def enroll_image_file(engine: FaceEngine, image_path: str):
    image = load_image(image_path)
    embedding, face_row = engine.extract_embedding(image)
    if embedding is None:
        raise ValueError("No face found in image")
    preview = engine.crop_face_preview(image, face_row)
    return {
        "embedding": engine.embedding_to_list(embedding),
        "preview_b64": engine.encode_image_jpeg(preview),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract face embedding from an image")
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()

    engine = FaceEngine()
    result = enroll_image_file(engine, args.image)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
