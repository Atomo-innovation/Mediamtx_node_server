"""
recognizer.py
=============
NPUFaceRecognizer — wraps MobileFaceNet ONNX for embedding extraction.

CONFIRMED PREPROCESSING (from ONNX weight analysis + benchmark):
  • Input : 112×112  BGR  float32
  • Norm  : (pixel - 127.5) / 127.5   — DO NOT change
  • Layout: NCHW  (transpose 2,0,1 → np.newaxis)
  • Output: 512-d float32 embedding   → ALWAYS L2-normalise before storing

NPU backend: TIM-VX via OpenCV DNN (cv2.dnn.DNN_TARGET_NPU)
CPU fallback: ONNX Runtime CPUExecutionProvider
"""

import cv2
import numpy as np


class NPUFaceRecognizer:
    """
    Face embedding extractor for MobileFaceNet / ArcFace-R50 ONNX models.

    Parameters
    ----------
    model_path : str   path to .onnx file
    model_type : str   'mobilefacenet' | 'arcface_r50'
    use_npu    : bool  True → OpenCV TIM-VX backend, False → ONNX Runtime CPU
    """

    # ── per-model preprocessing configs ──────────────────────────────────
    _CONFIGS = {
        "mobilefacenet": {
            "input_size": 112,
            "input_name": "input.1",
            # (x - mean) / scale    applied per-channel
            # For this model: same value for all channels (confirmed by analysis)
            "mean":  127.5,
            "scale": 127.5,
            "channel_swap": False,  # keep BGR — model is BGR-tolerant, BGR is slightly better
        },
        "arcface_r50": {
            "input_size": 112,
            "input_name": "input.1",
            "mean":  127.5,
            "scale": 127.5,
            "channel_swap": False,
        },
    }

    def __init__(self, model_path: str,
                 model_type: str = "mobilefacenet",
                 use_npu: bool = True):

        self.cfg       = self._CONFIGS.get(model_type, self._CONFIGS["mobilefacenet"])
        self.use_npu   = use_npu
        self._net_cv   = None
        self._sess_ort = None

        if use_npu:
            self._load_cv(model_path)
        else:
            self._load_ort(model_path)

    # ── loaders ──────────────────────────────────────────────────────────

    def _load_cv(self, path: str):
        try:
            net = cv2.dnn.readNetFromONNX(path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_TIMVX)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)
            # Warm-up to lock in NPU memory
            dummy = np.zeros(
                (1, 3, self.cfg["input_size"], self.cfg["input_size"]),
                dtype=np.float32)
            net.setInput(dummy)
            net.forward()
            self._net_cv = net
        except Exception as e:
            raise RuntimeError(f"[Recognizer] CV2/TIM-VX load failed: {e}")

    def _load_ort(self, path: str):
        try:
            import onnxruntime as ort
            self._sess_ort = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"])
        except ImportError:
            raise RuntimeError(
                "[Recognizer] onnxruntime not installed. "
                "Run: pip install onnxruntime --break-system-packages")
        except Exception as e:
            raise RuntimeError(f"[Recognizer] ORT load failed: {e}")

    # ── preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        aligned_bgr : uint8 BGR image, already aligned to 112×112.
        Returns     : float32 NCHW blob (1, 3, H, W).

        Confirmed normalization:
            blob = (pixel - 127.5) / 127.5
        Channel order: BGR (no swap needed — model is weight-tolerant,
        BGR slightly outperforms RGB in separation tests).
        """
        img  = aligned_bgr.astype(np.float32)
        mean  = self.cfg["mean"]
        scale = self.cfg["scale"]

        if self.cfg["channel_swap"]:
            img = img[:, :, ::-1].copy()   # BGR → RGB

        img = (img - mean) / scale         # (x-127.5)/127.5 → [-1, 1]
        blob = img.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW
        return blob.astype(np.float32)

    # ── inference ─────────────────────────────────────────────────────────

    def get_embedding(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a 512-d L2-normalised face embedding.

        Parameters
        ----------
        aligned_bgr : uint8 BGR image (112×112 after align_face())

        Returns
        -------
        embedding   : float32 ndarray, shape (512,), L2-normalised.
                      Always L2-normalise here — cosine distance is only
                      meaningful on unit vectors.
        """
        blob = self._preprocess(aligned_bgr)

        if self._net_cv is not None:
            self._net_cv.setInput(blob)
            emb = self._net_cv.forward().flatten()
        elif self._sess_ort is not None:
            input_name = self.cfg["input_name"]
            out        = self._sess_ort.run(None, {input_name: blob})
            emb        = out[0].flatten()
        else:
            raise RuntimeError("[Recognizer] No backend loaded")

        # L2 normalise — MANDATORY for cosine distance
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        return emb.astype(np.float32)
