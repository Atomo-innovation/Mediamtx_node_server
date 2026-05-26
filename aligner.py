"""
aligner.py
==========
5-point similarity-transform face aligner for MobileFaceNet / ArcFace.

CONFIRMED SETTINGS (from benchmark):
  • Target size : 112×112
  • Ref landmarks: ArcFace standard (not DeepFace, not VGG)
  • Transform   : estimateAffinePartial2D (LMEDS — robust to landmark noise)
  • Border mode : BORDER_REFLECT (no black edges on boundary faces)
  • Interpolation: INTER_LINEAR (best speed/quality for NPU board)
"""

import cv2
import numpy as np

# ArcFace standard 112×112 reference landmarks
# [left-eye, right-eye, nose-tip, left-mouth, right-mouth]
_REF_LMS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray,
               landmarks: np.ndarray,
               size: int = 112) -> np.ndarray:
    """
    Warp a face region into a canonical 112×112 crop aligned to ArcFace
    reference landmarks using a 5-point partial affine transform.

    Parameters
    ----------
    image     : BGR image (any resolution)
    landmarks : (5, 2) float32 array — [left-eye, right-eye, nose,
                                         left-mouth, right-mouth]
                in PIXEL coordinates of `image`
    size      : output size (default 112; must match model input)

    Returns
    -------
    aligned   : uint8 BGR image, shape (size, size, 3)
    """
    src = landmarks.astype(np.float32)

    # Scale ref landmarks if a non-standard size is requested
    if size != 112:
        ref = _REF_LMS_112 * (size / 112.0)
    else:
        ref = _REF_LMS_112

    # LMEDS: robust to 1-2 noisy landmark predictions
    M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)

    if M is None:
        # Fallback: centre-crop + resize (no warp)
        h, w = image.shape[:2]
        s    = min(h, w)
        y0   = (h - s) // 2
        x0   = (w - s) // 2
        return cv2.resize(image[y0:y0+s, x0:x0+s], (size, size),
                          interpolation=cv2.INTER_LINEAR)

    aligned = cv2.warpAffine(
        image, M, (size, size),
        flags       = cv2.INTER_LINEAR,
        borderMode  = cv2.BORDER_REFLECT,
    )
    return aligned
