"""Quadrilateral helpers: ordering corners, completing a missing corner, perspective warp."""
from __future__ import annotations

import cv2
import numpy as np

# ISO/IEC 7810 ID-1 (Vietnamese CCCD): 85.60 x 53.98 mm. 10 px/mm gives a sharp crop for OCR.
CARD_SIZE = (856, 540)
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 unordered points as TL, TR, BR, BL (image coordinates).

    Used when corner identity is unknown. It assumes the card is not rotated by more than
    45 degrees; detectors that predict named corners keep their own order instead.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def complete_quad(corners: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return TL, TR, BR, BL from named corners, inferring one missing corner as a parallelogram."""
    missing = [c for c in CORNERS if c not in corners]
    if len(missing) > 1:
        return None
    if missing:
        i = CORNERS.index(missing[0])
        prev, opp, nxt = (corners[CORNERS[(i + k) % 4]] for k in (3, 2, 1))
        corners = {**corners, missing[0]: prev + nxt - opp}
    return np.array([corners[c] for c in CORNERS], dtype=np.float32)


def is_plausible_quad(quad: np.ndarray, image_shape: tuple[int, ...], min_area_ratio: float = 0.02) -> bool:
    """Reject self-intersecting, non-convex or tiny quadrilaterals."""
    q = quad.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(q):
        return False
    return cv2.contourArea(q) >= min_area_ratio * image_shape[0] * image_shape[1]


def warp_quad(image: np.ndarray, quad: np.ndarray, size: tuple[int, int] = CARD_SIZE) -> np.ndarray:
    """Perspective-warp the TL, TR, BR, BL quadrilateral to a (width, height) rectangle."""
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, m, (w, h), flags=cv2.INTER_CUBIC)


def box_center(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
