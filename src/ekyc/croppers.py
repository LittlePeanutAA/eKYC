"""Card croppers: locate the card's 4 corners in a photo and warp it to a flat, upright image.

Two interchangeable strategies, both built on Ultralytics YOLOv8:

* ``CornerBoxCropper`` – object detector with one class per corner (top_left, top_right, ...).
  The card quad is formed from the box centres. One missing corner is inferred.
* ``KeypointCropper`` – pose model that regresses the 4 corners of the document directly
  (one ``paper`` instance with 4 keypoints ordered TL, TR, BR, BL).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CARD_SIZE, box_center, complete_quad, is_plausible_quad, warp_quad


@dataclass
class CropResult:
    image: np.ndarray  # warped card, BGR
    quad: np.ndarray  # (4, 2) TL, TR, BR, BL in the source image
    score: float  # mean confidence of the corners used
    inferred_corner: bool = False


class _YoloCropper:
    def __init__(self, weights: str, conf: float, device: str, size: tuple[int, int] = CARD_SIZE):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.device, self.size = conf, device, size

    def _predict(self, image: np.ndarray):
        return self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]

    def _finish(self, image, quad, score, inferred=False) -> CropResult | None:
        if quad is None or not is_plausible_quad(quad, image.shape):
            return None
        return CropResult(warp_quad(image, quad, self.size), quad, score, inferred)


class CornerBoxCropper(_YoloCropper):
    def __init__(self, weights: str, conf: float = 0.25, device: str = "cpu", **kw):
        super().__init__(weights, conf, device, **kw)

    def __call__(self, image: np.ndarray) -> CropResult | None:
        r = self._predict(image)
        best: dict[str, tuple[float, np.ndarray]] = {}
        for xyxy, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            name = r.names[int(cls)]
            if name not in best or conf > best[name][0]:  # keep the most confident box per corner
                best[name] = (float(conf), box_center(xyxy))
        quad = complete_quad({k: v[1] for k, v in best.items()})
        score = float(np.mean([v[0] for v in best.values()])) if best else 0.0
        return self._finish(image, quad, score, inferred=len(best) == 3)


class KeypointCropper(_YoloCropper):
    def __init__(self, weights: str, conf: float = 0.25, device: str = "cpu", **kw):
        super().__init__(weights, conf, device, **kw)

    def __call__(self, image: np.ndarray) -> CropResult | None:
        r = self._predict(image)
        if not len(r.boxes):
            return None
        i = int(r.boxes.conf.argmax())
        quad = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        return self._finish(image, quad, float(r.boxes.conf[i]))


CROPPERS = {"corners": CornerBoxCropper, "keypoints": KeypointCropper}
