"""Field detection on the warped card: one YOLOv8 box per text line, grouped by field."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FieldBox:
    name: str
    xyxy: np.ndarray  # int, (x1, y1, x2, y2) on the warped card
    conf: float


@dataclass
class FieldDetection:
    lines: dict[str, list[FieldBox]] = field(default_factory=dict)  # top-to-bottom per field

    def boxes(self) -> list[FieldBox]:
        return [b for bs in self.lines.values() for b in bs]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class FieldDetector:
    # Fields that hold a single line; extra boxes are duplicates and only the best one is kept.
    SINGLE_LINE = {"id", "name", "dob", "face"}

    def __init__(self, weights: str, conf: float = 0.25, device: str = "cpu", iou_merge: float = 0.5):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.device, self.iou_merge = conf, device, iou_merge

    @property
    def names(self) -> list[str]:
        return list(self.model.names.values())

    def __call__(self, card: np.ndarray) -> FieldDetection:
        r = self.model.predict(card, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = [
            FieldBox(r.names[int(c)], xyxy.round().astype(int), float(s))
            for xyxy, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
        ]
        boxes.sort(key=lambda b: -b.conf)

        det = FieldDetection()
        for b in boxes:
            kept = det.lines.setdefault(b.name, [])
            if b.name in self.SINGLE_LINE and kept:
                continue
            if any(_iou(b.xyxy, k.xyxy) > self.iou_merge for k in kept):  # duplicate of a kept line
                continue
            kept.append(b)
        for kept in det.lines.values():
            kept.sort(key=lambda b: (b.xyxy[1], b.xyxy[0]))  # reading order
        return det
