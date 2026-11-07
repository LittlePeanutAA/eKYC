"""End-to-end extraction: photo -> cropped card -> field boxes -> OCR -> validated fields."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .croppers import CROPPERS, CropResult
from .fields import FieldDetection, FieldDetector
from .ocr import TextRecognizer
from .postprocess import normalize
from .weights import get_weights

CROPPER_WEIGHTS = {"corners": "card_corners", "keypoints": "document_keypoints"}
TEXT_FIELDS = ("id", "name", "dob", "hometown")


class CardNotFoundError(RuntimeError):
    pass


@dataclass
class FieldResult:
    text: str
    raw: str
    confidence: float  # mean OCR probability over the field's lines
    boxes: list[list[int]]  # xyxy on the cropped card


@dataclass
class ExtractionResult:
    fields: dict[str, FieldResult]
    face_box: list[int] | None
    card_quad: list[list[float]] | None  # TL, TR, BR, BL in the input image; None if not cropped
    crop_score: float
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    card_image: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("card_image")
        return d


class EKYC:
    """Vietnamese ID-card (CCCD) information extractor.

    >>> ekyc = EKYC.from_pretrained(cropper="corners")
    >>> result = ekyc(cv2.imread("card.jpg"))
    >>> result.fields["name"].text
    """

    def __init__(self, cropper, detector: FieldDetector, recognizer: TextRecognizer, fallback_full_image: bool = False):
        self.cropper, self.detector, self.recognizer = cropper, detector, recognizer
        self.fallback_full_image = fallback_full_image

    @classmethod
    def from_pretrained(cls, cropper: str = "corners", device: str = "cpu", cropper_weights: str | None = None,
                        enhance_ocr: bool = False, **kw) -> "EKYC":
        if cropper not in CROPPERS:
            raise ValueError(f"cropper must be one of {sorted(CROPPERS)}")
        return cls(
            CROPPERS[cropper](cropper_weights or get_weights(CROPPER_WEIGHTS[cropper]), device=device),
            FieldDetector(get_weights("card_fields"), device=device),
            TextRecognizer(get_weights("vietocr"), device=device, enhance=enhance_ocr),
            **kw,
        )

    def __call__(self, image: np.ndarray) -> ExtractionResult:
        if image is None or image.ndim != 3:
            raise ValueError("expected a BGR image (H, W, 3)")
        t = {}
        t0 = time.perf_counter()
        crop: CropResult | None = self.cropper(image)
        t["crop"] = (time.perf_counter() - t0) * 1000
        warnings = []
        if crop is None:
            if not self.fallback_full_image:
                raise CardNotFoundError("could not locate the 4 corners of the card")
            warnings.append("card corners not found; using the full image")
        elif crop.inferred_corner:
            warnings.append("one card corner was inferred from the other three")
        card = crop.image if crop else image

        t0 = time.perf_counter()
        det: FieldDetection = self.detector(card)
        t["detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        order = [(name, b) for name in TEXT_FIELDS for b in det.lines.get(name, [])]
        crops = [card[b.xyxy[1]:b.xyxy[3], b.xyxy[0]:b.xyxy[2]] for _, b in order]
        texts = self.recognizer(crops)
        t["ocr"] = (time.perf_counter() - t0) * 1000

        fields: dict[str, FieldResult] = {}
        for name in TEXT_FIELDS:
            lines = [(b, txt) for (n, b), txt in zip(order, texts) if n == name]
            if not lines:
                warnings.append(f"{name}: not detected")
                continue
            raw = ", ".join(txt for _, (txt, _) in lines)
            text, w = normalize(name, raw)
            warnings += w
            fields[name] = FieldResult(text, raw, float(np.mean([p for _, (_, p) in lines])),
                                       [b.xyxy.tolist() for b, _ in lines])

        face = det.lines.get("face")
        t["total"] = sum(t.values())
        return ExtractionResult(
            fields=fields,
            face_box=face[0].xyxy.tolist() if face else None,
            card_quad=crop.quad.round(1).tolist() if crop else None,
            crop_score=round(crop.score, 4) if crop else 0.0,
            warnings=warnings,
            timings_ms={k: round(v, 1) for k, v in t.items()},
            card_image=card,
        )


def draw(result: ExtractionResult, image: np.ndarray | None = None) -> np.ndarray:
    """Visualise the result: card quad on the input (if given) beside the annotated crop."""
    card = result.card_image.copy()
    for name, f in result.fields.items():
        for x1, y1, x2, y2 in f.boxes:
            cv2.rectangle(card, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(card, name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 1, cv2.LINE_AA)
    if result.face_box:
        x1, y1, x2, y2 = result.face_box
        cv2.rectangle(card, (x1, y1), (x2, y2), (200, 120, 0), 2)
    if image is None or result.card_quad is None:
        return card
    src = image.copy()
    cv2.polylines(src, [np.array(result.card_quad, np.int32)], True, (0, 200, 0), max(2, src.shape[1] // 300))
    src = cv2.resize(src, (int(src.shape[1] * card.shape[0] / src.shape[0]), card.shape[0]))
    return np.hstack([src, card])
