"""Text-line recognition with VietOCR (VGG19-BN + Transformer), plus optional line clean-up."""
from __future__ import annotations

from importlib.resources import files

import cv2
import numpy as np
import yaml
from PIL import Image


def enhance_line(bgr: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """Upscale, remove uneven illumination/shadow and stretch contrast of a text-line crop.

    Background is estimated with dilation + median blur and subtracted, then the histogram is
    stretched (1% clipped). Returns a 3-channel image ready for VietOCR.
    """
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    flat = 255 - cv2.absdiff(gray, background)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    lo, hi = np.percentile(flat, (0.5, 99.5))
    if hi - lo > 1:
        flat = np.clip((flat.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


class TextRecognizer:
    def __init__(self, weights: str, device: str = "cpu", enhance: bool = False, beamsearch: bool = False):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = yaml.safe_load(files("ekyc.configs").joinpath("vietocr_vgg_transformer.yml").read_text(encoding="utf-8"))
        cfg.update(weights=weights, device=device)
        cfg["predictor"]["beamsearch"] = beamsearch
        self.predictor = Predictor(Cfg(cfg))
        self.enhance = enhance

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognise a batch of BGR line crops -> [(text, probability)]."""
        if not crops:
            return []
        imgs = [Image.fromarray(cv2.cvtColor(enhance_line(c) if self.enhance else c, cv2.COLOR_BGR2RGB)) for c in crops]
        texts, probs = self.predictor.predict_batch(imgs, return_prob=True)
        return [(t.strip(), float(p)) for t, p in zip(texts, probs)]
