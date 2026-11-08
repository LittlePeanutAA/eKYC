"""Command line: ``ekyc card.jpg [--cropper keypoints] [--vis out.jpg [--redact]]``."""
from __future__ import annotations

import argparse
import json
import sys

import cv2

from .pipeline import EKYC, CardNotFoundError, draw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ekyc", description="Extract information from Vietnamese ID card photos.")
    p.add_argument("images", nargs="+", help="input image path(s)")
    p.add_argument("--cropper", choices=["corners", "keypoints"], default="corners")
    p.add_argument("--cropper-weights", help="custom weights for the cropper")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, mps")
    p.add_argument("--enhance-ocr", action="store_true", help="shadow removal + contrast stretch before OCR")
    p.add_argument("--fallback-full-image", action="store_true", help="run on the whole image if no card is found")
    p.add_argument("--vis", help="save a visualisation (one image) or a directory (several images)")
    p.add_argument("--redact", action="store_true", help="cover the fields and face with opaque boxes in --vis")
    args = p.parse_args(argv)

    ekyc = EKYC.from_pretrained(args.cropper, args.device, args.cropper_weights, args.enhance_ocr,
                                fallback_full_image=args.fallback_full_image)
    status = 0
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            print(json.dumps({"image": path, "error": "cannot read image"}, ensure_ascii=False))
            status = 1
            continue
        try:
            result = ekyc(image)
        except CardNotFoundError as e:
            print(json.dumps({"image": path, "error": str(e)}, ensure_ascii=False))
            status = 1
            continue
        print(json.dumps({"image": path, **result.to_dict()}, ensure_ascii=False, indent=2))
        if args.vis:
            out = args.vis if len(args.images) == 1 else f"{args.vis.rstrip('/')}/{path.rsplit('/', 1)[-1]}"
            cv2.imwrite(out, draw(result, image, redact=args.redact))
    return status


if __name__ == "__main__":
    sys.exit(main())
