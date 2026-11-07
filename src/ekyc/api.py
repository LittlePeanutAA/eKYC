"""HTTP API (``pip install ekyc[api]``; run with ``uvicorn ekyc.api:app``).

Environment: ``EKYC_CROPPER`` (corners|keypoints), ``EKYC_DEVICE`` (cpu|cuda), ``EKYC_MAX_MB`` (upload limit).
Images are processed in memory and never written to disk.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from . import __version__
from .pipeline import EKYC, CardNotFoundError

MAX_BYTES = int(float(os.environ.get("EKYC_MAX_MB", 10)) * 1024 * 1024)


@lru_cache(maxsize=1)
def get_model() -> EKYC:
    return EKYC.from_pretrained(os.environ.get("EKYC_CROPPER", "corners"), os.environ.get("EKYC_DEVICE", "cpu"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()  # load the models once, before serving requests
    yield


app = FastAPI(title="eKYC – Vietnamese ID card extraction", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/id-card/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"image larger than {MAX_BYTES // 2**20} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "unsupported or corrupt image")
    try:
        result = await run_in_threadpool(get_model(), image)
    except CardNotFoundError as e:
        raise HTTPException(422, str(e)) from e
    return result.to_dict()
