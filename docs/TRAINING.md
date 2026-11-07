# Training guide

This guide explains how to build or retrain each model of the pipeline for a production ID-card reader.
It has two parts:

1. **[What the released weights were trained with](#1-released-weights)**: facts recorded in the checkpoints.
2. **[Recommended procedure](#2-recommended-procedure)**: how to collect data, label it, train, evaluate and ship
   a new version. Use it as the plan for a retraining project. It is a guide, not a log of past experiments.

The training images are photos of real ID cards. They are personal data and are not published.

---

## 1. Released weights

| | Card corners | Card fields | Document keypoints |
|---|---|---|---|
| File | `card_corners_yolov8s.pt` | `card_fields_yolov8s.pt` | `document_keypoints_yolov8n_pose.pt` |
| Base model | `yolov8s.pt` (COCO) | `yolov8s.pt` (COCO) | `yolov8n-pose.pt` (COCO keypoints) |
| Classes | 4 corners | `id`, `name`, `dob`, `hometown`, `face` | `paper` + 4 keypoints |
| Data | front side of CCCD photos, 400 images (300 train / 50 val / 50 test) | warped card crops | A4 paper document photos |
| Epochs / batch / imgsz | 100 / 16 / 640 | 200 / 16 / 640 | 300 / 32 / 640 |
| Augmentation | Ultralytics defaults (mosaic 1.0, **fliplr 0.5**) | defaults (fliplr 0.5) | mosaic 1.0, **fliplr 0** |
| Val mAP@50 / @50-95 | 0.843 / 0.388 | 0.991 / 0.877 | 0.993 / 0.992 (pose) |
| Ultralytics | 8.2.2 | 8.2.5 | 8.2.20 |

The OCR model is VietOCR `vgg_transformer`, used as released by its authors (no fine-tuning).

**Known issue:** horizontal flips (`fliplr=0.5`) are harmful for the corner detector. A flipped `top_left` corner
looks like a `top_right` one but keeps the `top_left` label, which explains most of its lower precision and recall.
Use `fliplr=0`, or keep flips with `flip_idx` in a pose model, which swaps the keypoints correctly.

---

## 2. Recommended procedure

### 2.1 Data collection

The model only works on conditions it has seen. Cover them on purpose:

| Factor | Cover |
|---|---|
| Devices | low-end and high-end Android phones, iPhones; front and back cameras |
| Capture | handheld, 0–45° tilt, card filling 20–90% of the frame, rotated 0/90/180/270° |
| Light | daylight, indoor warm light, flash glare on the laminate, shadows of the hand or phone |
| Background | desk, hand, patterned fabric, another card or paper in the frame |
| Quality | motion blur, defocus, JPEG compression from chat apps, screenshots |
| Negatives | no card, back side, other cards (bank, driving licence), to tune rejection |

Size: 1,000–2,000 front-side photos are a good target for the corner/field detectors.
Split **by person** (not by photo) into train / val / test (80/10/10) so the same card never appears in two splits.

Privacy: collect with written consent, store encrypted, restrict access, and delete raw data when the project ends.
Use synthetic or specimen cards for anything shown publicly, like `examples/`.

### 2.2 Labelling (Label Studio)

| Model | Label | Notes |
|---|---|---|
| Corner detector | 4 small boxes centred on each physical corner | consistent box size (~5% of card width) |
| Corner pose model (recommended) | 1 box around the card + 4 keypoints in TL, TR, BR, BL order **relative to the card's text** | the order defines orientation, so rotated photos are labelled by the card's own top-left |
| Field detector | one box per **text line** on the warped 856×540 card | multi-line fields (place of origin, residence) get one box per line |
| OCR | line crop + exact transcription, with diacritics | crops from the field detector; double-check digits |

Export as YOLO format. For pose models set `kpt_shape: [4, 2]` and `flip_idx: [1, 0, 3, 2]`.

### 2.3 Training

```bash
# corner detector (fix: no horizontal flips)
yolo detect train model=yolov8s.pt data=corners.yaml imgsz=640 epochs=150 batch=16 fliplr=0 degrees=10 perspective=0.0005

# lighter alternative: card corners as keypoints
yolo pose train model=yolov8n-pose.pt data=card_pose.yaml imgsz=640 epochs=300 batch=32

# field detector, on warped cards produced by stage 1
yolo detect train model=yolov8s.pt data=fields.yaml imgsz=640 epochs=200 batch=16 fliplr=0 mosaic=0.5
```

Field crops should come from the **cropper's own output**, not from hand-cropped cards, so the detector learns the
small misalignments it will see in production.

OCR fine-tuning (VietOCR trainer) is worth it once there are ~5,000 labelled line crops. Start from the released
`vgg_transformer` weights, use a low learning rate, and add the CCCD fonts' typical confusions
(`0/O`, `1/l`, `5/S`, `8/B`, tone marks) through augmentation.

### 2.4 Evaluation

Evaluate each stage and the whole pipeline on the **test** split only:

| Stage | Metric | Why |
|---|---|---|
| Cropper | corner error in % of the card diagonal (mean, 95th percentile); crop success rate | detection mAP does not say how good the warp is |
| Field detector | mAP@50-95 per class; missed-field rate | a missed field is unrecoverable downstream |
| OCR | character error rate (CER) per field | separates OCR errors from detection errors |
| End to end | **exact-match rate per field** and for the whole card; rejection rate on negatives | what the business sees |

Report latency (p50/p95) on the target hardware (server CPU, GPU, or phone) together with accuracy.

### 2.5 Release checklist

1. Run `pytest` (unit + specimen end-to-end tests).
2. Compare the end-to-end exact-match rate with the previous release on the same test split.
3. Upload the new weights to a new GitHub release and update `RELEASE` and the SHA-256 values in
   `src/ekyc/weights.py`.
4. Bump the version in `pyproject.toml` and `src/ekyc/__init__.py`.
