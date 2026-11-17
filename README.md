# Garbage Detection — YOLO11 dataset pipeline

This project turns the supplied JSONL manifest into a standard Ultralytics YOLO dataset, audits it, and trains a YOLO11 detector.

## Important findings in the supplied manifest

- The file contains one dataset metadata record and 517 image records.
- The final line is truncated/invalid JSON and is reported and skipped.
- The original records declare `train` and `test`; no `val` split is supplied.
- Seven image records have no `annotations.boxes` field. By default these are skipped because missing annotations do **not** prove an image is a negative sample.
- One URL is duplicated and is skipped to avoid downloading the same image twice.
- Signed image URLs may expire. Run the downloader as soon as possible. If they have expired, re-export the dataset/manifest from its source.

## 1. Setup

```bash
cd garbage_yolo11_pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch/Ultralytics installation fails due to a platform-specific wheel, follow the current PyTorch install selector for your machine, then install `ultralytics`.

## 2. Prepare the dataset

The supplied manifest is already at `data/manifest.jsonl`.

```bash
python scripts/prepare_dataset.py \
  --manifest data/manifest.jsonl \
  --output dataset \
  --workers 6 \
  --val-fraction 0.10
```

The script:
- Parses JSONL line by line and records malformed lines.
- Validates box shape/ranges and split names.
- Skips records with missing annotations by default.
- Deduplicates identical URLs.
- Preserves the original test split.
- Creates a deterministic validation split from the original training split.
- Downloads concurrently with retries and integrity checks.
- Writes YOLO label text files and `dataset/data.yaml`.
- Produces `dataset/download_report.json`.

If a download fails, run the exact same command again. Valid existing images are checked and reused; failed downloads are retried. To intentionally redownload all images, add `--overwrite`.

**Only use `--missing-annotations empty` if you have verified those seven images are truly negative images (no garbage objects).** Otherwise keep the default `skip`.

## 3. Audit the result

```bash
python scripts/audit_dataset.py --root dataset
```

Do not train until the audit passes and you have reviewed `dataset/download_report.json`.

## 4. Train YOLO11

Start with the nano model because laptop CPU training can be slow:

```bash
python scripts/train.py \
  --data dataset/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --workers 2 \
  --device auto
```

`auto` lets Ultralytics select an available accelerator. On a CPU-only laptop, use `--device cpu`; for a CUDA GPU, use `--device 0`.

Training outputs are saved under `runs/detect/garbage-yolo11n/`. The best checkpoint is normally `runs/detect/garbage-yolo11n/weights/best.pt`.

## 5. Evaluate the best checkpoint

```bash
yolo detect val \
  model=runs/detect/garbage-yolo11n/weights/best.pt \
  data=dataset/data.yaml \
  split=test \
  imgsz=640
```

The test split is held out from training and validation. Use it for final evaluation, not repeated hyperparameter tuning.

## 6. Run inference on an image

```bash
yolo detect predict \
  model=runs/detect/garbage-yolo11n/weights/best.pt \
  source=/path/to/cctv-frame.jpg \
  conf=0.25
```

Predictions are written under `runs/detect/predict/`.

## Dataset layout

```text
dataset/
├── data.yaml
├── download_report.json
├── source_manifest.jsonl
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Practical notes

- The manifest's declared dataset size is approximately 1.42 GB; ensure you have several GB free for images, labels, model weights, and training outputs.
- The original metadata advertises a single class: `garbage` (class ID 0).
- Keep the original `test` split untouched.
- If images look incorrectly labeled, inspect samples with plotted boxes before training. A model cannot compensate for incorrect labels.
- Do not commit `data/manifest.jsonl` to a public repository: its signed URLs may grant temporary access to the images.
