# Garbage Detection — YOLO11 dataset pipeline

This project turns the supplied JSONL manifest into a standard Ultralytics YOLO dataset, audits it, and trains a YOLO11 detector.

## v2: garbage + person detector for CCTV (current workflow)

Two classes: `0 garbage`, `1 person`. Model: **YOLO11m** (most accurate size that trains on a 6 GB GPU
and still compiles for the Raspberry Pi AI HAT / Hailo). Training adds CCTV-style degradations
(blur, noise, JPEG, low resolution, grayscale night IR, burned-in timestamps).

On the GPU training machine:

```bash
git clone https://github.com/jayesh-aashman01/Garbage-Detector-Model.git && cd Garbage-Detector-Model
./run.sh deps                                   # venv + requirements (incl. albumentations)
cp .env.example .env && nano .env               # paste ROBOFLOW_API_KEY (file is git-ignored)
# copy the old dataset folder here as ./dataset  (its download links expired 2026-11-20)
#   e.g. rsync -a OLD_MACHINE:path/Garbage_Detection/dataset ./
./run.sh roboflow                               # -> datasets_raw/roboflow/
./run.sh merge                                  # -> dataset_merged/ + merge_report.json, audited
python scripts/cctv_augment.py --preview dataset_merged/images/val   # eyeball the augmentations
./run.sh train                                  # yolo11m, 100 epochs, CCTV_AUG=medium
./run.sh evaluate                               # mAP on val
./run.sh predict cctv_test/ some_image.jpg rtsp://user:pass@cam/stream   # images/videos/streams
```

Check `dataset_merged/merge_report.json` before training: class mapping, dropped duplicates,
auto-labelled person boxes, and per-split counts. If Roboflow uses a class name the script
does not recognise, it stops and tells you to add e.g. `--class-map "Plastic=garbage"`.

Useful overrides: `EPOCHS=150 CCTV_AUG=heavy BATCH=8 ./run.sh train`, `GD_DEBUG=1` for `[DEBUG]` logs.
`predict_media.py` raises a video alert only when garbage is seen in `--persist 3` of the last
`--window 5` sampled frames (`--sample-fps 2`), which suppresses one-frame false alarms.
Real-world test clips: `tests/cctv_test_clips.txt`. Tests: `.venv/bin/python -m unittest discover -s tests -t tests -v`.

### Before merging on the GPU machine: repair the old labels
The training machine's `dataset/` had 0-byte label files, so every `old_*` image lost its garbage boxes.
Rebuild them from the manifest. The script backs up the labels folder first:
```bash
python scripts/restore_old_labels.py              # dataset/labels + dataset_merged/labels old_* from data/manifest.jsonl
python scripts/preview_dataset.py --data dataset_merged --out dataset_preview   # check boxes; empty ones go to */empty_labels/
```

### Adding CCTV frames (person + garbage) — draft, review, merge
Frames from the same clip always stay in one split, so there is no leakage. The steps:
```
clips/  ──► ./run.sh cctv-frames clips/      1 fps, dedupe, Grounding DINO garbage + yolo11x person drafts
        ──► review in X-AnyLabeling            fix every frame (drafts are NOT ground truth)
        ──► clips_to_frames.py --mark-reviewed  only reviewed clips are merged
        ──► ./run.sh merge                      adds datasets_raw/cctv automatically
```
1. **Draft (GPU machine).** Put the clips in one folder, then run:
   ```bash
   pip install transformers          # Grounding DINO (one-time); CLIP for YOLO-World installs itself
   ./run.sh cctv-frames cctv_clips/  # CUDA if available; stops starting new clips at CCTV_MAX_FRAMES=1500
   ```
   This writes `datasets_raw/cctv/{images,labels}/{train,val}/`, `classes.txt`, `review.json` and the preview `dataset_preview/cctv/`.
   About 15% of clips go to val automatically (stable per clip name). Use `--assign CLIP=test` to override.
   Garbage drafts use confidence 0.40. A box covering more than 80% of the image is dropped, and so is a box more than 80% inside a larger garbage box (one box per pile).
   Re-running skips clips that are already done. Add more clips and run again.
2. **Review (any machine, X-AnyLabeling desktop app).** For each split folder:
   - Open the folder `datasets_raw/cctv/images/train`.
   - **Import:** `Import` → YOLO HBB, choose `datasets_raw/cctv/classes.txt`, then the folder `datasets_raw/cctv/labels/train`.
   - Fix the boxes. Keys: `D` next image, `A` previous, `R` rectangle, `Delete` removes a box.
     - Delete garbage boxes on vehicles, people and signs.
     - Use one box per pile, and separate boxes only for separate items.
     - Box every person.
     - Delete frames that are not CCTV (ads, studio shots): remove both the image and its `.txt`.
   - **Export:** `Export Annotations` → `Export YOLO Annotations` with the same `classes.txt`.
     The export path defaults to a `labels` folder next to the images. **Point it to `datasets_raw/cctv/labels/train`.**
   - Repeat for `val` (and `test`).
   - Check the result: `python scripts/preview_dataset.py --data datasets_raw/cctv --out dataset_preview/cctv`.
3. **Mark reviewed:** `python scripts/clips_to_frames.py --out datasets_raw/cctv --mark-reviewed CLIP1 CLIP2` (or `all`).
4. **Merge:** `./run.sh merge`. Extra flags pass through, e.g. `./run.sh merge --rf-drop-closeup 0.5 --rf-drop-portrait 0.4 --rf-cap-train 2500`.
   `merge_report.json` → `sources.cctv` lists the reviewed and skipped clips.

Later, for the Pi (Hailo-8/8L AI HAT, compile on Linux x86_64 with Hailo DFC installed):
`yolo export model=.../best.pt format=hailo name=hailo8l data=dataset_merged/data.yaml`
(see https://docs.ultralytics.com/integrations/hailo). Not automated yet.

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

Training uses CUDA automatically when an NVIDIA GPU is available (falls back to CPU otherwise):

```bash
python scripts/train.py --data dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
# or
./run.sh train
```

GPU settings (tuned for a 6 GB card):
- `--device auto` picks CUDA GPU `0` if available; force with `--device 0` or `--device cpu`.
- `--max-vram-gb 5.5` caps this process's VRAM, leaving headroom for the display / CUDA context.
- `--batch auto` lets Ultralytics AutoBatch pick the largest batch that uses ~70% of that cap. Pass an int (e.g. `--batch 16`) to fix it.
- Mixed precision (AMP), cuDNN benchmark and TF32 are enabled on GPU.
- `--cache ram` caches decoded images in RAM for faster data loading if you have spare memory.
- VRAM usage is printed every epoch and logged to `runs/detect/<name>/vram.csv`; a warning appears if peak usage passes 90% of the cap. On out-of-memory, lower `--batch` or `--imgsz`.

Verify CUDA first with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, install a CUDA build of PyTorch from the PyTorch install selector.

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
