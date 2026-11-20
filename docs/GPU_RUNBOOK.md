# GPU machine runbook: CCTV data → merge → retrain

Run these on the CUDA training machine, from the repo root, in order.
Each stage starts with a **smoke test**. Look at its output before running the full stage.
Commands marked ⚠ overwrite something, and a backup step comes right before each of them.

## 0. Update code and environment
```bash
cd Garbage-Detector-Model
git pull
./run.sh deps                                   # venv + requirements (now includes transformers for Grounding DINO)
nvidia-smi                                      # GPU visible?
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True
.venv/bin/python -m unittest discover -s tests -t tests   # all tests OK
```

## 1. Repair the old garbage labels (this machine's `dataset/` has 0-byte label files)
```bash
python scripts/restore_old_labels.py            # backs up labels first (labels_backup_<time>/)
find dataset/labels -name '*.txt' -empty | wc -l  # should be about 177 (names missing from the manifest), not 4211
ls datasets_raw/roboflow/data.yaml || ./run.sh roboflow   # Roboflow raw data needed by the merge
```
`restore_old_labels.py` reads `data/manifest.jsonl` and needs no download.
The 2395 "unreadable" old images in the last merge mean some files under `dataset/images` are broken on this machine. If the next merge still reports `unreadable_image`, copy `dataset/images` from a machine where they open.

## 2. Collect CCTV clips (target: about 1500 frames, roughly 60 one-minute clips from 20+ different cameras or scenes)
Put every clip in one folder, e.g. `cctv_clips/`. It is git-ignored if you place it under `cctv_test/`, otherwise add it to `.gitignore`.
Best: your own camera at the real site (staged dumping, day and night). YouTube clips are for internal use only.
```bash
pipx install yt-dlp                             # separate tool, not a project dependency
mkdir -p cctv_test/cctv_clips && cd cctv_test/cctv_clips
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b" --merge-output-format mp4 \
       -o "%(id)s.%(ext)s" -- VIDEO_ID_1 VIDEO_ID_2 ...
cd ../..
```
Trim news intros and ads where you can. Otherwise delete those frames during review.

## 3. Draft labels with Grounding DINO + yolo11x (CUDA)
```bash
# smoke test: about 100 frames, then look at the preview
CCTV_MAX_FRAMES=100 ./run.sh cctv-frames cctv_test/cctv_clips/
xdg-open dataset_preview/cctv/train             # garbage = red, person = orange. Are the drafts sensible?

# full run: resumes and skips clips already done; stops starting new clips at 1500 frames
./run.sh cctv-frames cctv_test/cctv_clips/
cat datasets_raw/cctv/review.json | head -40    # per-clip split, frames, draft counts, device
```
The defaults are: 1 fps, garbage confidence 0.30, no garbage box over 80% of the image, no garbage box over 80% inside a larger one, and about 15% of clips to val.
To force a split: `./run.sh cctv-frames cctv_test/cctv_clips/ --assign CLIPID=test`.

## 4. Review every frame in X-AnyLabeling (drafts are NOT ground truth)
```bash
pip install x-anylabeling-cvhub                 # desktop app (PyPI name per the project); see its README for the launch command
```
For each of `train` and `val` (and `test` if used):
1. Open the folder `datasets_raw/cctv/images/<split>`.
2. **Import:** `Import` → YOLO HBB, choose `datasets_raw/cctv/classes.txt` and the folder `datasets_raw/cctv/labels/<split>`.
3. Fix the boxes. Keys: `D` next image, `A` previous, `R` rectangle, `Delete` removes a box.
   - Delete garbage boxes on vehicles (scooters, rickshaws), people and signs.
   - Use one box per pile, and separate boxes only for separate items. Box every person.
   - For ad, studio or non-CCTV frames, delete the image **and** its `.txt`.
4. **Export:** `Export Annotations` → `Export YOLO Annotations` with the same `classes.txt`. **Set the output to `datasets_raw/cctv/labels/<split>`**, because the default is a folder next to the images.

Then:
```bash
python scripts/preview_dataset.py --data datasets_raw/cctv --out dataset_preview/cctv   # check the corrected boxes
python scripts/clips_to_frames.py --out datasets_raw/cctv --mark-reviewed all          # or list clip ids
```

## 5. Merge (⚠ rebuilds dataset_merged)
```bash
cp -a dataset_merged dataset_merged_backup_$(date +%Y%m%d_%H%M)   # keep the previous merge
./run.sh merge --rf-drop-closeup 0.5 --rf-drop-portrait 0.4 --rf-cap-train 2500
python - <<'PY'
import json; r = json.load(open("dataset_merged/merge_report.json"))
print("cctv:", r["sources"]["cctv"]); print("dropped:", r["dropped"]); print("splits:", json.dumps(r["splits"], indent=1))
PY
rm -rf dataset_preview && python scripts/preview_dataset.py --data dataset_merged --out dataset_preview
cat dataset_preview/summary.json                 # empty_labels should be about 0; check the garbage/person box counts
```
Pass criteria: `sources.cctv.frames_used` is about 1500, `unreadable_image` is about 0, old images carry garbage boxes, and val contains CCTV frames.

## 6. Training smoke test (small model, few epochs; pipeline check only, not accuracy)
```bash
MODEL=yolo11n.pt EPOCHS=3 RUN_NAME=pilot-yolo11n ./run.sh train
tail -4 runs/detect/runs/detect/pilot-yolo11n/results.csv      # losses finite and falling, mAP > 0
RUN_NAME=pilot-yolo11n ./run.sh predict cctv_test/clips --max-seconds 20 --out cctv_test/out --name pilot
```
Stop and investigate if a loss is NaN or inf, if mAP50 is 0 for garbage, or if the pilot predictions are empty.

## 7. Full training (new run name, so the old best.pt is not overwritten)
```bash
tmux new -s train                               # keeps running if SSH drops
RUN_NAME=garbage-person-yolo11m-v3 EPOCHS=100 CCTV_AUG=medium ./run.sh train
# detach: Ctrl-b d · re-attach: tmux attach -t train
# resume after a crash: point --model at THIS run's last.pt. Without that, Ultralytics resumes the newest run it finds, which could be the pilot.
#   .venv/bin/python scripts/train.py --resume --name garbage-person-yolo11m-v3 \
#       --model runs/detect/runs/detect/garbage-person-yolo11m-v3/weights/last.pt
```
Per-epoch VRAM use is logged to `runs/detect/runs/detect/garbage-person-yolo11m-v3/vram.csv`. The default VRAM cap is `MAX_VRAM_GB=5.5`.

## 8. Evaluate against the old model
```bash
RUN_NAME=garbage-person-yolo11m-v3 ./run.sh evaluate                       # mAP on val (now includes CCTV frames)
RUN_NAME=garbage-person-yolo11m-v3 ./run.sh predict cctv_test/clips --sample-fps 5 --out cctv_test/out --name v3
```
Baseline (old model, 2026-11-20, 8 YouTube clips): garbage alert in 1 of 8 clips, and that one was a false positive on a scooter. People were detected well.
Compare `cctv_test/out/v3/summary.json`. Clips used for training must not be counted as test results.

## 9. Bring the model back
Copy `runs/detect/runs/detect/garbage-person-yolo11m-v3/weights/best.pt` to the dev PC and the Pi (NCNN export for the CPU demo, or Hailo DFC for a `.hef`).
Then commit the CHANGELOG and PROJECT_STATE updates with the results.
