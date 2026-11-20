# Changelog

All notable changes to this project. Newest first.

## [0.3.1-dev] - 2026-11-20 - Grounding DINO confidence 0.30, requirements, GPU runbook

### Changed
- `scripts/clips_to_frames.py`: the Grounding DINO garbage confidence default goes from 0.40 to **0.30** (user decision). The new function `default_garbage_conf(backend)` holds the defaults.
  Re-check at 0.40 on PEyhUwGBrnQ: garbage drafts fell from 38 to 1. Real litter scored about 0.34, about the same as false positives on vehicles (0.32–0.35), so vehicles are removed during review instead.
- `requirements.txt`: added `transformers>=5.23,<6` (Grounding DINO) and a comment that YOLO-World auto-installs CLIP from git.
- `README.md`: notes confidence 0.30 and links the runbook.

### Added
- `docs/GPU_RUNBOOK.md`: every command for the GPU machine, each stage with a smoke test first:
  setup → old-label repair → clip collection → Grounding DINO drafts → X-AnyLabeling review → merge (backup first) →
  yolo11n pilot → full yolo11m run under a new name → evaluate against the baseline.
- `tests/test_clips_to_frames.py`: `TestDefaultConf` (93 tests total).
- `README.md`: "Quick path" with two copy-paste blocks. Block A: repair old labels → CCTV drafts → preview. Block B, after human review: mark reviewed → back up → merge with Roboflow filtering → preview.

### Files touched
`scripts/clips_to_frames.py`, `requirements.txt`, `README.md`, `docs/GPU_RUNBOOK.md` (new), `tests/test_clips_to_frames.py`,
`CHANGELOG.md`, `PROJECT_STATE.md`.

## [0.3.0-dev] - 2026-11-20 - Dataset repair, preview, opt-in Roboflow down-sampling

### Added
- `scripts/preview_dataset.py`: draws ground-truth YOLO boxes on every image of a dataset (CPU only, no model).
  Writes `dataset_preview/<split>/` and `dataset_preview/<split>/empty_labels/`, plus `summary.json`.
- `scripts/restore_old_labels.py`: restores the old dataset's garbage boxes from `data/manifest.jsonl` into
  `dataset_merged/labels/*/old_*.txt` (keeps auto-labelled person lines) and rebuilds `dataset/labels` + `dataset/data.yaml`.
  Backs up each labels folder first (`labels_backup_<timestamp>/`).
- `scripts/merge_datasets.py`: opt-in Roboflow down-sampling, all off by default:
  `--rf-drop-closeup FRAC`, `--rf-drop-portrait FRAC`, `--rf-cap-train N`, `--seed N`.
  New function `filter_roboflow`; drop reasons `rf_closeup`, `rf_portrait`, `rf_cap` appear in `merge_report.json`
  under `dropped`, and the settings under `rf_downsampling`.
- `tests/test_rf_filter.py`: 9 tests (defaults, close-up, boundary, portrait, seeded cap, cap > count, no rf records,
  invalid values, CLI exit).
- `scripts/clips_to_frames.py` (Part B smoke test): CCTV clip → frames at `--fps` → dHash dedupe inside the clip →
  draft labels (person: yolo11x, garbage: YOLO-World `yolov8m-worldv2` with text prompts) → `datasets_raw/cctv/`
  (`images|labels/<split>/`, `data.yaml`, `review.json` with `reviewed=false`). A whole clip goes to one split (`--assign`).
  Re-running never resets `reviewed=true`.
- `tests/test_clips_to_frames.py`: 12 tests (assign parsing, fps sampling, dedupe incl. unreadable/empty, draft label
  order, background frames, split writing, review persistence, CLI failures). Models are faked.
- `scripts/clips_to_frames.py`: `--garbage-backend gdino` drafts garbage with Grounding DINO
  (`IDEA-Research/grounding-dino-tiny` by default, via `transformers`), with phrase prompts `GDINO_PROMPTS`,
  `--gdino-text-threshold` and class-agnostic NMS (`--nms-iou`). The default is still `yolo-world`. `review.json` records `garbage_backend`.
  New: `GroundingDinoDetector`, `gdino_prompt`, `_Boxes`. +4 tests (79 total).
- `scripts/clips_to_frames.py` (scale-up for about 1500 frames, approved 2026-11-20):
  - `--device auto` (default, via `gpu_utils.resolve_device`): CUDA when available, for both yolo11x and Grounding DINO.
  - Garbage confidence default is 0.40 for gdino (0.15 for yolo-world).
  - `filter_garbage_boxes`: drops garbage drafts larger than `--max-garbage-area 0.8` of the image, and drafts more than `--nested-frac 0.8` inside a larger garbage draft.
  - Folders of clips are accepted (`expand_videos`).
  - Stable automatic clip→split (`auto_split`, `--val-pct 15`, `--test-pct 0`).
  - `--max-frames` budget.
  - Writes `classes.txt` for X-AnyLabeling.
  - `--mark-reviewed CLIP|all`.
  - `review.json` also records `garbage_filtered` and `device`.
- `scripts/merge_datasets.py`: `--cctv DIR` with new `collect_cctv`. Only clips with `reviewed=true` are merged, and each clip is one leakage group (`cctv:<clip>`). `merge_report.json` → `sources.cctv`.
- `run.sh`:
  - new `cctv-frames` command (Grounding DINO, `CCTV_DIR`, `CCTV_MAX_FRAMES=1500`, preview).
  - `merge` adds `--cctv $CCTV_DIR` when `review.json` exists and passes extra args through.
- `README.md`: old-label repair steps and the CCTV draft → X-AnyLabeling review → merge workflow.
- `tests/test_merge_cctv.py` (4 tests). `tests/test_clips_to_frames.py`: +9 tests (92 total).
- Local only: the YOLO-World smoke output was moved to `datasets_raw/cctv_smoke_yoloworld` and `dataset_preview/cctv_smoke_yoloworld` (not deleted).
- New dependency (approved 2026-11-20): `transformers` 5.23.0, installed in the venv and not pinned in `requirements.txt` (user said not needed).
- New dependency, auto-installed by Ultralytics on first YOLO-World use (approved 2026-11-20):
  `clip` from `git+https://github.com/ultralytics/CLIP.git` (+ `ftfy`, `regex`, `wcwidth`). Not in `requirements.txt` yet.
- `.gitignore`: `dataset_preview/`.

### Fixed
- Data (not code): the training machine's `dataset/` had 0-byte label files, so all 1744 `old_*` images in
  `dataset_merged` had no garbage boxes. Restored with `scripts/restore_old_labels.py` (train +4489, test +569 garbage boxes).

### Files touched
`scripts/merge_datasets.py`, `scripts/preview_dataset.py` (new), `scripts/restore_old_labels.py` (new),
`tests/test_rf_filter.py` (new), `scripts/clips_to_frames.py` (new), `tests/test_clips_to_frames.py` (new), `tests/test_merge_cctv.py` (new),
`run.sh`, `README.md`,
`.gitignore`, `CHANGELOG.md`, `PROJECT_STATE.md`.

## [0.2.1] - 2026-11-19 - Roboflow class names + clearer auth errors

### Changed
- `scripts/merge_datasets.py`: `plastic`, `metal`, `paper`, `glass` are now recognised as garbage. This means the Roboflow
  dataset (People, Waste, Plastic, Metal, Paper, Glass) merges with `./run.sh merge` and no `--class-map` options.
- `scripts/download_roboflow.py`: on a 401/403 response the error now shows:
  - Roboflow's own error text, with the key masked
  - where the key came from (shell environment variable or `.env` path)
  - the key's length and last 3 characters
  - how to fix it
  New module variable: `KEY_SOURCE`.

### Added
- 4 tests: real Roboflow class names; 401 message (reason shown, key masked, source named); env-var source; empty 401 body.

### Files touched
`scripts/merge_datasets.py`, `scripts/download_roboflow.py`, `tests/test_merge_datasets.py`,
`tests/test_download_roboflow.py`, `CHANGELOG.md`, `PROJECT_STATE.md`.

## [0.2.0] - 2026-11-19 - Garbage + person CCTV detector

### Added
- `scripts/download_roboflow.py`: downloads the Roboflow "Human and Trash Detection" dataset
  (noyals-workspace/human-and-trash-detection-tciuj-3hmlf) through the Roboflow REST API. The key
  comes from `ROBOFLOW_API_KEY` in `.env`. It is masked in every message, and the zip is
  extracted with zip-slip protection.
- `scripts/merge_datasets.py`: merges the old dataset and the Roboflow dataset into `dataset_merged/` with the classes
  `garbage`(0) and `person`(1):
  - remaps classes by keyword (stops on unknown names unless `--class-map` says what to do)
  - auto-labels people in the old images with `yolo11x.pt` (results are cached)
  - removes Roboflow augmentation copies that leak across splits, and near-duplicate images (dHash)
  - optional `--backgrounds` folder of empty scenes
  - writes `merge_report.json`
- `scripts/cctv_augment.py`: Albumentations CCTV degradations (lighting, fog/rain, grayscale IR, blur, noise,
  downscale, a fake timestamp overlay, JPEG 20-75), with strength levels `off/light/medium/heavy` and a `--preview` tool.
- `scripts/predict_media.py`: runs inference on images, videos, folders, globs, RTSP/HTTP streams or a webcam.
  - Writes annotated output, `detections.jsonl` and `summary.json`.
  - Video alerts use persistence (`--persist`/`--window`).
  - Optional image-level accuracy against YOLO labels (`--labels`).
- `scripts/debug_log.py`: switchable `[DEBUG][function]` logging (`--debug` or `GD_DEBUG=1`).
- `tests/` (stdlib `unittest`, 50 tests) and `tests/cctv_test_clips.txt` (public CCTV dumping clips + baseline results).
- `.env.example`; `run.sh` commands `roboflow`, `merge`, `predict`.
- `CHANGELOG.md`, `PROJECT_STATE.md`.

### Changed
- `scripts/train.py`:
  - defaults are now `--model yolo11m.pt`, `--data dataset_merged/data.yaml` and `--name garbage-person-yolo11m`
  - new `--cctv-aug` (default `medium`), `--patience` (default 50, was hard-coded 25) and `--debug`
- `run.sh`: defaults are now MODEL=yolo11m.pt and DATA_YAML=dataset_merged/data.yaml, plus new variables RUN_NAME, CCTV_AUG, RF_VERSION and PERSON_MODEL;
  `all` now runs roboflow and merge instead of download. The file is now committed as executable.
- `requirements.txt`: + `albumentations>=2.0.8,<2.1` (MIT).
- `.gitignore`: + `datasets_raw/`, `dataset_merged/`, `cctv_test/`.
- `README.md`: v2 workflow section.

### Files touched
`scripts/train.py`, `run.sh`, `requirements.txt`, `.gitignore`, `README.md`, `.env.example`,
`scripts/{download_roboflow,merge_datasets,cctv_augment,predict_media,debug_log}.py`,
`tests/{_helpers,test_download_roboflow,test_merge_datasets,test_cctv_augment,test_predict_media}.py`,
`tests/cctv_test_clips.txt`, `CHANGELOG.md`, `PROJECT_STATE.md`.

## [0.1.x] - 2026-11-17/18 - Initial pipeline (from git history)
- `first commit`: JSONL manifest -> YOLO dataset (`prepare_dataset.py`), audit, YOLO11n training, test visualizer.
- `Train on CUDA with 6 GB VRAM cap and per-epoch VRAM tracking` (`gpu_utils.py`).
- `Manifesto`: committed `data/manifest.jsonl` (4,234 image records, signed URLs that expire on 2026-11-20).
