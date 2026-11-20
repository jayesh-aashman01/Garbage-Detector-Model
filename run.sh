#!/usr/bin/env bash
set -Eeuo pipefail

# YOLO11 Garbage Detection pipeline
# Usage: ./run.sh [all|deps|download|roboflow|merge|evaluate|train|test|predict]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
MANIFEST="${MANIFEST:-data/manifest.jsonl}"
DATA_YAML="${DATA_YAML:-dataset_merged/data.yaml}"  # CHANGED: merged garbage+person dataset
MODEL="${MODEL:-yolo11m.pt}"                       # CHANGED: was yolo11n.pt
RUN_NAME="${RUN_NAME:-garbage-person-yolo11m}"     # ADDED
BEST_MODEL="${BEST_MODEL:-runs/detect/runs/detect/$RUN_NAME/weights/best.pt}"  # CHANGED
CCTV_AUG="${CCTV_AUG:-medium}"                     # ADDED: off|light|medium|heavy
RF_VERSION="${RF_VERSION:-latest}"                 # ADDED: Roboflow dataset version
PERSON_MODEL="${PERSON_MODEL:-yolo11x.pt}"         # ADDED: COCO model that auto-labels people
CCTV_DIR="${CCTV_DIR:-datasets_raw/cctv}"          # ADDED: clips_to_frames.py output (reviewed clips are merged)
CCTV_MAX_FRAMES="${CCTV_MAX_FRAMES:-1500}"         # ADDED: frame budget for ./run.sh cctv-frames
CCTV_CLIP_LIST="${CCTV_CLIP_LIST:-data/cctv_train_clips.txt}"  # ADDED: YouTube ids for ./run.sh cctv-download
CCTV_CLIPS="${CCTV_CLIPS:-cctv_test/cctv_clips}"  # ADDED: where training clips are downloaded (git-ignored)
DEVICE="${DEVICE:-auto}"  # auto -> CUDA GPU 0 if available, else CPU
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-auto}"    # auto -> fit batch to VRAM cap
MAX_VRAM_GB="${MAX_VRAM_GB:-5.5}"  # 6 GB card minus headroom
WORKERS="${WORKERS:-4}"
SAMPLES="${SAMPLES:-10}"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Resolve DEVICE=auto to a concrete value for the yolo CLI.
resolved_device() {
    if [[ "$DEVICE" == "auto" ]]; then
        "$PYTHON" -c "import torch; print('0' if torch.cuda.is_available() else 'cpu')"
    else
        echo "$DEVICE"
    fi
}

activate_venv() {
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        PYTHON="python"
    fi
    command -v "$PYTHON" >/dev/null 2>&1 || die "Python not found: $PYTHON"
}

check_requirements_file() {
    [[ -f "$REQUIREMENTS" ]] || die "Missing $REQUIREMENTS in $ROOT_DIR"
}

deps() {
    log "Setting up Python environment and dependencies"
    command -v python3 >/dev/null 2>&1 || die "Install Python 3 first"
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR" || die "Could not create venv. On Fedora, install python3-pip and python3-devel if needed."
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    check_requirements_file
    python -m pip install -r "$REQUIREMENTS"
    PYTHON="python"
}

download() {
    log "Preparing / downloading dataset"
    [[ -f "$MANIFEST" ]] || die "Manifest not found: $MANIFEST"
    "$PYTHON" scripts/prepare_dataset.py --manifest "$MANIFEST" --output dataset --workers 6
    "$PYTHON" scripts/audit_dataset.py --root dataset
}

# ADDED: download the Roboflow human+trash dataset (API key comes from .env)
roboflow() {
    log "Downloading Roboflow human+trash dataset"
    "$PYTHON" scripts/download_roboflow.py --version "$RF_VERSION"
}

# ADDED: merge old garbage dataset + Roboflow dataset into dataset_merged/ (garbage, person)
merge() {
    log "Merging datasets (auto-labelling people in the old images)"
    [[ -d dataset/images ]] || die "Old dataset not found in ./dataset. Copy it here or run: ./run.sh download"
    # MODIFIED: add reviewed CCTV frames when present; extra args pass through (e.g. --rf-cap-train 2500)
    local cctv_args=()
    [[ -f "$CCTV_DIR/review.json" ]] && cctv_args=(--cctv "$CCTV_DIR")
    "$PYTHON" scripts/merge_datasets.py --old dataset --roboflow datasets_raw/roboflow \
        --output dataset_merged --person-model "$PERSON_MODEL" --device "$(resolved_device)" --overwrite \
        "${cctv_args[@]}" "$@"
}

# ADDED: CCTV clips -> 1 fps frames -> Grounding DINO + yolo11x draft labels (review before merging)
# ADDED: download the training clips listed in $CCTV_CLIP_LIST (skips clips already present; failures only warn)
cctv_download() {
    [[ -f "$CCTV_CLIP_LIST" ]] || die "Clip list not found: $CCTV_CLIP_LIST"
    mkdir -p "$CCTV_CLIPS"
    local ids=() id ok=0 skipped=0 failed=()
    mapfile -t ids < <(grep -v '^[[:space:]]*#' "$CCTV_CLIP_LIST" | awk 'NF {print $1}')
    log "Downloading ${#ids[@]} clips from $CCTV_CLIP_LIST into $CCTV_CLIPS"
    for id in "${ids[@]}"; do
        if [[ -f "$CCTV_CLIPS/$id.mp4" || -f "$CCTV_CLIPS/$id.webm" || -f "$CCTV_CLIPS/$id.mkv" ]]; then
            skipped=$((skipped + 1)); continue
        fi
        # video-only mp4 <=720p: no audio needed, and no ffmpeg merge step
        if "$PYTHON" -m yt_dlp -q --no-warnings --no-playlist \
            -f "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/b[height<=720]/b" \
            -o "$CCTV_CLIPS/%(id)s.%(ext)s" -- "$id"; then
            ok=$((ok + 1)); echo "  [ok] $id"
        else
            failed+=("$id"); echo "  [WARN] download failed: $id (removed/private/geo-blocked?)"
        fi
    done
    log "Clips: downloaded=$ok already_present=$skipped failed=${#failed[@]} ${failed[*]:-}"
    local have
    have=$(find "$CCTV_CLIPS" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.mkv' \) | wc -l)
    [[ "$have" -gt 0 ]] || die "No clips in $CCTV_CLIPS after download"
}

cctv_frames() {
    # MODIFIED: with no arguments, use the downloaded clips folder
    [[ $# -gt 0 ]] || set -- "$CCTV_CLIPS"
    [[ -d "$1" || -f "$1" ]] || die "No clips at $1. Run: ./run.sh cctv-download (or pass a folder of clips)"
    log "CCTV frames from: $* (budget $CCTV_MAX_FRAMES frames, device $DEVICE)"
    "$PYTHON" scripts/clips_to_frames.py "$@" --out "$CCTV_DIR" --garbage-backend gdino \
        --max-frames "$CCTV_MAX_FRAMES" --person-model "$PERSON_MODEL" --device "$DEVICE"
    "$PYTHON" scripts/preview_dataset.py --data "$CCTV_DIR" --out dataset_preview/cctv
}

# ADDED: run the trained model on images / videos / folders / rtsp:// streams
predict() {
    log "Predicting on: ${*:-<nothing>}"
    [[ $# -gt 0 ]] || die "Usage: ./run.sh predict <image|video|folder|rtsp://...> [more ...]"
    [[ -f "$BEST_MODEL" ]] || die "Trained model not found: $BEST_MODEL. Run: ./run.sh train"
    "$PYTHON" scripts/predict_media.py "$@" --model "$BEST_MODEL" --imgsz "$IMGSZ" --device "$(resolved_device)"
}

evaluate() {
    log "Evaluating model on validation split"
    [[ -f "$DATA_YAML" ]] || die "Dataset config not found: $DATA_YAML. Run: ./run.sh download"
    [[ -f "$BEST_MODEL" ]] || die "Trained model not found: $BEST_MODEL. Run: ./run.sh train"
    yolo detect val model="$BEST_MODEL" data="$DATA_YAML" split=val imgsz="$IMGSZ" device="$(resolved_device)"
}

train() {
    log "Training YOLO11"
    [[ -f "$DATA_YAML" ]] || die "Dataset config not found: $DATA_YAML. Run: ./run.sh download"
    "$PYTHON" scripts/train.py \
        --model "$MODEL" \
        --data "$DATA_YAML" \
        --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --max-vram-gb "$MAX_VRAM_GB" \
        --name "$RUN_NAME" \
        --cctv-aug "$CCTV_AUG"
}

test_model() {
    log "Testing on random images from the test split"
    [[ -f "$BEST_MODEL" ]] || die "Trained model not found: $BEST_MODEL. Run: ./run.sh train"
    "$PYTHON" scripts/test_visualizer.py \
        --model "$BEST_MODEL" \
        --data "$DATA_YAML" \
        --samples "$SAMPLES" \
        --imgsz "$IMGSZ" \
        --device "$(resolved_device)"
}

usage() {
    cat <<USAGE
Usage: ./run.sh [all|deps|download|roboflow|merge|evaluate|train|test|predict]

Commands:
  deps       Create .venv and install requirements
  download   Download/prepare the old dataset and audit it
  roboflow   Download the Roboflow human+trash dataset (needs ROBOFLOW_API_KEY in .env)
  merge      Build dataset_merged/ (garbage + person) and audit it
  cctv-download  Download training clips from data/cctv_train_clips.txt into cctv_test/cctv_clips/
  cctv-frames  CCTV clips -> draft-labelled frames for review (default folder: cctv_test/cctv_clips/)
  predict    Run best.pt on images/videos/folders/streams: ./run.sh predict clip.mp4 frames/
  evaluate   Evaluate best.pt on validation split
  train      Train YOLO11m (CCTV augmentation on)
  test       Visualize predictions on random test images
  all        deps, roboflow, merge, train, evaluate, then test (needs ./dataset present)

Environment overrides:
  DEVICE=auto|cpu|0  EPOCHS=100  IMGSZ=640  BATCH=auto|16  WORKERS=4
  MAX_VRAM_GB=5.5  (VRAM cap for training; per-epoch usage logged to <run>/vram.csv)
  MANIFEST=data/manifest.jsonl  BEST_MODEL=/path/to/best.pt
  MODEL=yolo11m.pt  RUN_NAME=garbage-person-yolo11m  CCTV_AUG=off|light|medium|heavy
  RF_VERSION=latest|N  PERSON_MODEL=yolo11x.pt  CCTV_DIR=datasets_raw/cctv  CCTV_MAX_FRAMES=1500
  CCTV_CLIP_LIST=data/cctv_train_clips.txt  CCTV_CLIPS=cctv_test/cctv_clips

Examples:
  ./run.sh all
  ./run.sh deps
  DEVICE=0 BATCH=16 ./run.sh train
USAGE
}

COMMAND="${1:-all}"
case "$COMMAND" in
    deps)
        deps
        ;;
    download)
        activate_venv
        download
        ;;
    roboflow)
        activate_venv
        roboflow
        ;;
    merge)
        activate_venv
        shift
        merge "$@"
        "$PYTHON" scripts/audit_dataset.py --root dataset_merged
        ;;
    predict)
        activate_venv
        shift
        predict "$@"
        ;;
    cctv-download)
        activate_venv
        cctv_download
        ;;
    cctv-frames)
        activate_venv
        shift
        cctv_frames "$@"
        ;;
    evaluate)
        activate_venv
        evaluate
        ;;
    train)
        activate_venv
        train
        ;;
    test)
        activate_venv
        test_model
        ;;
    all)
        deps
        roboflow
        merge
        train
        evaluate
        test_model
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        die "Unknown command: $COMMAND"
        ;;
esac

log "Done"
