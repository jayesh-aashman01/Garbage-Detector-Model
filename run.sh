#!/usr/bin/env bash
set -Eeuo pipefail

# YOLO11 Garbage Detection pipeline
# Usage: ./run.sh [all|deps|download|evaluate|train|test]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
MANIFEST="${MANIFEST:-data/manifest.jsonl}"
DATA_YAML="${DATA_YAML:-dataset/data.yaml}"
MODEL="${MODEL:-yolo11n.pt}"
BEST_MODEL="${BEST_MODEL:-runs/detect/runs/detect/garbage-yolo11n/weights/best.pt}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
SAMPLES="${SAMPLES:-10}"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

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

evaluate() {
    log "Evaluating model on validation split"
    [[ -f "$DATA_YAML" ]] || die "Dataset config not found: $DATA_YAML. Run: ./run.sh download"
    [[ -f "$BEST_MODEL" ]] || die "Trained model not found: $BEST_MODEL. Run: ./run.sh train"
    yolo detect val model="$BEST_MODEL" data="$DATA_YAML" split=val imgsz="$IMGSZ" device="$DEVICE"
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
        --device "$DEVICE"
}

test_model() {
    log "Testing on random images from the test split"
    [[ -f "$BEST_MODEL" ]] || die "Trained model not found: $BEST_MODEL. Run: ./run.sh train"
    "$PYTHON" scripts/test_visualize.py \
        --model "$BEST_MODEL" \
        --data "$DATA_YAML" \
        --samples "$SAMPLES" \
        --imgsz "$IMGSZ" \
        --device "$DEVICE"
}

usage() {
    cat <<USAGE
Usage: ./run.sh [all|deps|download|evaluate|train|test]

Commands:
  deps       Create .venv and install requirements
  download   Download/prepare dataset and audit it
  evaluate   Evaluate best.pt on validation split
  train      Train YOLO11n
  test       Visualize predictions on random test images
  all        Run dependencies, download, train, evaluate, then test

Environment overrides:
  DEVICE=cpu|0  EPOCHS=100  IMGSZ=640  BATCH=2  WORKERS=2
  MANIFEST=data/manifest.jsonl  BEST_MODEL=/path/to/best.pt

Examples:
  ./run.sh all
  ./run.sh deps
  DEVICE=cpu EPOCHS=100 ./run.sh train
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
        download
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
