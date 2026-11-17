"""CUDA device selection and VRAM tracking helpers (tuned for a 6 GB GPU)."""
import csv
from pathlib import Path

import torch

GB = 1024 ** 3


def resolve_device(device: str) -> str:
    """Turn 'auto' into '0' when CUDA is usable, otherwise 'cpu'."""
    device = str(device).strip().lower()
    if device in {"auto", "", "cuda"}:
        if torch.cuda.is_available():
            return "0"
        print("WARNING: CUDA not available - falling back to CPU (training will be slow).")
        print("         Check `nvidia-smi` and that torch was installed with CUDA support.")
        return "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"Device '{device}' requested but CUDA is not available.")
    return device


def cuda_index(device: str):
    """Return the first CUDA index for a device string, or None for CPU."""
    if device == "cpu":
        return None
    return int(device.split(",")[0].replace("cuda:", ""))


def configure_cuda(index: int, max_vram_gb: float) -> float:
    """Enable fast CUDA kernels and cap this process's VRAM. Returns the cap in GB."""
    torch.backends.cudnn.benchmark = True  # fixed imgsz -> pick fastest conv kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / GB
    cap_gb = min(max_vram_gb, total_gb)
    torch.cuda.set_per_process_memory_fraction(cap_gb / total_gb, index)
    torch.cuda.reset_peak_memory_stats(index)
    print(f"GPU {index}: {props.name} | {total_gb:.2f} GB total | VRAM cap {cap_gb:.2f} GB")
    return cap_gb


def vram_snapshot(index: int) -> dict:
    free, total = torch.cuda.mem_get_info(index)
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / GB,
        "reserved_gb": torch.cuda.memory_reserved(index) / GB,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(index) / GB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(index) / GB,
        "device_used_gb": (total - free) / GB,  # includes other processes / display
    }


def add_vram_tracker(model, index: int, cap_gb: float, warn_ratio: float = 0.90):
    """Attach Ultralytics callbacks that log per-epoch VRAM usage to <run>/vram.csv."""
    state = {"csv": None}

    def on_train_start(trainer):
        torch.cuda.reset_peak_memory_stats(index)
        state["csv"] = Path(trainer.save_dir) / "vram.csv"
        print(f"VRAM log: {state['csv']}")

    def on_fit_epoch_end(trainer):
        snap = vram_snapshot(index)
        epoch = trainer.epoch + 1
        print(
            f"[VRAM] epoch {epoch}: alloc {snap['allocated_gb']:.2f} GB | "
            f"peak {snap['peak_reserved_gb']:.2f}/{cap_gb:.2f} GB | "
            f"device used {snap['device_used_gb']:.2f} GB"
        )
        if snap["peak_reserved_gb"] > warn_ratio * cap_gb:
            print(
                f"[VRAM] WARNING: peak usage is above {warn_ratio:.0%} of the "
                f"{cap_gb:.2f} GB cap - lower --batch or --imgsz if you hit OOM."
            )
        path = state["csv"]
        if path is not None:
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", *snap.keys()])
                if new:
                    w.writeheader()
                w.writerow({"epoch": epoch, **{k: round(v, 3) for k, v in snap.items()}})
        torch.cuda.reset_peak_memory_stats(index)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
