#!/usr/bin/env python3
"""Train or validate a small YOLO11 detector using Ultralytics (CUDA-first, 6 GB VRAM aware)."""
import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

from gpu_utils import add_vram_tracker, configure_cuda, cuda_index, resolve_device


def parse_batch(value: str, on_gpu: bool, cap_gb: float, total_gb: float):
    """'auto' -> AutoBatch targeting ~70% of the VRAM cap on GPU, 4 on CPU."""
    if value == "auto":
        if not on_gpu:
            return 4
        # Ultralytics treats 0<batch<1 as a fraction of *total* device memory,
        # so scale it to stay inside our cap.
        return round(0.70 * cap_gb / total_gb, 3)
    return float(value) if "." in value else int(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", default="auto",
                   help="'auto' (fit to VRAM), an int, or a 0-1 fraction of GPU memory")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    p.add_argument("--max-vram-gb", type=float, default=5.5,
                   help="VRAM cap for this process (6 GB card minus headroom for display/CUDA context)")
    p.add_argument("--cache", default="false", choices=["false", "ram", "disk"],
                   help="Cache decoded images to speed up data loading")
    p.add_argument("--name", default="garbage-yolo11n")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()

    data = Path(args.data).resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}. Run prepare_dataset.py first.")

    device = resolve_device(args.device)
    idx = cuda_index(device)
    on_gpu = idx is not None
    cap_gb = total_gb = 0.0
    if on_gpu:
        cap_gb = configure_cuda(idx, args.max_vram_gb)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3

    model = YOLO(args.model)
    if args.validate_only:
        metrics = model.val(data=str(data), imgsz=args.imgsz, device=device, half=on_gpu)
        print(metrics)
        return

    batch = parse_batch(args.batch, on_gpu, cap_gb, total_gb)
    print(f"Device: {device} | batch: {batch} | workers: {args.workers} | cache: {args.cache}")
    if on_gpu:
        add_vram_tracker(model, idx, cap_gb)

    try:
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            workers=args.workers,
            device=device,
            project="runs/detect",
            name=args.name,
            patience=25,
            save=True,
            plots=True,
            cache=False if args.cache == "false" else args.cache,
            amp=on_gpu,  # mixed precision: ~half the activation memory and faster on CUDA
            exist_ok=True,
            resume=args.resume,
        )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"CUDA out of memory with batch={batch}. Retry with a smaller --batch "
            f"(e.g. 8) or --imgsz 512, or close other GPU apps."
        )
    finally:
        if on_gpu:
            peak = torch.cuda.max_memory_reserved(idx) / 1024 ** 3
            print(f"Final peak VRAM (last epoch): {peak:.2f} GB of {cap_gb:.2f} GB cap")


if __name__ == "__main__":
    main()
