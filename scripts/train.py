#!/usr/bin/env python3
"""Train or validate a small YOLO11 detector using Ultralytics."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="Start with yolo11n.pt on a laptop")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    p.add_argument("--name", default="garbage-yolo11n")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()

    data = Path(args.data).resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}. Run prepare_dataset.py first.")

    model = YOLO(args.model)
    if args.validate_only:
        metrics = model.val(data=str(data), imgsz=args.imgsz, device=args.device)
        print(metrics)
        return

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs/detect",
        name=args.name,
        patience=25,
        save=True,
        plots=True,
        cache=False,
        amp=True,
        exist_ok=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
