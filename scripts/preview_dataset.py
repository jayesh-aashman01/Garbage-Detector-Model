#!/usr/bin/env python3
"""Draw ground-truth YOLO boxes on every image of a dataset (no model, CPU only).

Output layout (default --out dataset_preview):
    dataset_preview/<split>/<image>.jpg              images with at least one box
    dataset_preview/<split>/empty_labels/<image>.jpg images whose label file is empty/missing
    dataset_preview/summary.json                     per-split counts

Usage:
    python scripts/preview_dataset.py --data dataset_merged --out dataset_preview
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(40, 40, 255), (255, 200, 40), (40, 220, 40), (220, 40, 220)]  # BGR per class


def load_names(root: Path) -> list:
    names = yaml.safe_load((root / "data.yaml").read_text()).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def read_boxes(label_path: Path):
    """Return [(cls, cx, cy, w, h)] in normalised coords; polygons become boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            boxes.append((cls, *vals))
        else:  # polygon: x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def draw_one(job):
    img_path, label_path, out_dir, names = job
    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable", {}
    h, w = img.shape[:2]
    boxes = read_boxes(Path(label_path))
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.5, min(h, w) / 900)
    counts = {}
    for cls, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = COLORS[cls % len(COLORS)]
        name = names[cls] if 0 <= cls < len(names) else f"cls{cls}"
        counts[name] = counts.get(name, 0) + 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(y1, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, name, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, thick - 1))
    kind = "boxed" if boxes else "empty"
    dest = Path(out_dir) / ("empty_labels" if not boxes else "")
    dest.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest / (img_path.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return kind, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_merged", help="dataset root with data.yaml, images/, labels/")
    ap.add_argument("--out", default="dataset_preview")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root, out = Path(args.data), Path(args.out)
    names = load_names(root)
    summary = {}
    for split in args.splits:
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            print(f"skip {split}: {img_dir} missing")
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        jobs = [(p, lab_dir / (p.stem + ".txt"), out / split, names) for p in imgs]
        stats = {"images": len(imgs), "with_boxes": 0, "empty_labels": 0, "unreadable": 0, "boxes": {}}
        with ProcessPoolExecutor(args.workers) as ex:
            for kind, counts in ex.map(draw_one, jobs, chunksize=32):
                key = {"boxed": "with_boxes", "empty": "empty_labels"}.get(kind, "unreadable")
                stats[key] += 1
                for k, v in counts.items():
                    stats["boxes"][k] = stats["boxes"].get(k, 0) + v
        summary[split] = stats
        print(f"{split}: {stats}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
