#!/usr/bin/env python3
"""Restore garbage boxes for the old dataset from data/manifest.jsonl.

Why: the old dataset/ copy on the training machine had 0-byte label files, so every
`old_*` image in dataset_merged/ lost its garbage boxes (only auto-labelled people survived).

What it does:
  1. dataset_merged/labels/<split>/old_*.txt  ->  manifest garbage boxes (class 0)
     + every existing non-garbage line (auto-labelled persons) kept as is.
  2. dataset/labels/<split>/*.txt and dataset/data.yaml are rewritten from the manifest
     (only for images present in dataset/images/<split>/).

A backup of each labels folder is written first (labels_backup_<timestamp>/).
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path


def safe_name(text: str) -> str:  # must match scripts/merge_datasets.py
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "img"


def load_manifest(path: Path) -> dict:
    """stem of manifest file name -> list of 'cls xc yc w h' lines."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != "image":
            continue
        boxes = (d.get("annotations") or {}).get("boxes") or []
        lines = []
        for b in boxes:
            cls, xc, yc, w, h = int(b[0]), *map(float, b[1:5])
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        out[Path(d["file"]).stem] = lines
    return out


def backup(labels_dir: Path) -> Path:
    dest = labels_dir.parent / f"labels_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(labels_dir, dest)
    return dest


def restore_merged(root: Path, manifest: dict) -> dict:
    # merged stem = safe_name("old_" + image file name) minus extension
    lookup = {Path(safe_name(f"old_{stem}.jpg")).stem: v for stem, v in manifest.items()}
    lookup.update({f"old_{stem}": v for stem, v in manifest.items()})
    stats = {}
    for split_dir in sorted((root / "labels").iterdir()):
        if not split_dir.is_dir():
            continue
        s = {"old_files": 0, "restored": 0, "garbage_boxes": 0, "kept_person_lines": 0, "not_in_manifest": 0}
        for lab in sorted(split_dir.glob("old_*.txt")):
            s["old_files"] += 1
            boxes = lookup.get(lab.stem)
            if boxes is None:
                s["not_in_manifest"] += 1
                continue
            kept = [l for l in lab.read_text().splitlines() if l.strip() and not l.startswith("0 ")]
            lab.write_text("".join(f"{l}\n" for l in boxes + kept))
            s["restored"] += 1
            s["garbage_boxes"] += len(boxes)
            s["kept_person_lines"] += len(kept)
        stats[split_dir.name] = s
    return stats


def restore_old_dataset(root: Path, manifest: dict) -> dict:
    stats = {}
    for split in ("train", "val", "test"):
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            continue
        lab_dir.mkdir(parents=True, exist_ok=True)
        n = missing = 0
        for img in img_dir.iterdir():
            boxes = manifest.get(img.stem)
            if boxes is None:
                missing += 1
                continue
            (lab_dir / f"{img.stem}.txt").write_text("".join(f"{l}\n" for l in boxes))
            n += 1
        stats[split] = {"written": n, "not_in_manifest": missing}
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n- garbage\n"
    )
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--merged", type=Path, default=Path("dataset_merged"))
    ap.add_argument("--old", type=Path, default=Path("dataset"))
    ap.add_argument("--skip-old", action="store_true", help="do not rewrite dataset/")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"manifest images: {len(manifest)}, with boxes: {sum(1 for v in manifest.values() if v)}")

    print(f"backup: {backup(args.merged / 'labels')}")
    for split, s in restore_merged(args.merged, manifest).items():
        print(f"merged {split}: {s}")

    if not args.skip_old and (args.old / "images").is_dir():
        print(f"backup: {backup(args.old / 'labels')}")
        for split, s in restore_old_dataset(args.old, manifest).items():
            print(f"dataset {split}: {s}")


if __name__ == "__main__":
    main()
