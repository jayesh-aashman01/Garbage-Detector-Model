#!/usr/bin/env python3
"""Find near-duplicate CCTV frames (same clip, same-looking image, same boxes) and move them aside.

Why: a fixed camera gives many 1-fps frames where nothing relevant changes. Whole-image hashes
alone also flag frames where only the person moved (useful variety), so a frame is a duplicate
only when BOTH hold, compared with an earlier kept frame of the same clip:
  1. 256-bit dHash distance <= --max-distance (default 40)
  2. same number of boxes and every box matches one of the same class with IoU >= --iou (0.7)
Boxes come from the X-AnyLabeling .json next to the image when it exists (your edits),
otherwise from labels/<split>/<name>.txt (the drafts).

Never moved: frames marked "checked" in X-AnyLabeling, and the earliest frame of each group.

Default run = preview only (contact sheets, nothing moved). --apply MOVES duplicates
(.jpg, .json, .txt) to <data>/_duplicates/<split>/... and appends to <data>/_duplicates/moved.jsonl.
Undo: move the files back from _duplicates/ to the same relative paths.
Close X-AnyLabeling before --apply (it caches the file list).

Example:
  python scripts/dedupe_cctv.py --data datasets_raw/cctv              # preview
  python scripts/dedupe_cctv.py --data datasets_raw/cctv --apply      # move duplicates
"""
import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402

FRAME_RE = re.compile(r"^(?P<clip>.+)_f\d{6}$")
SHEET_ROWS = 10      # pairs per preview sheet
THUMB = (400, 225)   # width, height of each thumbnail in the sheet


def dhash256(path: Path):
    """16x16 difference hash -> bool array of 256 bits, or None if the image can't be read."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def read_classes(data: Path) -> list[str]:
    p = data / "classes.txt"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return [c.strip() for c in p.read_text().splitlines() if c.strip()]


def load_boxes(img: Path, label_txt: Path, classes: list[str]) -> tuple[list, bool]:
    """Return ([(cls, x1, y1, x2, y2) normalised], checked).
    Transforms: X-AnyLabeling JSON rectangles (pixels) or YOLO lines (cx cy w h) into corner boxes."""
    js = img.with_suffix(".json")
    if js.exists():
        d = json.loads(js.read_text())
        w, h = d.get("imageWidth") or 1, d.get("imageHeight") or 1
        boxes = []
        for s in d.get("shapes", []):
            if s.get("label") not in classes or not s.get("points"):
                continue
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            boxes.append((classes.index(s["label"]), min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h))
        return boxes, bool(d.get("checked"))
    boxes = []
    if label_txt.exists():
        for line in label_txt.read_text().splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:])
            boxes.append((c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes, False


def iou(a, b) -> float:
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def same_boxes(A: list, B: list, iou_min: float) -> bool:
    """True when both frames have the same box count and every box in A matches a same-class box in B."""
    if len(A) != len(B):
        return False
    return all(any(a[0] == b[0] and iou(a, b) >= iou_min for b in B) for a in A)


def find_duplicates(data: Path, split: str, classes: list[str], max_distance: int, iou_min: float) -> list[dict]:
    """Return [{kept, dup, distance}] for one split. Frames are compared only within their clip,
    in frame order, against frames already kept."""
    dbg("find_duplicates", f"enter split={split} max_distance={max_distance} iou_min={iou_min}")
    img_dir, lbl_dir = data / "images" / split, data / "labels" / split
    clips = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        m = FRAME_RE.match(p.stem)
        clips[m.group("clip") if m else p.stem].append(p)
    pairs, n = [], 0
    for clip, frames in clips.items():
        kept = []  # (hash, boxes, path)
        for p in frames:
            n += 1
            h = dhash256(p)
            if h is None:
                dbg("find_duplicates", f"unreadable, skipped: {p.name}")
                continue
            boxes, checked = load_boxes(p, lbl_dir / f"{p.stem}.txt", classes)
            match = None
            if not checked:
                for kh, kb, kp in kept:
                    dist = int((h != kh).sum())
                    if dist <= max_distance and same_boxes(boxes, kb, iou_min):
                        match = (kp, dist)
                        break
            if match:
                pairs.append({"kept": match[0], "dup": p, "distance": match[1]})
            else:
                kept.append((h, boxes, p))
    dbg("find_duplicates", f"exit split={split} frames={n} clips={len(clips)} duplicates={len(pairs)}")
    return pairs


def write_preview(pairs: list[dict], out_dir: Path, split: str) -> list[Path]:
    """Write contact sheets: left = kept frame, right = frame that would be moved. Returns sheet paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for start in range(0, len(pairs), SHEET_ROWS):
        rows = []
        for pr in pairs[start:start + SHEET_ROWS]:
            ims = [cv2.resize(cv2.imread(str(p)), THUMB) for p in (pr["kept"], pr["dup"])]
            row = np.hstack([ims[0], np.full((THUMB[1], 6, 3), 255, np.uint8), ims[1]])
            cv2.putText(row, f"KEEP {pr['kept'].stem}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(row, f"MOVE {pr['dup'].stem} d={pr['distance']}", (THUMB[0] + 12, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            rows.append(row)
        p = out_dir / f"{split}_{start // SHEET_ROWS + 1:03d}.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        sheets.append(p)
    dbg("write_preview", f"split={split} pairs={len(pairs)} sheets={len(sheets)} dir={out_dir}")
    return sheets


def move_duplicates(data: Path, split: str, pairs: list[dict]) -> int:
    """Move each duplicate's .jpg/.json/.txt under <data>/_duplicates/ (same relative layout) and
    log it to _duplicates/moved.jsonl. Never overwrites a file already in _duplicates. Returns count moved."""
    dbg("move_duplicates", f"enter split={split} pairs={len(pairs)}")
    dup_root = data / "_duplicates"
    log = dup_root / "moved.jsonl"
    dup_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    with log.open("a") as fh:
        for pr in pairs:
            img = pr["dup"]
            srcs = [img, img.with_suffix(".json"), data / "labels" / split / f"{img.stem}.txt"]
            for src in srcs:
                if not src.exists():
                    continue
                dst = dup_root / src.relative_to(data)
                if dst.exists():
                    raise SystemExit(f"refusing to overwrite {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "split": split,
                                 "moved": img.name, "kept": pr["kept"].name,
                                 "distance": pr["distance"]}) + "\n")
            moved += 1
    dbg("move_duplicates", f"exit split={split} moved={moved} log={log}")
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets_raw/cctv", help="folder with images/, labels/, classes.txt")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-distance", type=int, default=40, help="max dHash bits (of 256) to call images alike")
    ap.add_argument("--iou", type=float, default=0.7, help="min IoU for boxes to count as unchanged")
    ap.add_argument("--preview", default="dataset_preview/cctv_duplicates", help="contact-sheet folder")
    ap.add_argument("--apply", action="store_true", help="move duplicates (default: preview only)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if a.debug:
        enable()
    data = Path(a.data)
    classes = read_classes(data)
    total = 0
    for split in a.splits:
        if not (data / "images" / split).is_dir():
            continue
        pairs = find_duplicates(data, split, classes, a.max_distance, a.iou)
        n_imgs = len(list((data / "images" / split).glob("*.jpg")))
        sheets = write_preview(pairs, Path(a.preview), split) if pairs else []
        print(f"{split}: {n_imgs} frames, {len(pairs)} duplicates, {n_imgs - len(pairs)} left"
              f"{f', preview {sheets[0].parent}/{split}_*.jpg ({len(sheets)} sheets)' if sheets else ''}")
        if a.apply and pairs:
            print(f"{split}: moved {move_duplicates(data, split, pairs)} to {data / '_duplicates'}")
        total += len(pairs)
    if not a.apply:
        print(f"preview only, nothing moved ({total} duplicates). Re-run with --apply after closing X-AnyLabeling.")


if __name__ == "__main__":
    main()
