#!/usr/bin/env python3
"""Merge the old garbage dataset and the Roboflow human+trash dataset into one 2-class YOLO dataset.

Target classes:  0 = garbage, 1 = person

What it does, in order:
 1. Reads each source (Ultralytics layout images/<split> or Roboflow layout <split>/images).
 2. Remaps every source class name to a target class (trash/waste/litter -> garbage,
    human/people/pedestrian -> person). Unknown names stop the run unless --class-map says
    what to do with them.
 3. Auto-labels people in sources that have no person class (the old dataset), using a
    COCO-pretrained YOLO model, so unlabeled people are not learned as "background".
 4. Prevents leakage: Roboflow augmented copies of one photo (`name_jpg.rf.<hash>.jpg`)
    and near-identical images (perceptual hash) may not sit in two different splits.
 5. Copies images + writes remapped labels to --output, writes data.yaml and merge_report.json.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402

TARGET_NAMES = ["garbage", "person"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}
SPLIT_PRIORITY = {"test": 3, "val": 2, "train": 1}  # who keeps an image when two splits clash
KEYWORDS = {
    0: ("garbage", "trash", "waste", "litter", "rubbish", "dump", "debris", "junk", "refuse",
        "plastic", "metal", "paper", "glass"),  # ADDED: Roboflow human+trash material classes
    1: ("person", "human", "people", "pedestrian", "man", "woman", "worker"),
}


# --------------------------------------------------------------------------- reading sources
def load_names(root: Path) -> list[str]:
    """Return class names from <root>/data.yaml (list or {id: name} dict)."""
    dbg("load_names", f"enter root={root}")
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    names = [str(n) for n in names]
    dbg("load_names", f"exit names={names}")
    return names


def build_class_map(names: list[str], overrides: dict[str, str]) -> dict[int, int | None]:
    """Map source class id -> target id (None = drop boxes of that class).

    Overrides (from --class-map name=garbage|person|drop) win over keyword matching.
    """
    dbg("build_class_map", f"enter names={names} overrides={overrides}")
    mapping, unknown = {}, []
    for idx, name in enumerate(names):
        low = name.strip().lower()
        if low in overrides:
            tgt = overrides[low]
            mapping[idx] = None if tgt == "drop" else TARGET_NAMES.index(tgt)
            continue
        hits = [t for t, words in KEYWORDS.items() if any(re.search(rf"\b{w}", low) for w in words)]
        if len(hits) == 1:
            mapping[idx] = hits[0]
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(
            f"Cannot map class name(s) {unknown} to {TARGET_NAMES}. "
            f"Add e.g. --class-map \"{unknown[0]}=garbage\" (or =person / =drop)."
        )
    dbg("build_class_map", f"exit mapping={ {names[k]: (TARGET_NAMES[v] if v is not None else 'drop') for k, v in mapping.items()} }")
    return mapping


def find_split_dirs(root: Path) -> dict[str, tuple[Path, Path]]:
    """Locate (images_dir, labels_dir) for each split in either supported layout."""
    found = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            for img_dir, lab_dir in ((root / "images" / alias, root / "labels" / alias),
                                     (root / alias / "images", root / alias / "labels")):
                if img_dir.is_dir():
                    found[split] = (img_dir, lab_dir)
                    break
            if split in found:
                break
    dbg("find_split_dirs", f"root={root} splits={ {k: str(v[0]) for k, v in found.items()} }")
    return found


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Read a YOLO label file. Polygon rows are converted to their bounding box.

    Returns (boxes, number_of_invalid_rows_dropped). Coordinates are clipped to [0, 1].
    """
    boxes, bad = [], 0
    if not path.is_file():
        return boxes, bad
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            bad += 1
            continue
        if len(vals) == 4:
            xc, yc, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon -> bbox
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            bad += 1
            continue
        # clip the box to the image
        x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        if cls < 0 or x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            bad += 1
            continue
        boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes, bad


def collect_source(name: str, root: Path, overrides: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Return one record per image of a source, with labels already remapped to target ids."""
    dbg("collect_source", f"enter name={name} root={root}")
    if not (root / "data.yaml").is_file():
        raise SystemExit(f"{name}: no data.yaml in {root}")
    names = load_names(root)
    cmap = build_class_map(names, overrides)
    records, invalid = [], 0
    for split, (img_dir, lab_dir) in find_split_dirs(root).items():
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            boxes, bad = parse_label_file(lab_dir / f"{img.stem}.txt")
            invalid += bad
            mapped = [(cmap[c], *b) for c, *b in boxes if c in cmap and cmap[c] is not None]
            records.append({"source": name, "split": split, "image": img, "boxes": mapped,
                            "group": f"{name}:{img.name.split('.rf.')[0]}"})
    if not records:
        raise SystemExit(f"{name}: no images found under {root}")
    dbg("collect_source", f"exit records={len(records)} invalid_rows_dropped={invalid} "
                          f"by_split={dict(Counter(r['split'] for r in records))}")
    return records, names


# --------------------------------------------------------------------------- leakage control
def fix_group_leakage(records: list[dict]) -> int:
    """Keep each Roboflow augmentation group in a single split (the highest-priority one).

    Example: photo_jpg.rf.aaa.jpg in test and photo_jpg.rf.bbb.jpg in train -> drop the train copy.
    Returns the number of records marked as dropped.
    """
    best = {}
    for r in records:
        g = r["group"]
        if g not in best or SPLIT_PRIORITY[r["split"]] > SPLIT_PRIORITY[best[g]]:
            best[g] = r["split"]
    dropped = 0
    for r in records:
        if r["split"] != best[r["group"]]:
            r["drop"] = "augmentation_group_in_other_split"
            dropped += 1
    dbg("fix_group_leakage", f"groups={len(best)} dropped={dropped}")
    return dropped


def dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash: robust to resizing / recompression, not to flips or crops."""
    try:
        with Image.open(path) as im:
            im.draft("L", (size * 4, size * 4))  # fast JPEG decode at reduced size
            px = list(im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR).getdata())
    except Exception:
        return None
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def fix_near_duplicates(records: list[dict], max_distance: int, workers: int) -> Counter:
    """Drop near-identical images that would leak across splits or repeat across sources.

    Rules for a pair within `max_distance` bits:
      * different splits          -> drop the copy in the lower-priority split
      * same split, other source  -> drop the second one (cross-dataset duplicate)
      * same split, same source   -> keep both (likely intentional augmentation copies)
    Uses 4 x 16-bit bands (LSH) so only images sharing a band are compared.
    """
    live = [r for r in records if not r.get("drop")]
    dbg("fix_near_duplicates", f"enter images={len(live)} max_distance={max_distance}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(lambda r: dhash(r["image"]), live))
    reasons = Counter()
    for r, h in zip(live, hashes):
        r["hash"] = h
        if h is None:
            r["drop"] = "unreadable_image"
            reasons["unreadable_image"] += 1
    buckets = defaultdict(list)
    for i, r in enumerate(live):
        if r.get("hash") is None:
            continue
        for band in range(4):
            buckets[(band, (r["hash"] >> (16 * band)) & 0xFFFF)].append(i)
    checked = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:  # huge bucket = flat/blank images, skip
            continue
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                ra, rb = live[a], live[b]
                if ra.get("drop") or rb.get("drop"):
                    continue
                if bin(ra["hash"] ^ rb["hash"]).count("1") > max_distance:
                    continue
                if ra["split"] != rb["split"]:
                    victim = ra if SPLIT_PRIORITY[ra["split"]] < SPLIT_PRIORITY[rb["split"]] else rb
                    victim["drop"] = "near_duplicate_across_splits"
                elif ra["source"] != rb["source"]:
                    victim = rb
                    victim["drop"] = "near_duplicate_across_sources"
                else:
                    continue
                reasons[victim["drop"]] += 1
    dbg("fix_near_duplicates", f"exit pairs_checked={len(checked)} dropped={dict(reasons)}")
    return reasons


# --------------------------------------------------------------------------- person auto-label
def autolabel_people(records, model_name, conf, imgsz, device, batch, cache_path: Path) -> int:
    """Add person boxes (target id 1) predicted by a COCO model to the given records.

    Results are cached per absolute image path in cache_path so re-running the merge is fast.
    Returns the number of person boxes added.
    """
    dbg("autolabel_people", f"enter images={len(records)} model={model_name} conf={conf} device={device}")
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    meta = f"{model_name}|{conf}|{imgsz}"
    if cache.get("_meta") != meta:
        cache = {"_meta": meta}
    todo = [r for r in records if str(r["image"].resolve()) not in cache]
    dbg("autolabel_people", f"cached={len(records) - len(todo)} to_predict={len(todo)}")
    if todo:
        from ultralytics import YOLO  # imported lazily: only needed when auto-labelling

        model = YOLO(model_name)
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            results = model.predict([str(r["image"]) for r in chunk], classes=[0], conf=conf,
                                    imgsz=imgsz, device=device, verbose=False)
            for r, res in zip(chunk, results):
                cache[str(r["image"].resolve())] = [
                    [round(v, 6) for v in box] for box in res.boxes.xywhn.tolist()
                ]
            print(f"  auto-label people: {min(start + batch, len(todo))}/{len(todo)}", end="\r")
            if (start // batch) % 20 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache))
        print()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    added = 0
    for r in records:
        people = cache.get(str(r["image"].resolve()), [])
        r["boxes"] = r["boxes"] + [(1, *b) for b in people]
        r["autolabeled_people"] = len(people)
        added += len(people)
    dbg("autolabel_people", f"exit person_boxes_added={added} "
                            f"images_with_people={sum(1 for r in records if r['autolabeled_people'])}")
    return added


# --------------------------------------------------------------------------- writing
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "img"


def write_dataset(records: list[dict], out: Path, workers: int) -> dict:
    """Copy kept images to out/images/<split>/, write labels, return per-split statistics."""
    keep = [r for r in records if not r.get("drop")]
    dbg("write_dataset", f"enter out={out} images={len(keep)}")
    used = set()
    for r in keep:
        name = safe_name(f"{r['source']}_{r['image'].name}")
        stem, suffix = Path(name).stem, Path(name).suffix
        n = 1
        while (r["split"], name.lower()) in used:
            name = f"{stem}_{n}{suffix}"
            n += 1
        used.add((r["split"], name.lower()))
        r["out_name"] = name

    def _copy(r):
        img_dst = out / "images" / r["split"] / r["out_name"]
        lab_dst = out / "labels" / r["split"] / f"{Path(r['out_name']).stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["image"], img_dst)
        lab_dst.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in r["boxes"]),
                           encoding="utf-8")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_copy, keep))

    stats = defaultdict(lambda: Counter())
    for r in keep:
        s = stats[r["split"]]
        s["images"] += 1
        s[f"images_from_{r['source']}"] += 1
        s["background_images"] += int(not r["boxes"])
        for c, *_ in r["boxes"]:
            s[f"boxes_{TARGET_NAMES[c]}"] += 1
    result = {k: dict(v) for k, v in sorted(stats.items())}
    dbg("write_dataset", f"exit stats={result}")
    return result


def parse_overrides(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--class-map expects name=garbage|person|drop, got {item!r}")
        k, v = (s.strip().lower() for s in item.split("=", 1))
        if v not in TARGET_NAMES + ["drop"]:
            raise SystemExit(f"--class-map target must be garbage, person or drop, got {v!r}")
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, default=Path("dataset"), help="old garbage-only dataset")
    ap.add_argument("--roboflow", type=Path, default=Path("datasets_raw/roboflow"))
    ap.add_argument("--backgrounds", type=Path, default=None,
                    help="optional folder of images with NO garbage/people (e.g. empty frames from "
                         "your cameras); added to train with empty labels to cut false alarms")
    ap.add_argument("--output", type=Path, default=Path("dataset_merged"))
    ap.add_argument("--class-map", nargs="*", default=[], metavar="NAME=TARGET")
    ap.add_argument("--no-autolabel", action="store_true", help="skip person auto-labelling of the old set")
    ap.add_argument("--person-model", default="yolo11x.pt", help="COCO model used to auto-label people")
    ap.add_argument("--person-conf", type=float, default=0.40)
    ap.add_argument("--person-imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cpu, 0, ... (default: Ultralytics picks)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=Path("datasets_raw/autolabel_people_cache.json"))
    ap.add_argument("--dup-distance", type=int, default=4, help="max dHash bit difference for a duplicate")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="delete --output first if it exists")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    enable(args.debug)

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(args.output)
    overrides = parse_overrides(args.class_map)

    old, old_names = collect_source("old", args.old, overrides)
    if not args.roboflow.is_dir():
        raise SystemExit(f"Roboflow dataset not found at {args.roboflow}. Run scripts/download_roboflow.py first.")
    rf, rf_names = collect_source("rf", args.roboflow, overrides)
    records = old + rf

    if args.backgrounds:
        bg = sorted(p for p in args.backgrounds.rglob("*") if p.suffix.lower() in IMG_EXTS)
        records += [{"source": "bg", "split": "train", "image": p, "boxes": [], "group": f"bg:{p}"} for p in bg]
        print(f"Added {len(bg)} background images to train")

    leak = fix_group_leakage(records)
    dups = fix_near_duplicates(records, args.dup_distance, args.workers)

    # Auto-label people only for sources whose original classes contain no person class.
    added = 0
    old_has_person = any(build_class_map([n], overrides).get(0) == 1 for n in old_names)
    if not args.no_autolabel and not old_has_person:
        targets = [r for r in records if r["source"] == "old" and not r.get("drop")]
        print(f"Auto-labelling people in {len(targets)} old images with {args.person_model} ...")
        added = autolabel_people(targets, args.person_model, args.person_conf, args.person_imgsz,
                                 args.device, args.batch, args.cache)

    stats = write_dataset(records, args.output, args.workers)
    yaml_data = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "nc": len(TARGET_NAMES), "names": TARGET_NAMES}
    (args.output / "data.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    report = {
        "sources": {"old": {"path": str(args.old), "names": old_names},
                    "rf": {"path": str(args.roboflow), "names": rf_names}},
        "target_names": TARGET_NAMES,
        "dropped": dict(Counter(r["drop"] for r in records if r.get("drop"))),
        "augmentation_group_leaks_removed": leak,
        "near_duplicates_removed": dict(dups),
        "person_boxes_autolabeled": added,
        "person_model": None if args.no_autolabel else args.person_model,
        "person_conf": args.person_conf,
        "splits": stats,
    }
    (args.output / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for split in ("train", "val", "test"):
        if split not in stats:
            print(f"WARNING: no {split} images in the merged dataset")
    print(f"\nMerged dataset: {args.output}/data.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
