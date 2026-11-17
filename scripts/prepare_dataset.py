#!/usr/bin/env python3
"""Download a JSONL image manifest and convert its normalized boxes to YOLO labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

LOG = logging.getLogger("prepare_dataset")
VALID_SPLITS = {"train", "val", "test"}


def safe_filename(name: str, fallback: str) -> str:
    # Do not allow manifest values to escape the destination directory.
    name = Path(str(name or fallback)).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or fallback


def read_manifest(path: Path):
    metadata = {}
    records = []
    malformed = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                malformed.append({"line": line_no, "error": str(exc), "reason": "invalid_json"})
                continue
            if item.get("type") == "dataset":
                metadata = item
            elif item.get("type") == "image":
                records.append((line_no, item))
    return metadata, records, malformed


def normalize_box(box, line_no):
    if not isinstance(box, (list, tuple)) or len(box) != 5:
        raise ValueError(f"line {line_no}: expected [class_id, x_center, y_center, width, height]")
    cls, xc, yc, w, h = box
    cls = int(cls)
    vals = [float(xc), float(yc), float(w), float(h)]
    if cls < 0 or any(v < 0 or v > 1 for v in vals) or vals[2] == 0 or vals[3] == 0:
        raise ValueError(f"line {line_no}: invalid YOLO box {box!r}")
    return cls, *vals


def build_records(records, malformed, missing_policy, val_fraction, seed):
    accepted, rejected = [], []
    seen_urls = set()
    for line_no, item in records:
        url = item.get("url")
        filename = safe_filename(item.get("file"), f"image_{line_no}.jpg")
        if not url or urlparse(url).scheme not in ("http", "https"):
            rejected.append({"line": line_no, "file": filename, "reason": "missing_or_invalid_url"})
            continue
        if url in seen_urls:
            rejected.append({"line": line_no, "file": filename, "reason": "duplicate_url"})
            continue
        seen_urls.add(url)

        ann = item.get("annotations")
        if not isinstance(ann, dict) or "boxes" not in ann:
            if missing_policy == "empty":
                boxes = []
            else:
                rejected.append({"line": line_no, "file": filename, "reason": "missing_annotations"})
                continue
        else:
            raw_boxes = ann.get("boxes")
            if not isinstance(raw_boxes, list):
                rejected.append({"line": line_no, "file": filename, "reason": "invalid_boxes"})
                continue
            try:
                boxes = [normalize_box(b, line_no) for b in raw_boxes]
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append({"line": line_no, "file": filename, "reason": str(exc)})
                continue

        split = str(item.get("split", "")).lower()
        if split not in VALID_SPLITS:
            rejected.append({"line": line_no, "file": filename, "reason": f"unknown_split:{split}"})
            continue
        rec = {
            "line": line_no, "file": filename, "url": url, "split": split,
            "boxes": boxes, "width": item.get("width"), "height": item.get("height")
        }
        accepted.append(rec)

    # Preserve the provided test split. Derive validation only from original train records.
    train = [r for r in accepted if r["split"] == "train"]
    random.Random(seed).shuffle(train)
    n_val = max(1, round(len(train) * val_fraction)) if len(train) > 1 and val_fraction > 0 else 0
    val_ids = {r["line"] for r in train[:n_val]}
    for rec in accepted:
        if rec["split"] == "train" and rec["line"] in val_ids:
            rec["split"] = "val"
    return accepted, rejected


def make_session():
    session = requests.Session()
    retry = Retry(
        total=4, connect=4, read=4, status=4, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "dataset-preparer/1.0"})
    return session


def download_one(rec, images_root: Path, timeout: int, overwrite: bool):
    split_dir = images_root / rec["split"]
    split_dir.mkdir(parents=True, exist_ok=True)
    target = split_dir / rec["file"]
    if target.exists() and not overwrite:
        try:
            with Image.open(target) as im:
                im.verify()
            return rec, True, "already_valid"
        except Exception:
            target.unlink(missing_ok=True)

    part = target.with_suffix(target.suffix + ".part")
    session = make_session()
    try:
        with session.get(rec["url"], stream=True, timeout=(10, timeout)) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError(f"server returned HTML instead of an image ({content_type})")
            with part.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        out.write(chunk)
        with Image.open(part) as im:
            im.verify()
        os.replace(part, target)
        return rec, True, "downloaded"
    except Exception as exc:
        part.unlink(missing_ok=True)
        return rec, False, f"{type(exc).__name__}: {exc}"
    finally:
        session.close()


def write_labels(records, dataset_root: Path):
    for rec in records:
        label_dir = dataset_root / "labels" / rec["split"]
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{Path(rec['file']).stem}.txt"
        # Empty annotation list intentionally creates an empty label file (negative image).
        content = "".join(
            f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            for cls, xc, yc, w, h in rec["boxes"]
        )
        label_path.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("dataset"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--missing-annotations", choices=["skip", "empty"], default="skip",
                    help="Missing annotations are ambiguous. Default skips; use empty only if they are known negatives.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.manifest.is_file():
        ap.error(f"manifest not found: {args.manifest}")
    if args.workers < 1 or not 0 <= args.val_fraction < 1:
        ap.error("workers must be >=1 and val-fraction must be in [0,1)")

    metadata, raw_records, malformed = read_manifest(args.manifest)
    if not raw_records:
        ap.error("no image records found in manifest")

    records, rejected = build_records(
        raw_records, malformed, args.missing_annotations, args.val_fraction, args.seed
    )
    if not records:
        ap.error("no usable records remain after validation")

    # Resolve filename collisions within a split without overwriting a different image.
    used = set()
    for rec in records:
        key = (rec["split"], rec["file"])
        if key in used:
            stem, suffix = Path(rec["file"]).stem, Path(rec["file"]).suffix
            rec["file"] = f"{stem}_{hashlib.sha1(rec['url'].encode()).hexdigest()[:8]}{suffix}"
        used.add((rec["split"], rec["file"]))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_out = args.output / "source_manifest.jsonl"
    shutil.copy2(args.manifest, manifest_out)

    LOG.info("Downloading %d usable images with %d workers", len(records), args.workers)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, r, args.output / "images", args.timeout, args.overwrite)
                   for r in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            results.append(future.result())

    successful = [rec for rec, ok, _ in results if ok]
    failed = [{"file": rec["file"], "split": rec["split"], "url": rec["url"], "error": msg}
              for rec, ok, msg in results if not ok]

    # Only write labels for images successfully present on disk.
    write_labels(successful, args.output)

    # Generate Ultralytics data config.
    classes = metadata.get("class_names") or {"0": "garbage"}
    names = [str(classes[k]) for k in sorted(classes, key=lambda x: int(x))]
    yaml_data = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    report = {
        "source_manifest": str(args.manifest),
        "dataset_name": metadata.get("name"),
        "raw_image_records": len(raw_records),
        "malformed_lines": malformed,
        "accepted_records": len(records),
        "downloaded_or_already_valid": len(successful),
        "failed_downloads": failed,
        "rejected_records": rejected,
        "successful_by_split": dict(Counter(r["split"] for r in successful)),
        "objects_by_split": dict(Counter(
            r["split"] for r in successful for _ in r["boxes"]
        )),
        "missing_annotation_policy": args.missing_annotations,
        "validation_fraction_from_train": args.val_fraction,
        "seed": args.seed,
    }
    (args.output / "download_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Finished: %d/%d images available", len(successful), len(records))
    LOG.info("Dataset config: %s", args.output / "data.yaml")
    LOG.info("Review report: %s", args.output / "download_report.json")
    if failed:
        LOG.warning("%d downloads failed. Re-run the same command to retry.", len(failed))


if __name__ == "__main__":
    main()
