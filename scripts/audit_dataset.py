#!/usr/bin/env python3
"""Check image/label pairing, image readability and YOLO label ranges."""
from pathlib import Path
from collections import Counter
from PIL import Image
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset"))
    args = p.parse_args()
    root = args.root
    issues, counts, objects = [], Counter(), Counter()
    for split in ("train", "val", "test"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = sorted([x for x in image_dir.glob("*") if x.is_file()]) if image_dir.exists() else []
        labels = {x.stem: x for x in label_dir.glob("*.txt")} if label_dir.exists() else {}
        counts[split] = len(images)
        for image in images:
            try:
                with Image.open(image) as im:
                    im.verify()
            except Exception as e:
                issues.append(f"Unreadable image {image}: {e}")
            label = labels.get(image.stem)
            if label is None:
                issues.append(f"Missing label file: {image}")
                continue
            for n, line in enumerate(label.read_text().splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"Malformed label {label}:{n}")
                    continue
                try:
                    cls = int(parts[0]); vals = list(map(float, parts[1:]))
                    if cls < 0 or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                        raise ValueError("out of range")
                    objects[split] += 1
                except Exception:
                    issues.append(f"Invalid YOLO values {label}:{n} -> {line}")
        extra = set(labels) - {x.stem for x in images}
        for stem in sorted(extra):
            issues.append(f"Orphan label: {label_dir / (stem + '.txt')}")
    print("Images:", dict(counts))
    print("Objects:", dict(objects))
    print("Issues:", len(issues))
    for issue in issues[:100]:
        print("-", issue)
    if len(issues) > 100:
        print(f"... and {len(issues)-100} more")
    if issues:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
