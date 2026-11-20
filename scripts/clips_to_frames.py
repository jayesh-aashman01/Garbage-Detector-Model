#!/usr/bin/env python3
"""Turn CCTV video clips into pre-labelled YOLO frames for human review.

Pipeline per clip:
  1. sample frames at --fps (default 1 per second)
  2. drop near-duplicate frames inside the clip (dHash, reused from merge_datasets.py)
  3. draft labels: person (class 1) from a COCO model (yolo11x), garbage (class 0)
     from YOLO-World with text prompts
  4. write <out>/images/<split>/<clip>_f<frame>.jpg + <out>/labels/<split>/... (YOLO format)
  5. write <out>/review.json: one entry per clip with its split and reviewed=false

The whole clip goes to ONE split (no train/val leakage between frames of the same video).
Labels are DRAFTS: correct them (e.g. in X-AnyLabeling), then set reviewed=true.

Example:
  python scripts/clips_to_frames.py cctv_test/clips/MU5JriIfo5k.mp4 cctv_test/clips/PEyhUwGBrnQ.mp4 \
      --assign MU5JriIfo5k=train PEyhUwGBrnQ=val --out datasets_raw/cctv --debug
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402
from merge_datasets import dhash, safe_name  # noqa: E402

GARBAGE_PROMPTS = [
    "garbage pile", "trash bag", "plastic bag", "litter", "garbage", "rubbish",
    "cardboard box", "plastic bottle", "waste",
]
GDINO_PROMPTS = ["garbage pile", "trash bag", "litter", "rubbish heap", "waste"]
VIDEO_EXT ={".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_assign(items: list[str]) -> dict:
    """['clip=train', ...] -> {'clip': 'train'}. Raises ValueError on bad input."""
    dbg("parse_assign", f"enter items={items}")
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--assign expects CLIP=SPLIT, got {item!r}")
        clip, split = item.split("=", 1)
        if split not in ("train", "val", "test") or not clip:
            raise ValueError(f"--assign: bad clip or split in {item!r} (split must be train/val/test)")
        out[clip] = split
    dbg("parse_assign", f"exit {out}")
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_frames(video: Path, fps: float, tmp_dir: Path) -> list[dict]:
    """Sample frames at `fps` and save them as JPEG. Returns [{frame, time_s, path}].
    Transforms: a video into a short list of still images (about fps * seconds)."""
    dbg("extract_frames", f"enter video={video} fps={fps}")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    dbg("extract_frames", f"src_fps={src_fps:.2f} keep every {step}th frame")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = tmp_dir / f"{safe_name(video.stem)}_f{idx:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out.append({"frame": idx, "time_s": round(idx / src_fps, 2), "path": p})
        idx += 1
    cap.release()
    dbg("extract_frames", f"exit total_frames={idx} sampled={len(out)}")
    return out


def dedupe_frames(frames: list[dict], max_distance: int) -> tuple[list[dict], int]:
    """Keep a frame only if its dHash differs from the last KEPT frame by > max_distance bits.
    Unreadable frames are dropped. Returns (kept, dropped_count)."""
    dbg("dedupe_frames", f"enter frames={len(frames)} max_distance={max_distance}")
    kept, last = [], None
    for f in frames:
        h = dhash(f["path"])
        if h is None:
            continue
        if last is not None and hamming(h, last) <= max_distance:
            f["path"].unlink(missing_ok=True)
            continue
        kept.append(f)
        last = h
    dropped = len(frames) - len(kept)
    dbg("dedupe_frames", f"exit kept={len(kept)} dropped={dropped}")
    return kept, dropped


class _Boxes:
    """Minimal stand-in for Ultralytics `result.boxes` (only what draft_labels/to_yolo_lines use)."""

    def __init__(self, xywhn: list):
        self._xywhn = xywhn

    @property
    def xywhn(self):
        rows = self._xywhn
        return type("T", (), {"tolist": staticmethod(lambda: rows)})()

    def __len__(self):
        return len(self._xywhn)


def gdino_prompt(prompts: list[str]) -> str:
    """['a', 'B c'] -> 'a. b c.' (Grounding DINO wants lowercase phrases ending with '.')."""
    return " ".join(p.strip().lower().rstrip(".") + "." for p in prompts if p.strip())


class GroundingDinoDetector:
    """Grounding DINO (Hugging Face transformers) wrapped to look like an Ultralytics model:
    predict(path, conf=..., ...) -> [result] with result.boxes.xywhn. All phrases map to one class.
    Overlapping boxes from different phrases are merged with class-agnostic NMS."""

    def __init__(self, model_id: str, prompts: list[str], text_threshold: float = 0.25,
                 nms_iou: float = 0.5, device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        dbg("GroundingDinoDetector", f"enter model={model_id} prompts={prompts} "
                                     f"text_threshold={text_threshold} nms_iou={nms_iou}")
        self.torch = torch
        # MODIFIED: device comes from gpu_utils.resolve_device ("0", "0,1" or "cpu")
        if device not in (None, "cpu") and torch.cuda.is_available():
            self.device = f"cuda:{str(device).split(',')[0].replace('cuda:', '')}"
        else:
            self.device = "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.text = gdino_prompt(prompts)
        self.text_threshold, self.nms_iou = text_threshold, nms_iou
        dbg("GroundingDinoDetector", f"exit device={self.device} text={self.text!r}")

    def predict(self, path, conf=0.3, **_):
        from PIL import Image
        from torchvision.ops import nms

        image = Image.open(path).convert("RGB")
        w, h = image.size
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=conf, text_threshold=self.text_threshold,
            target_sizes=[(h, w)])[0]
        boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
        if len(boxes):
            keep = nms(boxes.float(), scores.float(), self.nms_iou)
            boxes, scores = boxes[keep], scores[keep]
        xywhn = []
        for x1, y1, x2, y2 in boxes.tolist():
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            xywhn.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        dbg("GroundingDinoDetector.predict", f"{Path(path).name}: raw={len(res['boxes'])} "
                                             f"after_nms={len(xywhn)} scores={[round(s, 2) for s in scores.tolist()]}")
        return [type("R", (), {"boxes": _Boxes(xywhn)})()]


def to_yolo_lines(result, cls_out: int) -> list[str]:
    """Ultralytics result -> 'cls xc yc w h' lines (normalised), all boxes mapped to cls_out."""
    lines = []
    for xc, yc, w, h in result.boxes.xywhn.tolist():
        lines.append(f"{cls_out} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def draft_labels(paths: list[Path], person_model, garbage_model, person_conf: float,
                 garbage_conf: float, imgsz: int, device) -> dict:
    """Run both models on every frame. Returns {path: [yolo lines]} (garbage first, then person).
    Transforms: raw frames into draft class-0 (garbage) and class-1 (person) boxes."""
    dbg("draft_labels", f"enter frames={len(paths)} person_conf={person_conf} garbage_conf={garbage_conf}")
    out = {p: [] for p in paths}
    for p in paths:
        g = garbage_model.predict(str(p), conf=garbage_conf, imgsz=imgsz, device=device, verbose=False)[0]
        h = person_model.predict(str(p), conf=person_conf, imgsz=imgsz, device=device,
                                 classes=[0], verbose=False)[0]
        out[p] = to_yolo_lines(g, 0) + to_yolo_lines(h, 1)
        dbg("draft_labels", f"{p.name}: garbage={len(g.boxes)} person={len(h.boxes)}")
    dbg("draft_labels", "exit")
    return out


# ADDED (2026-11-20): post-filters for garbage drafts, stable clip->split, batch helpers
def _parse_line(line: str):
    c, xc, yc, w, h = line.split()[:5]
    return int(c), float(xc), float(yc), float(w), float(h)


def _inside_frac(a, b) -> float:
    """Fraction of box a's area that lies inside box b (boxes as xc, yc, w, h)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    area = a[2] * a[3]
    return (iw * ih) / area if area > 0 else 0.0


def filter_garbage_boxes(lines: list[str], max_area: float, nested_frac: float) -> tuple[list[str], dict]:
    """Clean garbage (class 0) drafts; person lines pass through unchanged.
    - drop a garbage box covering more than `max_area` of the image (0 or >=1 disables)
    - drop a garbage box when more than `nested_frac` of it lies inside a LARGER garbage box
      (one box per pile; 0 disables)
    Returns (kept_lines, {'too_big': n, 'nested': n})."""
    dbg("filter_garbage_boxes", f"enter lines={len(lines)} max_area={max_area} nested_frac={nested_frac}")
    stats = {"too_big": 0, "nested": 0}
    garbage = [(l, _parse_line(l)[1:]) for l in lines if l.startswith("0 ")]
    others = [l for l in lines if not l.startswith("0 ")]
    if 0 < max_area < 1:
        before = len(garbage)
        garbage = [(l, b) for l, b in garbage if b[2] * b[3] <= max_area]
        stats["too_big"] = before - len(garbage)
    kept = []
    for l, b in garbage:
        nested = nested_frac > 0 and any(
            o is not b and o[2] * o[3] > b[2] * b[3] and _inside_frac(b, o) > nested_frac
            for _, o in garbage)
        if nested:
            stats["nested"] += 1
        else:
            kept.append(l)
    dbg("filter_garbage_boxes", f"exit kept_garbage={len(kept)} person={len(others)} dropped={stats}")
    return kept + others, stats


def default_garbage_conf(backend: str) -> float:
    """Default garbage draft confidence per backend. gdino 0.30: at 0.40 real litter (~0.34) was lost
    in the 2026-11-20 smoke test; vehicles score similarly, so they are removed in human review."""
    if backend not in ("gdino", "yolo-world"):
        raise ValueError(f"unknown garbage backend {backend!r}")
    return 0.30 if backend == "gdino" else 0.15


def auto_split(clip: str, val_pct: int, test_pct: int) -> str:
    """Stable clip -> split from an md5 of the clip name, so re-runs and new clips never move
    an existing clip to another split. Raises ValueError on bad percentages."""
    if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
        raise ValueError("--val-pct/--test-pct must be >= 0 and sum to < 100")
    bucket = int(hashlib.md5(clip.encode()).hexdigest(), 16) % 100
    split = "val" if bucket < val_pct else "test" if bucket < val_pct + test_pct else "train"
    dbg("auto_split", f"{clip}: bucket={bucket} -> {split}")
    return split


def expand_videos(paths: list[Path]) -> list[Path]:
    """Files are kept; folders are searched recursively for video files (sorted)."""
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(v for v in p.rglob("*") if v.suffix.lower() in VIDEO_EXT)
        elif p.suffix.lower() in VIDEO_EXT:
            out.append(p)
    dbg("expand_videos", f"inputs={len(paths)} videos={len(out)}")
    return out


def mark_reviewed(out: Path, clips: list[str]) -> list[str]:
    """Set reviewed=true for the given clips (or every clip for ['all']). Returns clips changed.
    Raises SystemExit if review.json is missing or a clip is unknown."""
    path = out / "review.json"
    if not path.exists():
        raise SystemExit(f"no review.json in {out}")
    data = json.loads(path.read_text())
    names = list(data) if clips == ["all"] else clips
    unknown = [c for c in names if c not in data]
    if unknown:
        raise SystemExit(f"unknown clip(s): {unknown}. Known: {sorted(data)}")
    for c in names:
        data[c]["reviewed"] = True
    path.write_text(json.dumps(data, indent=2))
    dbg("mark_reviewed", f"marked={names}")
    return names


def write_clip(frames: list[dict], labels: dict, out: Path, split: str) -> int:
    """Move kept frames into out/images/<split>/ and write matching label files."""
    dbg("write_clip", f"enter frames={len(frames)} split={split}")
    img_dir, lab_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        dest = img_dir / f["path"].name
        f["path"].replace(dest)
        (lab_dir / f"{dest.stem}.txt").write_text("".join(f"{l}\n" for l in labels.get(f["path"], [])))
    dbg("write_clip", f"exit wrote={len(frames)} to {img_dir}")
    return len(frames)


def update_review(out: Path, clip: str, entry: dict) -> None:
    """Add/replace one clip in out/review.json. Keeps reviewed=true if already set."""
    path = out / "review.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if data.get(clip, {}).get("reviewed"):
        entry["reviewed"] = True
    data[clip] = entry
    path.write_text(json.dumps(data, indent=2))
    dbg("update_review", f"{clip} -> {entry}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", type=Path, help="video files and/or folders of videos")
    ap.add_argument("--out", type=Path, default=Path("datasets_raw/cctv"))
    ap.add_argument("--assign", nargs="*", default=[],
                    help="CLIP=train|val|test; clips not listed get a stable automatic split")
    ap.add_argument("--val-pct", type=int, default=15, help="automatic split: %% of clips to val")
    ap.add_argument("--test-pct", type=int, default=0, help="automatic split: %% of clips to test")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop starting new clips once review.json holds this many frames (0 = no limit)")
    ap.add_argument("--max-garbage-area", type=float, default=0.8,
                    help="drop garbage drafts covering more than this fraction of the image (0 = off)")
    ap.add_argument("--nested-frac", type=float, default=0.8,
                    help="drop a garbage draft this much inside a larger garbage draft (0 = off)")
    ap.add_argument("--mark-reviewed", nargs="+", default=None, metavar="CLIP",
                    help="only set reviewed=true for these clips (or 'all') and exit")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dedupe-distance", type=int, default=6, help="dHash bits; <= this = duplicate")
    ap.add_argument("--person-model", default="yolo11x.pt")
    ap.add_argument("--garbage-backend", choices=["yolo-world", "gdino"], default="yolo-world",
                    help="draft garbage with YOLO-World (Ultralytics) or Grounding DINO (transformers)")
    ap.add_argument("--garbage-model", default=None,
                    help="default: yolov8m-worldv2.pt (yolo-world) or IDEA-Research/grounding-dino-tiny (gdino)")
    ap.add_argument("--gdino-text-threshold", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="gdino: merge overlapping garbage boxes")
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--garbage-conf", type=float, default=None,
                    help="default: 0.15 (yolo-world) or 0.30 (gdino)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="auto (CUDA if available), cpu, 0, ...")
    ap.add_argument("--overwrite", action="store_true", help="re-process a clip already in review.json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    enable(args.debug)

    if args.mark_reviewed:
        done = mark_reviewed(args.out, args.mark_reviewed)
        print(f"reviewed=true: {done}")
        return 0

    try:
        assign = parse_assign(args.assign)
        auto_split("check", args.val_pct, args.test_pct)
    except ValueError as exc:
        raise SystemExit(str(exc))
    missing = [str(v) for v in args.videos if not v.exists()]
    videos = expand_videos(args.videos)
    if not videos or missing:
        raise SystemExit(f"no usable videos (missing: {missing})")

    from gpu_utils import resolve_device
    args.device = resolve_device(args.device)
    if args.garbage_conf is None:
        args.garbage_conf = default_garbage_conf(args.garbage_backend)
    print(f"device={args.device} garbage_backend={args.garbage_backend} garbage_conf={args.garbage_conf} "
          f"videos={len(videos)} max_frames={args.max_frames or 'no limit'}")

    from ultralytics import YOLO
    person_model = YOLO(args.person_model)
    if args.garbage_backend == "gdino":
        args.garbage_model = args.garbage_model or "IDEA-Research/grounding-dino-tiny"
        garbage_model = GroundingDinoDetector(args.garbage_model, GDINO_PROMPTS, args.gdino_text_threshold,
                                              args.nms_iou, args.device)
    else:
        from ultralytics import YOLOWorld
        args.garbage_model = args.garbage_model or "yolov8m-worldv2.pt"
        garbage_model = YOLOWorld(args.garbage_model)
        garbage_model.set_classes(GARBAGE_PROMPTS)
    dbg("main", f"garbage backend={args.garbage_backend} model={args.garbage_model}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 2\nnames:\n- garbage\n- person\n")
    (args.out / "classes.txt").write_text("garbage\nperson\n")  # ADDED: X-AnyLabeling YOLO import/export
    review = json.loads((args.out / "review.json").read_text()) if (args.out / "review.json").exists() else {}

    for video in videos:
        clip = safe_name(video.stem)
        total = sum(e.get("frames", 0) for e in review.values())
        if args.max_frames and total >= args.max_frames:
            print(f"reached --max-frames {args.max_frames} (have {total}); stopping")
            break
        split = assign.get(video.stem, assign.get(clip)) or auto_split(clip, args.val_pct, args.test_pct)
        if clip in review and not args.overwrite:
            print(f"skip {clip}: already processed (use --overwrite)")
            continue
        t0 = time.time()
        frames = extract_frames(video, args.fps, args.out / "_tmp" / clip)
        kept, dropped = dedupe_frames(frames, args.dedupe_distance)
        labels = draft_labels([f["path"] for f in kept], person_model, garbage_model,
                              args.person_conf, args.garbage_conf, args.imgsz, args.device)
        filtered = {"too_big": 0, "nested": 0}
        for p in labels:
            labels[p], st = filter_garbage_boxes(labels[p], args.max_garbage_area, args.nested_frac)
            filtered = {k: filtered[k] + st[k] for k in filtered}
        n = write_clip(kept, labels, args.out, split)
        g = sum(1 for ls in labels.values() for l in ls if l.startswith("0 "))
        p = sum(1 for ls in labels.values() for l in ls if l.startswith("1 "))
        entry = {"source": str(video), "split": split, "frames": n, "duplicates_dropped": dropped,
                 "draft_garbage_boxes": g, "draft_person_boxes": p, "reviewed": False,
                 "garbage_filtered": filtered, "device": args.device,
                 "garbage_backend": args.garbage_backend,
                 "garbage_model": args.garbage_model, "garbage_conf": args.garbage_conf,
                 "person_model": args.person_model, "person_conf": args.person_conf}
        update_review(args.out, clip, entry)
        review[clip] = entry
        try:
            (args.out / "_tmp" / clip).rmdir()
        except OSError:
            pass
        print(f"{clip}: split={split} frames={n} dup_dropped={dropped} "
              f"garbage={g} person={p} filtered={filtered} ({time.time() - t0:.0f}s)")
    try:
        (args.out / "_tmp").rmdir()
    except OSError:
        pass
    print(f"Saved to {args.out}/  (labels are DRAFTS: review, then set reviewed=true in review.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
