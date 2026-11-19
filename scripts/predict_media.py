#!/usr/bin/env python3
"""Run the garbage/person detector on images, videos, folders, RTSP/HTTP streams or a webcam.

Outputs (in --out/<name>/):
  * annotated images (<image>.jpg) and annotated videos (<video>_pred.mp4)
  * detections.jsonl  - one line per processed frame with every box
  * summary.json      - per-source totals, video alert events, optional accuracy
  * printed summary table

Video logic ("is garbage present?"): garbage is static, people and cars move. A frame
only counts toward an alert when garbage is detected, and an alert turns ON only when
garbage was seen in at least --persist of the last --window sampled frames. This removes
most one-frame false alarms.

Image-level accuracy: pass --labels DIR (YOLO label files named like the images). A frame
is "garbage present" in ground truth if its label file has at least one class-0 line.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def expand_sources(sources: list[str]) -> list[tuple[str, str]]:
    """Turn CLI sources into (kind, path) pairs. kind is image, video or stream."""
    dbg("expand_sources", f"enter sources={sources}")
    out = []
    for src in sources:
        if src.isdigit() or src.lower().startswith(STREAM_PREFIXES):
            out.append(("stream", src))
            continue
        p = Path(src)
        paths = sorted(Path(x) for x in glob.glob(src)) if any(c in src for c in "*?[") else \
            sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
        for q in paths:
            ext = q.suffix.lower()
            if not q.is_file():
                raise SystemExit(f"Source not found: {q}")
            if ext in IMAGE_EXTS:
                out.append(("image", str(q)))
            elif ext in VIDEO_EXTS:
                out.append(("video", str(q)))
    if not out:
        raise SystemExit(f"No images or videos found in {sources}")
    dbg("expand_sources", f"exit images={sum(k == 'image' for k, _ in out)} "
                          f"videos={sum(k == 'video' for k, _ in out)} streams={sum(k == 'stream' for k, _ in out)}")
    return out


def class_ids(names: dict) -> tuple[int, int | None]:
    """Find the garbage and person class ids from the model's names (works for old 1-class models)."""
    lookup = {str(v).lower(): int(k) for k, v in names.items()}
    garbage = lookup.get("garbage", lookup.get("trash", 0))
    person = lookup.get("person", lookup.get("human"))
    dbg("class_ids", f"names={names} -> garbage={garbage} person={person}")
    return garbage, person


def frame_detections(result, garbage_id, person_id) -> dict:
    """Summarise one Ultralytics result into plain numbers + box list."""
    boxes = []
    for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                           result.boxes.cls.tolist()):
        boxes.append({"cls": int(cls), "name": result.names[int(cls)], "conf": round(conf, 4),
                      "xyxy": [round(v, 1) for v in (x1, y1, x2, y2)]})
    g = [b for b in boxes if b["cls"] == garbage_id]
    return {
        "boxes": boxes,
        "garbage": len(g),
        "garbage_max_conf": max((b["conf"] for b in g), default=0.0),
        "person": sum(b["cls"] == person_id for b in boxes) if person_id is not None else 0,
    }


class PersistenceAlert:
    """Sliding-window vote: alert ON when garbage seen in >= persist of the last `window` frames."""

    def __init__(self, persist: int, window: int):
        if not 1 <= persist <= window:
            raise ValueError(f"need 1 <= persist ({persist}) <= window ({window})")
        self.persist, self.hist, self.on = persist, deque(maxlen=window), False

    def update(self, garbage_seen: bool) -> str | None:
        """Feed one frame. Returns 'start' / 'end' when the alert state flips, else None."""
        self.hist.append(bool(garbage_seen))
        now_on = sum(self.hist) >= self.persist
        change = None if now_on == self.on else ("start" if now_on else "end")
        self.on = now_on
        return change


def run_image(model, path, args, ids, out_dir, log) -> dict:
    res = model.predict(path, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
    det = frame_detections(res, *ids)
    if not args.no_save:
        cv2.imwrite(str(out_dir / f"{Path(path).stem}.jpg"), res.plot())
    log.write(json.dumps({"source": path, "frame": 0, **det}) + "\n")
    dbg("run_image", f"{Path(path).name}: garbage={det['garbage']} (max {det['garbage_max_conf']}) person={det['person']}")
    return {"kind": "image", "garbage_present": det["garbage"] > 0, **{k: det[k] for k in
            ("garbage", "garbage_max_conf", "person")}}


def run_video(model, src, kind, args, ids, out_dir, log) -> dict:
    """Sample frames at --sample-fps, predict, apply the persistence alert, write annotated video."""
    dbg("run_video", f"enter src={src} kind={kind} sample_fps={args.sample_fps}")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        return {"kind": kind, "error": "could not open source"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / args.sample_fps))
    alert = PersistenceAlert(args.persist, args.window)
    writer, idx, sampled, hit_frames, events, t0 = None, 0, 0, 0, [], time.time()
    stem = "stream" if kind == "stream" else Path(src).stem
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ts = idx / fps
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        det = frame_detections(res, *ids)
        change = alert.update(det["garbage"] > 0)
        if change == "start":
            events.append({"start_s": round(ts, 2), "end_s": None})
        elif change == "end" and events:
            events[-1]["end_s"] = round(ts, 2)
        hit_frames += det["garbage"] > 0
        log.write(json.dumps({"source": src, "frame": idx, "time_s": round(ts, 2), "alert": alert.on, **det}) + "\n")
        if not args.no_save:
            plotted = res.plot()
            label = "GARBAGE ALERT" if alert.on else "clear"
            cv2.putText(plotted, label, (10, plotted.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if alert.on else (0, 200, 0), 2, cv2.LINE_AA)
            if writer is None:
                h, w = plotted.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / f"{stem}_pred.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, fps / step), (w, h))
            writer.write(plotted)
        sampled += 1
        idx += 1
        elapsed = (time.time() - t0) if kind == "stream" else ts  # wall clock for live streams
        if args.max_seconds and elapsed > args.max_seconds:
            break
    cap.release()
    if writer:
        writer.release()
    if events and events[-1]["end_s"] is None:
        events[-1]["end_s"] = round(idx / fps, 2)
    summary = {"kind": kind, "frames_sampled": sampled, "frames_with_garbage": hit_frames,
               "garbage_present": bool(events), "alert_events": events,
               "seconds_processed": round(idx / fps, 2)}
    dbg("run_video", f"exit {summary}")
    return summary


def image_accuracy(per_source: dict, labels_dir: Path) -> dict:
    """Image-level 'garbage present?' confusion matrix against YOLO label files (class 0 = garbage)."""
    tp = fp = fn = tn = 0
    missing = 0
    for src, s in per_source.items():
        if s.get("kind") != "image":
            continue
        lab = labels_dir / f"{Path(src).stem}.txt"
        if not lab.is_file():
            missing += 1
            continue
        truth = any(line.split()[:1] == ["0"] for line in lab.read_text().splitlines())
        pred = s["garbage_present"]
        tp += truth and pred
        fp += (not truth) and pred
        fn += truth and not pred
        tn += (not truth) and not pred
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "images_without_label_file": missing,
           "precision": round(prec, 4), "recall": round(rec, 4),
           "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
           "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4)}
    dbg("image_accuracy", f"exit {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="image/video file, folder, glob, rtsp:// URL or webcam index")
    ap.add_argument("--model", default="runs/detect/runs/detect/garbage-person-yolo11m/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cpu, 0, ... (default: Ultralytics picks)")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="frames per second analysed in videos")
    ap.add_argument("--persist", type=int, default=3, help="garbage frames needed inside the window")
    ap.add_argument("--window", type=int, default=5, help="sliding window length in sampled frames")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop each video/stream after N seconds (0 = all)")
    ap.add_argument("--labels", type=Path, default=None, help="YOLO labels folder for image-level accuracy")
    ap.add_argument("--out", type=Path, default=Path("runs/predict_media"))
    ap.add_argument("--name", default=None, help="sub-folder name (default: timestamp)")
    ap.add_argument("--no-save", action="store_true", help="do not write annotated images/videos")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    enable(args.debug)
    dbg("main", f"enter args={vars(args)}")

    if not Path(args.model).is_file():
        raise SystemExit(f"Model not found: {args.model}. Train first or pass --model.")
    PersistenceAlert(args.persist, args.window)  # validate early
    sources = expand_sources(args.source)

    from ultralytics import YOLO  # imported late so --help works without loading torch

    model = YOLO(args.model)
    ids = class_ids(model.names)
    out_dir = args.out / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    per_source = {}
    with (out_dir / "detections.jsonl").open("w", encoding="utf-8") as log:
        for i, (kind, src) in enumerate(sources, 1):
            print(f"[{i}/{len(sources)}] {kind}: {src}")
            if kind == "image":
                per_source[src] = run_image(model, src, args, ids, out_dir, log)
            else:
                per_source[src] = run_video(model, src, kind, args, ids, out_dir, log)

    images = [s for s in per_source.values() if s.get("kind") == "image"]
    summary = {
        "model": args.model, "conf": args.conf, "imgsz": args.imgsz, "class_names": model.names,
        "persist": args.persist, "window": args.window, "sample_fps": args.sample_fps,
        "images": len(images), "images_with_garbage": sum(s["garbage_present"] for s in images),
        "videos": sum(s.get("kind") in ("video", "stream") for s in per_source.values()),
        "videos_with_alert": sum(bool(s.get("alert_events")) for s in per_source.values()),
        "per_source": per_source,
    }
    if args.labels:
        summary["image_accuracy"] = image_accuracy(per_source, args.labels)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'source':60} {'garbage':>8} {'person':>7}  result")
    for src, s in per_source.items():
        if s.get("kind") == "image":
            print(f"{Path(src).name[:60]:60} {s['garbage']:>8} {s['person']:>7}  "
                  f"{'GARBAGE' if s['garbage_present'] else 'clear'}")
        else:
            ev = ", ".join(f"{e['start_s']}-{e['end_s']}s" for e in s.get("alert_events", [])) or "none"
            print(f"{Path(src).name[:60]:60} {s.get('frames_with_garbage', 0):>8} {'':>7}  alerts: {ev}")
    print(f"\nImages flagged: {summary['images_with_garbage']}/{summary['images']} | "
          f"videos with alert: {summary['videos_with_alert']}/{summary['videos']}")
    if "image_accuracy" in summary:
        print(f"Image-level accuracy: {summary['image_accuracy']}")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
