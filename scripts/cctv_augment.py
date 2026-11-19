#!/usr/bin/env python3
"""CCTV-style image degradations used during training (Albumentations).

Why: the training photos are clean and sharp, but real CCTV frames are low resolution,
heavily compressed (JPEG/H.264), blurry, noisy, often grayscale infrared at night, and
carry a burned-in timestamp. Training on degraded copies teaches the model to cope.

These transforms change pixels only (no geometry), so YOLO boxes stay valid.
Ultralytics applies them after mosaic/affine via  model.train(augmentations=[...]).

Preview what they do:
    python scripts/cctv_augment.py --preview dataset_merged/images/val --out runs/aug_preview
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402

LEVELS = ("off", "light", "medium", "heavy")


class TimestampOverlay(A.ImageOnlyTransform):
    """Burn a fake CCTV date/time + camera name into a corner of the image.

    Real cameras print this text over the scene; detectors that never saw it during
    training can mistake it for objects or miss objects behind it.
    """

    def __init__(self, p: float = 0.3):
        super().__init__(p=p)

    def get_params(self):
        rnd = self.py_random
        y, mo, d = rnd.randint(2015, 2026), rnd.randint(1, 12), rnd.randint(1, 28)
        hh, mm, ss = rnd.randint(0, 23), rnd.randint(0, 59), rnd.randint(0, 59)
        day = rnd.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        style = rnd.choice([
            f"{y:04d}-{mo:02d}-{d:02d} {day} {hh:02d}:{mm:02d}:{ss:02d}",
            f"{mo:02d}-{d:02d}-{y:04d} {day} {hh:02d}:{mm:02d}:{ss:02d}",
            f"{d:02d}/{mo:02d}/{y:04d} {hh:02d}:{mm:02d}:{ss:02d}",
        ])
        return {
            "text": style,
            "cam": rnd.choice(["", "CAM 01", "CH2", "Camera 3", "IPC", "D04"]),
            "corner": rnd.choice(["tl", "tr", "bl", "br"]),
            "color": rnd.choice([(255, 255, 255), (255, 255, 0), (230, 230, 230)]),
            "scale": rnd.uniform(0.6, 1.2),
        }

    def apply(self, img, text="", cam="", corner="tl", color=(255, 255, 255), scale=1.0, **_):
        out = img.copy()
        h, w = out.shape[:2]
        font_scale = max(0.3, h / 640 * 0.7 * scale)
        thick = max(1, int(round(font_scale * 2)))
        for line_no, line in enumerate([text, cam]):
            if not line:
                continue
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
            x = 8 if corner in ("tl", "bl") else max(0, w - tw - 8)
            y = 8 + th + line_no * (th + 8) if corner in ("tl", "tr") else h - 8 - line_no * (th + 8)
            col = color if out.ndim == 3 else int(np.mean(color))
            cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0) if out.ndim == 3 else 0,
                        thick + 2, cv2.LINE_AA)
            cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thick, cv2.LINE_AA)
        return out

    def get_transform_init_args_names(self):
        return ("p",)


def build_cctv_augmentations(level: str = "medium") -> list | None:
    """Return the Albumentations list for a strength level, or None for 'off'.

    'off' returns None so Ultralytics falls back to its own (very mild) defaults.
    The level scales the probability of each degradation; the order mimics a real
    camera: scene lighting/weather -> sensor (gray/blur/noise) -> encoder (resize, JPEG).
    """
    dbg("build_cctv_augmentations", f"enter level={level}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if level == "off":
        dbg("build_cctv_augmentations", "exit -> None (Ultralytics defaults)")
        return None
    k = {"light": 0.5, "medium": 1.0, "heavy": 1.5}[level]

    def p(x):
        return round(min(0.95, x * k), 3)

    transforms = [
        # Scene: night / overexposed / low contrast / weather
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.2), contrast_limit=(-0.35, 0.15), p=p(0.35)),
        A.RandomGamma(gamma_limit=(70, 160), p=p(0.2)),
        A.OneOf([A.RandomFog(fog_coef_range=(0.1, 0.35), alpha_coef=0.08),
                 A.RandomRain(slant_range=(-10, 10), drop_width=1, blur_value=3, brightness_coefficient=0.8)],
                p=p(0.08)),
        # Sensor: infrared night mode is grayscale, plus focus / motion blur and noise
        A.ToGray(p=p(0.2)),
        A.CLAHE(clip_limit=(1, 3), p=p(0.05)),
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)),
                 A.GaussianBlur(sigma_limit=(0.3, 1.8)),
                 A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3))], p=p(0.3)),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.08)),
                 A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4))], p=p(0.3)),
        # Encoder: low resolution stream, heavy compression, burned-in timestamp
        A.Downscale(scale_range=(0.3, 0.7), p=p(0.3)),
        TimestampOverlay(p=p(0.3)),
        A.ImageCompression(quality_range=(20, 75), p=p(0.5)),
    ]
    dbg("build_cctv_augmentations", f"exit {len(transforms)} transforms, probs="
                                    f"{[(type(t).__name__, t.p) for t in transforms]}")
    return transforms


def preview(src: Path, out: Path, level: str, n: int, seed: int) -> list[Path]:
    """Save side-by-side original | augmented images so you can eyeball the degradations."""
    dbg("preview", f"enter src={src} out={out} level={level} n={n}")
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise SystemExit(f"No images in {src}")
    random.Random(seed).shuffle(imgs)
    aug = A.Compose(build_cctv_augmentations(level) or [], seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for path in imgs[:n]:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = aug(image=rgb)["image"]
        both = np.concatenate([rgb, res], axis=1)
        dst = out / f"{path.stem}_{level}.jpg"
        cv2.imwrite(str(dst), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
        written.append(dst)
    dbg("preview", f"exit wrote={len(written)}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, required=True, help="folder of images to augment")
    ap.add_argument("--out", type=Path, default=Path("runs/aug_preview"))
    ap.add_argument("--level", choices=LEVELS, default="medium")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    enable(args.debug)
    files = preview(args.preview, args.out, args.level, args.n, args.seed)
    print(f"Wrote {len(files)} previews (left = original, right = augmented) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
