"""Tests for scripts/cctv_augment.py.

Run:  .venv/bin/python -m unittest tests.test_cctv_augment -v
"""
import tempfile
import unittest
from pathlib import Path

import albumentations as A
import numpy as np
from _helpers import make_image, step

import cctv_augment as ca


class BuildTests(unittest.TestCase):
    def test_off_returns_none(self):
        step("level off -> None (Ultralytics keeps its defaults)")
        self.assertIsNone(ca.build_cctv_augmentations("off"))

    def test_invalid_level(self):
        step("level 'extreme' -> ValueError")
        with self.assertRaises(ValueError):
            ca.build_cctv_augmentations("extreme")

    def test_levels_scale_probabilities(self):
        step("heavy has higher JPEG probability than light")
        light = {type(t).__name__: t.p for t in ca.build_cctv_augmentations("light")}
        heavy = {type(t).__name__: t.p for t in ca.build_cctv_augmentations("heavy")}
        step(f"ImageCompression light={light['ImageCompression']} heavy={heavy['ImageCompression']}")
        self.assertLess(light["ImageCompression"], heavy["ImageCompression"])
        self.assertTrue(all(0 < p <= 0.95 for p in heavy.values()))

    def test_only_pixel_transforms(self):
        step("no geometric transforms (boxes must stay valid)")
        for t in ca.build_cctv_augmentations("heavy"):
            self.assertNotIsInstance(t, A.DualTransform, type(t).__name__)


class ApplyTests(unittest.TestCase):
    def test_shape_and_dtype_preserved(self):
        step("50 random passes at heavy level keep HxWx3 uint8")
        comp = A.Compose(ca.build_cctv_augmentations("heavy"), seed=0)
        img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        for _ in range(50):
            out = comp(image=img)["image"]
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_serialization_roundtrip(self):
        step("to_dict/from_dict round trip (Ultralytics stores augmentations in checkpoints)")
        dicts = [A.to_dict(t) for t in ca.build_cctv_augmentations("medium")]
        restored = [A.from_dict(d) for d in dicts]
        step(f"restored -> {[type(t).__name__ for t in restored]}")
        self.assertIsInstance(restored[-2], ca.TimestampOverlay)

    def test_timestamp_overlay_draws_text(self):
        step("TimestampOverlay(p=1) changes pixels in a corner of a flat gray image")
        img = np.full((480, 640, 3), 128, np.uint8)
        out = A.Compose([ca.TimestampOverlay(p=1.0)], seed=3)(image=img)["image"]
        changed = int((out != img).any(axis=2).sum())
        step(f"changed pixels -> {changed}")
        self.assertGreater(changed, 100)

    def test_timestamp_on_tiny_image(self):
        step("very small image (32x32) must not crash")
        out = A.Compose([ca.TimestampOverlay(p=1.0)], seed=1)(image=np.zeros((32, 32, 3), np.uint8))["image"]
        self.assertEqual(out.shape, (32, 32, 3))


class PreviewTests(unittest.TestCase):
    def test_preview_writes_side_by_side(self):
        step("preview 2 images -> files twice as wide as the input")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            make_image(src / "a.jpg", 1, (64, 48))
            make_image(src / "b.jpg", 2, (64, 48))
            files = ca.preview(src, Path(d) / "out", "medium", 5, 0)
            import cv2
            h, w = cv2.imread(str(files[0])).shape[:2]
            step(f"written -> {len(files)}, size -> {w}x{h}")
            self.assertEqual((len(files), w, h), (2, 128, 48))

    def test_preview_empty_folder(self):
        step("empty folder -> SystemExit")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                ca.preview(Path(d), Path(d) / "out", "medium", 5, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
