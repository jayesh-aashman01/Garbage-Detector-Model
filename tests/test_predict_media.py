"""Tests for scripts/predict_media.py.

Unit tests need nothing extra. The end-to-end test runs a real model on a generated image
and video; it uses $GD_TEST_MODEL or the trained best.pt and is skipped if neither exists.

Run:  .venv/bin/python -m unittest tests.test_predict_media -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from _helpers import ROOT, make_image, step

import predict_media as pm


class SourceTests(unittest.TestCase):
    def test_expand_folder_and_streams(self):
        step("folder with 1 jpg, 1 mp4, 1 txt + an rtsp URL + webcam '0'")
        with tempfile.TemporaryDirectory() as d:
            make_image(Path(d) / "a.jpg", 1)
            (Path(d) / "v.mp4").write_bytes(b"x")
            (Path(d) / "notes.txt").write_text("ignore me")
            out = pm.expand_sources([d, "rtsp://cam/1", "0"])
            step(f"sources -> {out}")
            kinds = sorted(k for k, _ in out)
            self.assertEqual(kinds, ["image", "stream", "stream", "video"])

    def test_missing_file(self):
        step("non-existent file -> SystemExit")
        with self.assertRaises(SystemExit):
            pm.expand_sources(["/nope/clip.mp4"])

    def test_folder_without_media(self):
        step("folder with only a txt -> SystemExit")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.txt").write_text("")
            with self.assertRaises(SystemExit):
                pm.expand_sources([d])


class ClassIdTests(unittest.TestCase):
    def test_new_two_class_model(self):
        step("names {0: garbage, 1: person}")
        self.assertEqual(pm.class_ids({0: "garbage", 1: "person"}), (0, 1))

    def test_old_one_class_model(self):
        step("old model names {0: garbage} -> person id None")
        self.assertEqual(pm.class_ids({0: "garbage"}), (0, None))

    def test_trash_name(self):
        step("names {0: human, 1: trash}")
        self.assertEqual(pm.class_ids({0: "human", 1: "trash"}), (1, 0))


class PersistenceTests(unittest.TestCase):
    def test_single_frame_spike_is_ignored(self):
        step("persist=3/window=5: one detection then nothing -> never ON")
        a = pm.PersistenceAlert(3, 5)
        changes = [a.update(x) for x in [1, 0, 0, 0, 0]]
        step(f"changes -> {changes}")
        self.assertEqual(changes, [None] * 5)

    def test_alert_starts_and_ends(self):
        step("3 hits in a row -> start; then 3 misses -> end")
        a = pm.PersistenceAlert(3, 5)
        changes = [a.update(x) for x in [1, 1, 1, 1, 0, 0, 0]]
        step(f"changes -> {changes}")
        self.assertEqual(changes.index("start"), 2)
        self.assertIn("end", changes)

    def test_invalid_parameters(self):
        step("persist > window -> ValueError")
        with self.assertRaises(ValueError):
            pm.PersistenceAlert(6, 5)
        with self.assertRaises(ValueError):
            pm.PersistenceAlert(0, 5)


class AccuracyTests(unittest.TestCase):
    def test_image_accuracy_confusion(self):
        step("4 images: TP, FP, FN, TN + one without a label file")
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d)
            (lab / "tp.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (lab / "fp.txt").write_text("1 0.5 0.5 0.1 0.1\n")  # only a person -> no garbage
            (lab / "fn.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (lab / "tn.txt").write_text("")
            per = {f"/x/{n}.jpg": {"kind": "image", "garbage_present": p}
                   for n, p in [("tp", True), ("fp", True), ("fn", False), ("tn", False), ("nolabel", True)]}
            acc = pm.image_accuracy(per, lab)
            step(f"accuracy -> {acc}")
            self.assertEqual((acc["tp"], acc["fp"], acc["fn"], acc["tn"]), (1, 1, 1, 1))
            self.assertEqual(acc["images_without_label_file"], 1)
            self.assertEqual(acc["precision"], 0.5)


def _find_model():
    env = os.environ.get("GD_TEST_MODEL")
    for cand in [env, ROOT / "runs/detect/runs/detect/garbage-person-yolo11m/weights/best.pt",
                 ROOT / "runs/detect/runs/detect/garbage-yolo11n/weights/best.pt"]:
        if cand and Path(cand).is_file():
            return str(cand)
    return None


@unittest.skipUnless(_find_model(), "no trained model available (set GD_TEST_MODEL=path/to/best.pt)")
class EndToEndTests(unittest.TestCase):
    def test_image_and_video(self):
        model = _find_model()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            step(f"model -> {model}")
            make_image(d / "img.jpg", 3, (320, 240))
            (d / "labels").mkdir()
            (d / "labels/img.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            step("write a 3-second 10 fps synthetic video")
            vw = cv2.VideoWriter(str(d / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            for i in range(30):
                vw.write(np.full((240, 320, 3), (i * 8) % 255, np.uint8))
            vw.release()
            rc = pm.main([str(d / "img.jpg"), str(d / "clip.mp4"), "--model", model, "--device", "cpu",
                          "--out", str(d / "out"), "--name", "t", "--labels", str(d / "labels"),
                          "--sample-fps", "5"])
            summary = json.loads((d / "out/t/summary.json").read_text())
            step(f"images={summary['images']} videos={summary['videos']} "
                 f"accuracy={summary['image_accuracy']}")
            self.assertEqual(rc, 0)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["videos"], 1)
            self.assertEqual(summary["per_source"][str(d / "clip.mp4")]["frames_sampled"], 15)
            self.assertTrue((d / "out/t/clip_pred.mp4").is_file())
            self.assertTrue((d / "out/t/img.jpg").is_file())
            lines = (d / "out/t/detections.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 16)

    def test_missing_model(self):
        step("missing model path -> SystemExit")
        with self.assertRaises(SystemExit):
            pm.main(["x.jpg", "--model", "/nope/best.pt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
