"""Tests for scripts/dedupe_cctv.py (near-duplicate CCTV frame removal).

Run:  .venv/bin/python -m unittest tests.test_dedupe_cctv -v
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _helpers import make_image, step

import dedupe_cctv as dd

CLASSES = ["garbage", "person"]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "cctv"
        (self.data / "labels" / "train").mkdir(parents=True)
        (self.data / "classes.txt").write_text("garbage\nperson\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def frame(self, name, seed, lines=(), copy_of=None, js=None):
        img = self.data / "images" / "train" / f"{name}.jpg"
        if copy_of:
            img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(copy_of, img)
        else:
            make_image(img, seed)
        (self.data / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines))
        if js is not None:
            img.with_suffix(".json").write_text(json.dumps(js))
        return img


class TestBoxes(Base):
    def test_same_boxes(self):
        step("same class + IoU>=0.7 matches; different count / class / position does not")
        a = [(1, 0.1, 0.1, 0.3, 0.5)]
        self.assertTrue(dd.same_boxes(a, [(1, 0.11, 0.1, 0.31, 0.5)], 0.7))
        self.assertFalse(dd.same_boxes(a, [(0, 0.1, 0.1, 0.3, 0.5)], 0.7))
        self.assertFalse(dd.same_boxes(a, [(1, 0.5, 0.5, 0.7, 0.9)], 0.7))
        self.assertFalse(dd.same_boxes(a, a + a, 0.7))
        self.assertTrue(dd.same_boxes([], [], 0.7))

    def test_load_boxes_prefers_json(self):
        step("JSON rectangle (pixels) wins over the .txt draft and carries the checked flag")
        img = self.frame("c_f000000", 1, ["0 0.5 0.5 0.2 0.2"], js={
            "imageWidth": 100, "imageHeight": 50, "checked": True,
            "shapes": [{"label": "person", "points": [[10, 5], [30, 5], [30, 25], [10, 25]]},
                       {"label": "car", "points": [[0, 0], [1, 1]]}]})
        boxes, checked = dd.load_boxes(img, self.data / "labels/train/c_f000000.txt", CLASSES)
        step(f"boxes={boxes} checked={checked}")
        self.assertEqual(boxes, [(1, 0.1, 0.1, 0.3, 0.5)])
        self.assertTrue(checked)

    def test_load_boxes_txt_fallback(self):
        step("no JSON -> YOLO txt converted to corners; bad lines ignored")
        img = self.frame("c_f000000", 1, ["1 0.5 0.5 0.2 0.4", "garbage line"])
        boxes, checked = dd.load_boxes(img, self.data / "labels/train/c_f000000.txt", CLASSES)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][1], 0.4)
        self.assertFalse(checked)


class TestFindAndMove(Base):
    def test_duplicate_found_and_moved(self):
        step("identical image + same boxes in one clip -> later frame flagged and moved with its txt/json")
        first = self.frame("clipA_f000000", 1, ["1 0.5 0.5 0.2 0.4"])
        self.frame("clipA_f000030", 0, ["1 0.5 0.5 0.2 0.4"], copy_of=first, js={"shapes": []} | {
            "imageWidth": 96, "imageHeight": 96,
            "shapes": [{"label": "person", "points": [[38.4, 28.8], [57.6, 67.2]]}]})
        pairs = dd.find_duplicates(self.data, "train", CLASSES, 40, 0.7)
        self.assertEqual([(p["kept"].name, p["dup"].name) for p in pairs],
                         [("clipA_f000000.jpg", "clipA_f000030.jpg")])
        sheets = dd.write_preview(pairs, self.tmp / "prev", "train")
        self.assertTrue(sheets[0].exists())
        self.assertEqual(dd.move_duplicates(self.data, "train", pairs), 1)
        dup = self.data / "_duplicates"
        for p in ("images/train/clipA_f000030.jpg", "images/train/clipA_f000030.json",
                  "labels/train/clipA_f000030.txt"):
            step(f"moved: {p}")
            self.assertTrue((dup / p).exists())
            self.assertFalse((self.data / p).exists())
        self.assertTrue(first.exists())
        log = [json.loads(line) for line in (dup / "moved.jsonl").read_text().splitlines()]
        self.assertEqual(log[0]["moved"], "clipA_f000030.jpg")

    def test_person_moved_is_kept(self):
        step("same image but the person box moved -> NOT a duplicate")
        first = self.frame("clipA_f000000", 1, ["1 0.2 0.5 0.1 0.3"])
        self.frame("clipA_f000030", 0, ["1 0.8 0.5 0.1 0.3"], copy_of=first)
        self.assertEqual(dd.find_duplicates(self.data, "train", CLASSES, 40, 0.7), [])

    def test_other_clip_and_different_image_kept(self):
        step("same image in another clip, or a different image -> NOT a duplicate")
        first = self.frame("clipA_f000000", 1, [])
        self.frame("clipB_f000000", 0, [], copy_of=first)
        self.frame("clipA_f000030", 99, [])
        self.assertEqual(dd.find_duplicates(self.data, "train", CLASSES, 40, 0.7), [])

    def test_checked_frame_never_moved(self):
        step("frame marked checked in X-AnyLabeling is kept even if identical")
        first = self.frame("clipA_f000000", 1, [])
        self.frame("clipA_f000030", 0, [], copy_of=first,
                   js={"imageWidth": 96, "imageHeight": 96, "checked": True, "shapes": []})
        self.assertEqual(dd.find_duplicates(self.data, "train", CLASSES, 40, 0.7), [])

    def test_refuses_overwrite(self):
        step("a file already in _duplicates is never overwritten")
        first = self.frame("clipA_f000000", 1, [])
        self.frame("clipA_f000030", 0, [], copy_of=first)
        pairs = dd.find_duplicates(self.data, "train", CLASSES, 40, 0.7)
        clash = self.data / "_duplicates/images/train/clipA_f000030.jpg"
        clash.parent.mkdir(parents=True)
        clash.write_text("x")
        with self.assertRaises(SystemExit):
            dd.move_duplicates(self.data, "train", pairs)

    def test_missing_classes_file(self):
        step("missing classes.txt -> SystemExit")
        (self.data / "classes.txt").unlink()
        with self.assertRaises(SystemExit):
            dd.read_classes(self.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
