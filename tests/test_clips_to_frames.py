"""Tests for scripts/clips_to_frames.py (no models needed: model calls are faked).

Run:  .venv/bin/python -m unittest tests.test_clips_to_frames -v
  or: .venv/bin/python -m unittest discover -s tests -t tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from _helpers import step, make_image

import clips_to_frames as cf


def make_video(path: Path, frames: list, fps: int = 10) -> Path:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def noise(seed, size=(64, 96)):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, (*size, 3))).astype(np.uint8)


class FakeBoxes:
    def __init__(self, xywhn):
        self._x = xywhn

    @property
    def xywhn(self):
        return SimpleNamespace(tolist=lambda: self._x)

    def __len__(self):
        return len(self._x)


class FakeModel:
    def __init__(self, boxes):
        self.boxes, self.calls = boxes, []

    def predict(self, path, **kw):
        self.calls.append((path, kw))
        return [SimpleNamespace(boxes=FakeBoxes(self.boxes))]


class TestParseAssign(unittest.TestCase):
    def test_normal(self):
        step("parse two valid assignments")
        out = cf.parse_assign(["a=train", "b=val"])
        print("   result:", out)
        self.assertEqual(out, {"a": "train", "b": "val"})

    def test_empty(self):
        step("empty/None list gives {}")
        self.assertEqual(cf.parse_assign([]), {})
        self.assertEqual(cf.parse_assign(None), {})

    def test_bad(self):
        for bad in ["a", "a=foo", "=train"]:
            step(f"invalid {bad!r} raises ValueError")
            with self.assertRaises(ValueError):
                cf.parse_assign([bad])


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_extract_samples_by_fps(self):
        step("20 frames at 10 fps, sample 1 fps -> 2 frames")
        v = make_video(self.tmp / "clip.mp4", [noise(i) for i in range(20)], fps=10)
        out = cf.extract_frames(v, 1.0, self.tmp / "t")
        print("   frames:", [(f["frame"], f["time_s"]) for f in out])
        self.assertEqual([f["frame"] for f in out], [0, 10])
        self.assertTrue(all(f["path"].exists() for f in out))

    def test_extract_bad_fps_and_missing_video(self):
        step("fps <= 0 raises ValueError; missing video raises SystemExit")
        with self.assertRaises(ValueError):
            cf.extract_frames(self.tmp / "x.mp4", 0, self.tmp / "t")
        with self.assertRaises(SystemExit):
            cf.extract_frames(self.tmp / "missing.mp4", 1, self.tmp / "t")

    def test_dedupe(self):
        step("identical images collapse to one, distinct one is kept")
        a = make_image(self.tmp / "a.jpg", 1)
        b = self.tmp / "b.jpg"
        b.write_bytes(a.read_bytes())
        c = make_image(self.tmp / "c.jpg", 99)
        frames = [{"path": a}, {"path": b}, {"path": c}]
        kept, dropped = cf.dedupe_frames(frames, max_distance=6)
        print("   kept:", [f["path"].name for f in kept], "dropped:", dropped)
        self.assertEqual([f["path"].name for f in kept], ["a.jpg", "c.jpg"])
        self.assertEqual(dropped, 1)
        self.assertFalse(b.exists(), "duplicate file should be deleted")

    def test_dedupe_unreadable_dropped(self):
        step("unreadable frame is dropped, not crashed")
        bad = self.tmp / "bad.jpg"
        bad.write_bytes(b"not an image")
        kept, dropped = cf.dedupe_frames([{"path": bad}], 6)
        self.assertEqual((kept, dropped), ([], 1))

    def test_dedupe_empty(self):
        step("empty frame list")
        self.assertEqual(cf.dedupe_frames([], 6), ([], 0))


class TestLabelsAndWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_draft_labels_garbage_first_then_person(self):
        step("fake models: 1 garbage box (cls 0) + 2 person boxes (cls 1)")
        p = make_image(self.tmp / "f.jpg", 3)
        g = FakeModel([[0.5, 0.5, 0.2, 0.2]])
        h = FakeModel([[0.1, 0.1, 0.1, 0.2], [0.9, 0.9, 0.1, 0.1]])
        out = cf.draft_labels([p], h, g, 0.4, 0.15, 640, "cpu")
        print("   lines:", out[p])
        self.assertEqual([l.split()[0] for l in out[p]], ["0", "1", "1"])
        self.assertEqual(h.calls[0][1]["classes"], [0], "person model must be limited to COCO person")

    def test_draft_labels_no_boxes(self):
        step("no detections -> empty list (background frame)")
        p = make_image(self.tmp / "f.jpg", 4)
        out = cf.draft_labels([p], FakeModel([]), FakeModel([]), 0.4, 0.15, 640, None)
        self.assertEqual(out[p], [])

    def test_write_clip_and_review(self):
        step("write frames into split folders + review.json keeps reviewed=true")
        p = make_image(self.tmp / "tmp" / "clip_f000000.jpg", 5)
        n = cf.write_clip([{"path": p}], {p: ["0 0.5 0.5 0.1 0.1"]}, self.tmp / "out", "val")
        lab = self.tmp / "out" / "labels" / "val" / "clip_f000000.txt"
        print("   wrote:", n, lab.read_text().strip())
        self.assertEqual(n, 1)
        self.assertTrue((self.tmp / "out" / "images" / "val" / "clip_f000000.jpg").exists())
        self.assertEqual(lab.read_text(), "0 0.5 0.5 0.1 0.1\n")

        out = self.tmp / "out"
        cf.update_review(out, "clip", {"split": "val", "reviewed": False})
        data = json.loads((out / "review.json").read_text())
        data["clip"]["reviewed"] = True
        (out / "review.json").write_text(json.dumps(data))
        cf.update_review(out, "clip", {"split": "val", "reviewed": False})
        final = json.loads((out / "review.json").read_text())
        print("   review.json:", final)
        self.assertTrue(final["clip"]["reviewed"], "re-running must not reset a human review")


class TestGdinoHelpers(unittest.TestCase):
    def test_prompt_format(self):
        step("phrases are lowercased and each ends with exactly one '.'")
        out = cf.gdino_prompt(["Garbage Pile", "trash bag.", "  ", "litter"])
        print("   prompt:", repr(out))
        self.assertEqual(out, "garbage pile. trash bag. litter.")

    def test_prompt_empty(self):
        step("empty prompt list gives empty string")
        self.assertEqual(cf.gdino_prompt([]), "")

    def test_boxes_adapter_works_with_to_yolo_lines(self):
        step("_Boxes adapter feeds to_yolo_lines like an Ultralytics result")
        r = SimpleNamespace(boxes=cf._Boxes([[0.5, 0.5, 0.2, 0.4]]))
        lines = cf.to_yolo_lines(r, 0)
        print("   lines:", lines)
        self.assertEqual(lines, ["0 0.500000 0.500000 0.200000 0.400000"])
        self.assertEqual(len(cf._Boxes([])), 0)

    def test_bad_backend_rejected(self):
        step("unknown --garbage-backend is rejected by argparse")
        with self.assertRaises(SystemExit):
            cf.main(["x.mp4", "--garbage-backend", "bogus"])


class TestGarbageFilter(unittest.TestCase):
    BIG = "0 0.5 0.5 0.95 0.95"        # area 0.9025
    PILE = "0 0.4 0.5 0.6 0.6"         # area 0.36
    BAG_IN_PILE = "0 0.4 0.5 0.1 0.1"  # fully inside PILE
    SEPARATE = "0 0.9 0.1 0.1 0.1"     # outside PILE
    PERSON = "1 0.5 0.5 0.9 0.9"

    def test_drop_big_and_nested_keep_person(self):
        step("big box (>0.8 area) and nested bag are dropped; pile, separate box, person kept")
        out, st = cf.filter_garbage_boxes([self.BIG, self.PILE, self.BAG_IN_PILE, self.SEPARATE, self.PERSON],
                                          0.8, 0.8)
        print("   kept:", out, "stats:", st)
        self.assertEqual(out, [self.PILE, self.SEPARATE, self.PERSON])
        self.assertEqual(st, {"too_big": 1, "nested": 1})

    def test_disabled(self):
        step("0 disables both filters")
        lines = [self.BIG, self.PILE, self.BAG_IN_PILE]
        out, st = cf.filter_garbage_boxes(lines, 0, 0)
        self.assertEqual(out, lines)
        self.assertEqual(st, {"too_big": 0, "nested": 0})

    def test_identical_boxes_both_kept(self):
        step("two identical garbage boxes: neither is LARGER, so nested rule keeps both")
        out, _ = cf.filter_garbage_boxes([self.PILE, self.PILE], 0.8, 0.8)
        self.assertEqual(len(out), 2)

    def test_area_boundary_kept(self):
        step("box exactly at max_area is kept")
        line = "0 0.5 0.5 0.8 1.0"  # area 0.8
        out, _ = cf.filter_garbage_boxes([line], 0.8, 0)
        self.assertEqual(out, [line])

    def test_empty(self):
        self.assertEqual(cf.filter_garbage_boxes([], 0.8, 0.8), ([], {"too_big": 0, "nested": 0}))


class TestAutoSplitAndBatch(unittest.TestCase):
    def test_stable_and_distribution(self):
        step("same clip always gets same split; ~15% of 400 clips go to val")
        self.assertEqual(cf.auto_split("abc", 15, 0), cf.auto_split("abc", 15, 0))
        splits = [cf.auto_split(f"clip{i}", 15, 5) for i in range(400)]
        counts = {s: splits.count(s) for s in ("train", "val", "test")}
        print("   counts:", counts)
        self.assertTrue(30 <= counts["val"] <= 90, counts)
        self.assertTrue(5 <= counts["test"] <= 40, counts)

    def test_bad_pct(self):
        step("negative or >=100 total raises ValueError")
        for v, t in [(-1, 0), (60, 40), (100, 0)]:
            with self.assertRaises(ValueError):
                cf.auto_split("x", v, t)

    def test_expand_videos(self):
        step("folders are searched recursively; non-video files ignored")
        tmp = Path(tempfile.mkdtemp())
        (tmp / "sub").mkdir()
        for n in ["a.mp4", "sub/b.MOV", "notes.txt"]:
            (tmp / n).write_bytes(b"x")
        out = cf.expand_videos([tmp, tmp / "notes.txt"])
        print("   videos:", [p.name for p in out])
        self.assertEqual([p.name for p in out], ["a.mp4", "b.MOV"])

    def test_mark_reviewed(self):
        step("mark one clip, then 'all'; unknown clip and missing file raise SystemExit")
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(SystemExit):
            cf.mark_reviewed(tmp, ["a"])
        (tmp / "review.json").write_text(json.dumps({"a": {"reviewed": False}, "b": {"reviewed": False}}))
        self.assertEqual(cf.mark_reviewed(tmp, ["a"]), ["a"])
        data = json.loads((tmp / "review.json").read_text())
        self.assertEqual((data["a"]["reviewed"], data["b"]["reviewed"]), (True, False))
        cf.mark_reviewed(tmp, ["all"])
        self.assertTrue(json.loads((tmp / "review.json").read_text())["b"]["reviewed"])
        with self.assertRaises(SystemExit):
            cf.mark_reviewed(tmp, ["nope"])


class TestMainFailures(unittest.TestCase):
    def test_bad_assign_and_missing_video(self):
        step("bad --assign -> SystemExit; missing video -> SystemExit (before loading models)")
        with self.assertRaises(SystemExit):
            cf.main(["x.mp4", "--assign", "x=bogus"])
        with self.assertRaises(SystemExit):
            cf.main(["/nonexistent/clip.mp4"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
