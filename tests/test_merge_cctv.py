"""Tests for the CCTV source in scripts/merge_datasets.py (collect_cctv).

Run:  .venv/bin/python -m unittest tests.test_merge_cctv -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from _helpers import step, make_image

import merge_datasets as md


def make_cctv(root: Path, review: dict, frames: dict) -> Path:
    """frames: {split: [(stem, label_text)]}"""
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("names:\n- garbage\n- person\n")
    seed = 0
    for split, items in frames.items():
        for stem, text in items:
            seed += 1
            make_image(root / "images" / split / f"{stem}.jpg", seed)
            lab = root / "labels" / split / f"{stem}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text(text)
    if review is not None:
        (root / "review.json").write_text(json.dumps(review))
    return root


class TestCollectCctv(unittest.TestCase):
    def test_only_reviewed_clips_and_group_per_clip(self):
        step("clip A reviewed (2 frames), clip B not reviewed, one stray frame")
        root = make_cctv(Path(tempfile.mkdtemp()),
                         {"clipA": {"reviewed": True}, "clipB": {"reviewed": False}},
                         {"train": [("clipA_f000000", "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.2\n"),
                                    ("clipA_f000030", ""),
                                    ("clipB_f000000", "0 0.5 0.5 0.2 0.2\n"),
                                    ("random", "")]})
        recs, info = md.collect_cctv(root, {})
        print("   info:", info)
        self.assertEqual(sorted(r["image"].stem for r in recs), ["clipA_f000000", "clipA_f000030"])
        self.assertEqual({r["group"] for r in recs}, {"cctv:clipA"})
        self.assertEqual({r["source"] for r in recs}, {"cctv"})
        self.assertEqual(info["frames_skipped"], {"not_reviewed": 1, "unknown_clip": 1})
        first = [r for r in recs if r["image"].stem == "clipA_f000000"][0]
        print("   boxes:", first["boxes"])
        self.assertEqual([b[0] for b in first["boxes"]], [0, 1], "garbage=0, person=1 kept")

    def test_clip_name_with_underscores(self):
        step("clip names containing '_f' or underscores still map correctly")
        root = make_cctv(Path(tempfile.mkdtemp()), {"my_f_clip_2": {"reviewed": True}},
                         {"val": [("my_f_clip_2_f000120", "")]})
        recs, _ = md.collect_cctv(root, {})
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["split"], "val")

    def test_none_reviewed(self):
        step("nothing reviewed -> zero records, no crash")
        root = make_cctv(Path(tempfile.mkdtemp()), {"c": {"reviewed": False}},
                         {"train": [("c_f000000", "")]})
        recs, info = md.collect_cctv(root, {})
        self.assertEqual((recs, info["frames_used"]), ([], 0))

    def test_missing_review_json(self):
        step("missing review.json -> SystemExit")
        root = make_cctv(Path(tempfile.mkdtemp()), None, {"train": [("c_f000000", "")]})
        with self.assertRaises(SystemExit):
            md.collect_cctv(root, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
