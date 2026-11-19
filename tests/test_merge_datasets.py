"""Tests for scripts/merge_datasets.py using tiny fake datasets (no model download needed).

Run:  .venv/bin/python -m unittest tests.test_merge_datasets -v
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from _helpers import make_image, step

import merge_datasets as md


def make_old(root: Path):
    """Ultralytics layout, one class 'garbage'. old_dup_val.jpg duplicates rf train image seed 100."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["garbage"]}))
    for split, seeds in {"train": [1, 2], "val": [3], "test": [4]}.items():
        for s in seeds:
            make_image(root / "images" / split / f"old{s}_jpg.rf.aaa{s}.jpg", s)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / f"old{s}_jpg.rf.aaa{s}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    make_image(root / "images/val/old_dup_val.jpg", 100)
    (root / "labels/val/old_dup_val.txt").write_text("0 0.5 0.5 0.3 0.3\n")


def make_rf(root: Path):
    """Roboflow layout, names [human, trash]; photo 'p9' has augmented copies in train AND test."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["human", "trash"]}))
    files = {
        "train": [("p1_jpg.rf.x1.jpg", 10, "0 0.3 0.3 0.1 0.2\n1 0.6 0.6 0.2 0.2\n"),
                  ("p9_jpg.rf.x2.jpg", 11, "1 0.5 0.5 0.2 0.2\n"),
                  ("dup_jpg.rf.x3.jpg", 100, "1 0.5 0.5 0.3 0.3\n")],
        "valid": [("p2_jpg.rf.y1.jpg", 12, "1 0.4 0.4 0.1 0.1\n")],
        "test": [("p9_jpg.rf.z1.jpg", 13, "1 0.5 0.5 0.2 0.2\n"),
                 ("bg_jpg.rf.z2.jpg", 14, "")],
    }
    for split, items in files.items():
        for name, seed, lab in items:
            make_image(root / split / "images" / name, seed)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / name.replace(".jpg", ".txt")).write_text(lab)


class ClassMapTests(unittest.TestCase):
    def test_keyword_mapping(self):
        step("names [Human, trash, Waste-Pile, pedestrian] -> person, garbage, garbage, person")
        m = md.build_class_map(["Human", "trash", "Waste-Pile", "pedestrian"], {})
        step(f"mapping -> {m}")
        self.assertEqual(m, {0: 1, 1: 0, 2: 0, 3: 1})

    def test_unknown_name_stops(self):
        step("'bicycle' is unknown -> SystemExit that suggests --class-map")
        with self.assertRaises(SystemExit) as cm:
            md.build_class_map(["bicycle"], {})
        step(f"error -> {cm.exception}")
        self.assertIn("--class-map", str(cm.exception))

    def test_override_and_drop(self):
        step("override bicycle=drop, bag=garbage")
        m = md.build_class_map(["bicycle", "bag"], md.parse_overrides(["bicycle=drop", "bag=garbage"]))
        self.assertEqual(m, {0: None, 1: 0})

    def test_bad_override(self):
        step("override target 'car' is invalid")
        with self.assertRaises(SystemExit):
            md.parse_overrides(["x=car"])


class LabelParsingTests(unittest.TestCase):
    def test_parse_box_polygon_invalid_and_clip(self):
        step("file with: normal box, polygon, text garbage, zero-size box, box past the edge")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\nfoo bar\n"
                         "0 0.5 0.5 0 0.1\n0 0.95 0.5 0.2 0.2\n")
            boxes, bad = md.parse_label_file(p)
            step(f"boxes -> {boxes}  invalid -> {bad}")
            self.assertEqual(len(boxes), 3)
            self.assertEqual(bad, 2)
            cls, xc, yc, w, h = boxes[1]
            self.assertAlmostEqual(xc, 0.2)
            self.assertAlmostEqual(h, 0.3)
            self.assertLessEqual(boxes[2][1] + boxes[2][3] / 2, 1.0 + 1e-9)

    def test_missing_label_file_is_empty(self):
        step("missing label file -> no boxes")
        self.assertEqual(md.parse_label_file(Path("/nope.txt")), ([], 0))


class LeakageTests(unittest.TestCase):
    def test_group_leakage_keeps_test_copy(self):
        step("same photo group in train and test -> train copy dropped")
        recs = [{"group": "rf:p9", "split": "train"}, {"group": "rf:p9", "split": "test"},
                {"group": "rf:p1", "split": "train"}]
        n = md.fix_group_leakage(recs)
        step(f"dropped -> {n}, records -> {recs}")
        self.assertEqual(n, 1)
        self.assertEqual(recs[0]["drop"], "augmentation_group_in_other_split")
        self.assertNotIn("drop", recs[1])

    def test_dhash_identical_and_different(self):
        step("same image twice -> distance 0, different image -> large distance")
        with tempfile.TemporaryDirectory() as d:
            a = make_image(Path(d) / "a.jpg", 5)
            b = make_image(Path(d) / "b.jpg", 5)
            c = make_image(Path(d) / "c.jpg", 6)
            ha, hb, hc = md.dhash(a), md.dhash(b), md.dhash(c)
            step(f"d(a,b)={bin(ha ^ hb).count('1')} d(a,c)={bin(ha ^ hc).count('1')}")
            self.assertEqual(ha, hb)
            self.assertGreater(bin(ha ^ hc).count("1"), 4)

    def test_unreadable_image_is_dropped(self):
        step("corrupt file -> dhash None -> dropped as unreadable_image")
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jpg"
            bad.write_bytes(b"not an image")
            recs = [{"image": bad, "split": "train", "source": "old"}]
            reasons = md.fix_near_duplicates(recs, 4, 2)
            self.assertEqual(reasons["unreadable_image"], 1)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        make_old(self.tmp / "old")
        make_rf(self.tmp / "rf")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, *extra):
        args = ["--old", str(self.tmp / "old"), "--roboflow", str(self.tmp / "rf"),
                "--output", str(self.tmp / "out"), "--cache", str(self.tmp / "cache.json"), *extra]
        step(f"merge_datasets.main({' '.join(extra) or 'defaults'})")
        return md.main(args)

    def test_merge_without_autolabel(self):
        self._run("--no-autolabel")
        out = self.tmp / "out"
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        step(f"data.yaml -> nc={cfg['nc']} names={cfg['names']}")
        self.assertEqual(cfg["names"], ["garbage", "person"])

        step("rf 'human'(0) -> person(1), rf 'trash'(1) -> garbage(0)")
        lab = (out / "labels/train/rf_p1_jpg.rf.x1.txt").read_text().split("\n")
        self.assertTrue(lab[0].startswith("1 ") and lab[1].startswith("0 "))

        report = json.loads((out / "merge_report.json").read_text())
        step(f"report dropped -> {report['dropped']}")
        self.assertFalse((out / "images/train/rf_p9_jpg.rf.x2.jpg").exists(), "leaked train copy must go")
        self.assertTrue((out / "images/test/rf_p9_jpg.rf.z1.jpg").exists())
        self.assertFalse((out / "images/train/rf_dup_jpg.rf.x3.jpg").exists(), "train dup of a val image must go")
        self.assertTrue((out / "images/val/old_old_dup_val.jpg").exists())

        step("background test image keeps an empty label file")
        self.assertEqual((out / "labels/test/rf_bg_jpg.rf.z2.txt").read_text(), "")
        self.assertEqual(report["splits"]["test"]["background_images"], 1)

    def test_autolabel_uses_cache_and_adds_person_boxes(self):
        step("pre-fill the auto-label cache so no model is downloaded")
        old_train = self.tmp / "old/images/train/old1_jpg.rf.aaa1.jpg"
        cache = {"_meta": "fake.pt|0.4|640", str(old_train.resolve()): [[0.2, 0.3, 0.1, 0.4]]}
        for p in (self.tmp / "old/images").rglob("*.jpg"):
            cache.setdefault(str(p.resolve()), [])
        (self.tmp / "cache.json").write_text(json.dumps(cache))
        self._run("--person-model", "fake.pt")
        lab = (self.tmp / "out/labels/train/old_old1_jpg.rf.aaa1.txt").read_text().splitlines()
        step(f"label lines -> {lab}")
        self.assertEqual(lab[0].split()[0], "0")
        self.assertEqual(lab[1].split()[0], "1")
        report = json.loads((self.tmp / "out/merge_report.json").read_text())
        self.assertEqual(report["person_boxes_autolabeled"], 1)

    def test_existing_output_requires_overwrite(self):
        step("second run without --overwrite must refuse")
        self._run("--no-autolabel")
        with self.assertRaises(SystemExit):
            self._run("--no-autolabel")
        step("with --overwrite it rebuilds")
        self.assertEqual(self._run("--no-autolabel", "--overwrite"), 0)

    def test_backgrounds_folder_added_to_train(self):
        step("add 2 empty-scene images via --backgrounds")
        make_image(self.tmp / "bg/cam1.jpg", 50)
        make_image(self.tmp / "bg/cam2.jpg", 51)
        self._run("--no-autolabel", "--backgrounds", str(self.tmp / "bg"))
        self.assertEqual((self.tmp / "out/labels/train/bg_cam1.txt").read_text(), "")

    def test_missing_roboflow_dir(self):
        step("missing roboflow dir -> SystemExit")
        shutil.rmtree(self.tmp / "rf")
        with self.assertRaises(SystemExit):
            self._run("--no-autolabel")


if __name__ == "__main__":
    unittest.main(verbosity=2)
