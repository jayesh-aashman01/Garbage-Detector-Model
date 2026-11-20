"""Checks data/cctv_train_clips.txt (training clips for ./run.sh cctv-download).

Run:  .venv/bin/python -m unittest tests.test_cctv_clip_list -v
"""
import re
import unittest

from _helpers import ROOT, step

TRAIN_LIST = ROOT / "data" / "cctv_train_clips.txt"
TEST_LIST = ROOT / "tests" / "cctv_test_clips.txt"
# clips used to evaluate the model on 2026-11-20 (cctv_test/clips/); must stay out of training
EVAL_IDS = {"vqnAGFps9Rc", "DfOLXE7xvWM", "IVYKuVCXTAw", "w649RJET4YA", "mQAihCEoQ74", "yFEORmcEScU",
            "Li3aJbDmJoU", "G_UrcGce6EU", "fFvs2VTBLIY", "PEyhUwGBrnQ", "eTKYIlTR7Ss", "MU5JriIfo5k",
            "-9G1iDGuAXU", "yygeklsPy_k"}
YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def read_ids(path):
    return [line.split()[0] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


class TestTrainClipList(unittest.TestCase):
    def setUp(self):
        self.ids = read_ids(TRAIN_LIST)

    def test_ids_valid_and_unique(self):
        step(f"{len(self.ids)} ids: 11-char YouTube format, no duplicates")
        bad = [i for i in self.ids if not YT_ID.match(i)]
        self.assertEqual(bad, [])
        self.assertEqual(len(self.ids), len(set(self.ids)))

    def test_enough_clips_for_1500_frames(self):
        step("at least 50 clips (about 25 usable frames each gives ~1500)")
        self.assertGreaterEqual(len(self.ids), 50)

    def test_no_overlap_with_test_clips(self):
        test_ids = set(read_ids(TEST_LIST)) | EVAL_IDS
        overlap = sorted(set(self.ids) & test_ids)
        step(f"overlap with test/eval clips: {overlap}")
        self.assertEqual(overlap, [], "training clips must not be used for evaluation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
