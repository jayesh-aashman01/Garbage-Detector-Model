"""Tests for merge_datasets.filter_roboflow (opt-in Roboflow down-sampling).

Run:  .venv/bin/python -m unittest tests.test_rf_filter -v
  or: .venv/bin/python -m unittest discover -s tests -t tests -v
"""
import unittest

from _helpers import step

import merge_datasets as md


def rec(source="rf", split="train", boxes=(), name="x.jpg", drop=None):
    r = {"source": source, "split": split, "image": name, "boxes": list(boxes)}
    if drop:
        r["drop"] = drop
    return r


class TestFilterRoboflow(unittest.TestCase):
    def test_defaults_change_nothing(self):
        step("all options None -> no record touched")
        recs = [rec(boxes=[(0, .5, .5, .9, .9)]), rec(boxes=[(1, .5, .5, .9, .9)])]
        out = md.filter_roboflow(recs)
        step(f"result={dict(out)} drops={[r.get('drop') for r in recs]}")
        self.assertEqual(out, {})
        self.assertTrue(all("drop" not in r for r in recs))

    def test_closeup_drops_single_big_box(self):
        step("one big box (0.81) dropped, two boxes kept, small box kept, old source ignored")
        big = rec(boxes=[(0, .5, .5, .9, .9)], name="big")
        two = rec(boxes=[(0, .5, .5, .9, .9), (0, .1, .1, .1, .1)], name="two")
        small = rec(boxes=[(0, .5, .5, .3, .3)], name="small")
        old = rec(source="old", boxes=[(0, .5, .5, .9, .9)], name="old")
        out = md.filter_roboflow([big, two, small, old], closeup=0.5)
        step(f"result={dict(out)}")
        self.assertEqual(big.get("drop"), "rf_closeup")
        self.assertNotIn("drop", two)
        self.assertNotIn("drop", small)
        self.assertNotIn("drop", old)

    def test_closeup_boundary_equal_is_kept(self):
        step("area exactly == threshold (0.5*1.0=0.5) is kept")
        r = rec(boxes=[(0, .5, .5, 0.5, 1.0)])
        md.filter_roboflow([r], closeup=0.5)
        step(f"drop={r.get('drop')}")
        self.assertNotIn("drop", r)

    def test_portrait_drops_big_lone_person(self):
        step("lone big person dropped; person+garbage kept; two people kept")
        lone = rec(boxes=[(1, .5, .5, .8, .8)], name="lone")
        mixed = rec(boxes=[(1, .5, .5, .8, .8), (0, .1, .1, .1, .1)], name="mixed")
        pair = rec(boxes=[(1, .3, .5, .8, .8), (1, .7, .5, .8, .8)], name="pair")
        out = md.filter_roboflow([lone, mixed, pair], portrait=0.4)
        step(f"result={dict(out)}")
        self.assertEqual(lone.get("drop"), "rf_portrait")
        self.assertNotIn("drop", mixed)
        self.assertNotIn("drop", pair)

    def test_cap_only_train_and_seeded(self):
        step("cap 3 of 10 train; val untouched; same seed -> same pick")
        def build():
            return [rec(name=f"t{i}", boxes=[(0, .5, .5, .1, .1)]) for i in range(10)] + \
                   [rec(split="val", name=f"v{i}") for i in range(5)]
        a, b = build(), build()
        out = md.filter_roboflow(a, cap_train=3, seed=7)
        md.filter_roboflow(b, cap_train=3, seed=7)
        kept = [r["image"] for r in a if r["split"] == "train" and not r.get("drop")]
        step(f"result={dict(out)} kept={kept}")
        self.assertEqual(out["rf_cap"], 7)
        self.assertEqual(len(kept), 3)
        self.assertTrue(all("drop" not in r for r in a if r["split"] == "val"))
        self.assertEqual(kept, [r["image"] for r in b if r["split"] == "train" and not r.get("drop")])

    def test_cap_larger_than_count_and_already_dropped(self):
        step("cap 100 with 2 live train records -> nothing capped; pre-dropped record untouched")
        recs = [rec(name="a"), rec(name="b"), rec(name="c", drop="near_duplicate_across_splits")]
        out = md.filter_roboflow(recs, cap_train=100)
        step(f"result={dict(out)}")
        self.assertEqual(out, {})
        self.assertEqual(recs[2]["drop"], "near_duplicate_across_splits")

    def test_no_rf_records(self):
        step("only old records -> nothing happens")
        recs = [rec(source="old", boxes=[(0, .5, .5, .9, .9)])]
        self.assertEqual(md.filter_roboflow(recs, closeup=0.1, portrait=0.1, cap_train=0), {})

    def test_invalid_args_raise(self):
        for kw in ({"closeup": 0}, {"closeup": -0.2}, {"portrait": 1.5}, {"cap_train": -1}):
            step(f"invalid {kw} -> ValueError")
            with self.assertRaises(ValueError):
                md.filter_roboflow([rec()], **kw)

    def test_cli_invalid_arg_exits(self):
        step("main() with --rf-drop-closeup 2 and missing sources -> SystemExit, nothing written")
        with self.assertRaises(SystemExit):
            md.main(["--old", "/nonexistent_old", "--roboflow", "/nonexistent_rf",
                     "--output", "/nonexistent_out_xyz", "--rf-drop-closeup", "2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
