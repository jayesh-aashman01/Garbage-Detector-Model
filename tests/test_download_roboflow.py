"""Tests for scripts/download_roboflow.py (no network: HTTP is mocked).

Run:  .venv/bin/python -m unittest tests.test_download_roboflow -v
"""
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from _helpers import step

import download_roboflow as dr

KEY = "SECRETKEY123"


class EnvAndKeyTests(unittest.TestCase):
    def test_load_env_file_parses_and_ignores_comments(self):
        step("write a .env with a comment, blank line, quoted value")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("# comment\n\nROBOFLOW_API_KEY='abc'\nOTHER = x\nbadline\n")
            vals = dr.load_env_file(p)
            step(f"parsed keys -> {sorted(vals)}")
            self.assertEqual(vals["ROBOFLOW_API_KEY"], "abc")
            self.assertEqual(vals["OTHER"], "x")
            self.assertNotIn("badline", vals)

    def test_missing_env_file_returns_empty(self):
        step("load a .env that does not exist")
        self.assertEqual(dr.load_env_file(Path("/nonexistent/.env")), {})

    def test_placeholder_key_is_rejected(self):
        step("placeholder value from .env.example must stop the run")
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True):
            p = Path(d) / ".env"
            p.write_text(f"ROBOFLOW_API_KEY={dr.PLACEHOLDER}\n")
            with self.assertRaises(SystemExit) as cm:
                dr.get_api_key(p)
            step(f"error -> {cm.exception}")

    def test_environment_variable_wins_over_file(self):
        step("env var ROBOFLOW_API_KEY should be used before .env")
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": "fromenv"}):
            p = Path(d) / ".env"
            p.write_text("ROBOFLOW_API_KEY=fromfile\n")
            self.assertEqual(dr.get_api_key(p), "fromenv")

    def test_mask_hides_key(self):
        step("mask() must remove the key from text")
        self.assertEqual(dr.mask(f"https://x/?api_key={KEY}", KEY), "https://x/?api_key=***")


class ApiTests(unittest.TestCase):
    def _resp(self, status, payload=None, text=""):
        r = mock.Mock(status_code=status, text=text)
        r.json.return_value = payload or {}
        return r

    def test_api_error_never_leaks_key(self):
        step("HTTP 500 whose body echoes the key -> message must be masked")
        with mock.patch.object(dr.requests, "get", return_value=self._resp(500, text=f"bad {KEY}")):
            with self.assertRaises(SystemExit) as cm:
                dr.api_get("https://api.roboflow.com/w/p", KEY)
        step(f"error -> {cm.exception}")
        self.assertNotIn(KEY, str(cm.exception))

    def test_bad_key_message(self):
        step("HTTP 401 -> clear message about the key")
        with mock.patch.object(dr.requests, "get", return_value=self._resp(401)):
            with self.assertRaises(SystemExit) as cm:
                dr.api_get("https://api.roboflow.com/w/p", KEY)
        self.assertIn("API key", str(cm.exception))

    def test_401_shows_reason_source_and_masks_key(self):
        step("401 whose body echoes the key -> reason shown, key masked, source named")
        body = f'{{"error": "This API key does not exist (or has been revoked). key={KEY}"}}'
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True):
            env = Path(d) / ".env"
            env.write_text(f"ROBOFLOW_API_KEY={KEY}\n")
            dr.get_api_key(env)
            with mock.patch.object(dr.requests, "get", return_value=self._resp(401, text=body)):
                with self.assertRaises(SystemExit) as cm:
                    dr.api_get("https://api.roboflow.com/w/p", KEY)
        msg = str(cm.exception)
        step(f"error ->\n{msg}")
        self.assertIn("does not exist", msg)
        self.assertNotIn(KEY, msg)
        self.assertIn("file", msg)
        self.assertIn("length 12", msg)

    def test_key_source_env_variable(self):
        step("key from shell env -> KEY_SOURCE names the environment variable")
        with mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": KEY}):
            dr.get_api_key(Path("/nonexistent/.env"))
        step(f"KEY_SOURCE -> {dr.KEY_SOURCE}")
        self.assertIn("environment variable", dr.KEY_SOURCE)

    def test_401_with_empty_body(self):
        step("401 with empty body -> '(empty response)'")
        with mock.patch.object(dr.requests, "get", return_value=self._resp(401, text="")):
            with self.assertRaises(SystemExit) as cm:
                dr.api_get("https://api.roboflow.com/w/p", KEY)
        self.assertIn("(empty response)", str(cm.exception))

    def test_resolve_latest_version(self):
        step("versions w/p/1, w/p/3, w/p/2 -> latest is 3")
        payload = {"versions": [{"id": "w/p/1"}, {"id": "w/p/3"}, {"id": "w/p/2"}]}
        with mock.patch.object(dr, "api_get", return_value=payload):
            self.assertEqual(dr.resolve_version("w", "p", "latest", KEY), "3")
        step("explicit version is passed through")
        self.assertEqual(dr.resolve_version("w", "p", "7", KEY), "7")

    def test_resolve_version_none_found(self):
        step("no versions -> SystemExit asking for --version")
        with mock.patch.object(dr, "api_get", return_value={"versions": []}):
            with self.assertRaises(SystemExit):
                dr.resolve_version("w", "p", "latest", KEY)

    def test_export_link_polls_until_ready(self):
        step("first call: progress 0.5, second call: link ready")
        seq = [{"progress": 0.5}, {"export": {"link": "https://storage.example/x.zip"}}]
        with mock.patch.object(dr, "api_get", side_effect=seq), mock.patch.object(dr.time, "sleep") as sl:
            link = dr.get_export_link("w", "p", "1", "yolov11", KEY, poll_seconds=1)
        step(f"link -> {link}, sleeps -> {sl.call_count}")
        self.assertEqual(link, "https://storage.example/x.zip")
        self.assertEqual(sl.call_count, 1)

    def test_export_link_times_out(self):
        step("never ready -> SystemExit after max_wait")
        with mock.patch.object(dr, "api_get", return_value={"progress": 0.1}), mock.patch.object(dr.time, "sleep"):
            with self.assertRaises(SystemExit):
                dr.get_export_link("w", "p", "1", "yolov11", KEY, poll_seconds=5, max_wait=10)


class ExtractTests(unittest.TestCase):
    def test_safe_extract_normal_zip(self):
        step("extract a zip with data.yaml and one label")
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "a.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("data.yaml", "names: [a]\n")
                zf.writestr("train/labels/x.txt", "0 0.5 0.5 0.1 0.1\n")
            n = dr.safe_extract(z, Path(d) / "out")
            step(f"files extracted -> {n}")
            self.assertEqual(n, 2)
            self.assertTrue((Path(d) / "out/train/labels/x.txt").is_file())

    def test_safe_extract_blocks_zip_slip(self):
        step("zip containing ../evil.txt must be refused")
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "evil.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("../evil.txt", "x")
            with self.assertRaises(SystemExit) as cm:
                dr.safe_extract(z, Path(d) / "out")
            step(f"error -> {cm.exception}")
            self.assertFalse((Path(d) / "evil.txt").exists())

    def test_main_skips_when_already_downloaded(self):
        step("existing data.yaml + no --overwrite -> nothing downloaded")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data.yaml").write_text("names: [a]\n")
            with mock.patch.object(dr, "get_api_key") as gk:
                self.assertEqual(dr.main(["--output", d]), 0)
                gk.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
