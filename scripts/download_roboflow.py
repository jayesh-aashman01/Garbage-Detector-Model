#!/usr/bin/env python3
"""Download a Roboflow Universe dataset version as a YOLO export (no Roboflow SDK needed).

Uses the documented REST route  GET https://api.roboflow.com/<workspace>/<project>/<version>/<format>?api_key=...
which answers with {"export": {"link": ...}} once the export is ready (it may first
answer with a progress value while Roboflow builds the zip, so we poll).
Docs: https://docs.roboflow.com/developer/rest-api/export-data

The API key is read from the ROBOFLOW_API_KEY environment variable or the project's
.env file. It is never printed: every URL / error message is passed through mask().
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_log import dbg, enable  # noqa: E402

API = "https://api.roboflow.com"
DEFAULT_WORKSPACE = "noyals-workspace"
DEFAULT_PROJECT = "human-and-trash-detection-tciuj-3hmlf"
PLACEHOLDER = "paste-your-roboflow-api-key-here"


def load_env_file(path: Path) -> dict:
    """Parse simple KEY=VALUE lines from a .env file (comments and blanks ignored)."""
    dbg("load_env_file", f"enter path={path} exists={path.is_file()}")
    values = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    dbg("load_env_file", f"exit keys={sorted(values)}  (values hidden)")
    return values


def get_api_key(env_path: Path) -> str:
    """Return the API key from the environment first, then .env. Raises SystemExit if missing."""
    key = os.environ.get("ROBOFLOW_API_KEY") or load_env_file(env_path).get("ROBOFLOW_API_KEY", "")
    dbg("get_api_key", f"key found={bool(key)} placeholder={key == PLACEHOLDER} length={len(key)}")
    if not key or key == PLACEHOLDER:
        raise SystemExit(
            f"ROBOFLOW_API_KEY is not set. Paste it into {env_path} (see .env.example) "
            "or export ROBOFLOW_API_KEY=... in your shell."
        )
    return key


def mask(text: str, key: str) -> str:
    """Remove the API key from any string before it is printed or raised."""
    return str(text).replace(key, "***") if key else str(text)


def api_get(url: str, key: str, timeout: int = 60) -> dict:
    """GET a Roboflow API URL and return JSON. Errors are re-raised with the key masked."""
    dbg("api_get", f"enter url={mask(url, key)}")
    try:
        resp = requests.get(url, params={"api_key": key}, timeout=timeout)
    except requests.RequestException as exc:
        raise SystemExit(f"Network error talking to Roboflow: {mask(exc, key)}") from None
    dbg("api_get", f"status={resp.status_code}")
    if resp.status_code in (401, 403):
        raise SystemExit("Roboflow rejected the API key (HTTP %d). Check ROBOFLOW_API_KEY." % resp.status_code)
    if resp.status_code == 404:
        raise SystemExit(f"Not found on Roboflow: {mask(url, key)}. Check workspace/project/version.")
    if resp.status_code >= 400:
        raise SystemExit(f"Roboflow error HTTP {resp.status_code}: {mask(resp.text[:300], key)}")
    try:
        data = resp.json()
    except ValueError:
        raise SystemExit(f"Roboflow returned non-JSON: {mask(resp.text[:200], key)}") from None
    dbg("api_get", f"exit json keys={list(data)[:10]}")
    return data


def resolve_version(workspace: str, project: str, version: str, key: str) -> str:
    """Turn 'latest' into the highest version number listed for the project."""
    dbg("resolve_version", f"enter workspace={workspace} project={project} version={version}")
    if version != "latest":
        return str(int(version))
    info = api_get(f"{API}/{workspace}/{project}", key)
    numbers = []
    for v in info.get("versions", []) or []:
        vid = str(v.get("id", "")) if isinstance(v, dict) else ""
        tail = vid.rsplit("/", 1)[-1]
        if tail.isdigit():
            numbers.append(int(tail))
    dbg("resolve_version", f"versions found={numbers}")
    if not numbers:
        raise SystemExit("Could not find any dataset versions. Pass --version N explicitly "
                         "(the number shown on the Roboflow dataset page).")
    return str(max(numbers))


def get_export_link(workspace, project, version, fmt, key, poll_seconds=5, max_wait=600) -> str:
    """Ask Roboflow for the export zip link, polling while the export is being generated."""
    url = f"{API}/{workspace}/{project}/{version}/{fmt}"
    dbg("get_export_link", f"enter url={url}")
    waited = 0
    while True:
        data = api_get(url, key)
        link = (data.get("export") or {}).get("link")
        if link:
            dbg("get_export_link", f"exit link received (host={link.split('/')[2] if '//' in link else '?'})")
            return link
        progress = data.get("progress")
        print(f"Roboflow is preparing the export (progress={progress}); waiting {poll_seconds}s ...")
        if waited >= max_wait:
            raise SystemExit("Timed out waiting for Roboflow to build the export. Try again later.")
        time.sleep(poll_seconds)
        waited += poll_seconds


def download_file(link: str, dest: Path, key: str) -> Path:
    """Stream the zip to dest (via a .part file so a failed download never looks complete)."""
    dbg("download_file", f"enter dest={dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(link, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            total = 0
            with part.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
                    total += len(chunk)
        os.replace(part, dest)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {mask(exc, key)}") from None
    dbg("download_file", f"exit bytes={total}")
    return dest


def safe_extract(zip_path: Path, out_dir: Path) -> int:
    """Extract a zip, refusing any entry that would land outside out_dir (zip-slip protection)."""
    dbg("safe_extract", f"enter zip={zip_path} out={out_dir}")
    out_root = out_dir.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_root / member.filename).resolve()
            if target != out_root and out_root not in target.parents:
                raise SystemExit(f"Unsafe path in zip, refusing to extract: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    dbg("safe_extract", f"exit files extracted={count}")
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--version", default="latest", help="dataset version number, or 'latest'")
    ap.add_argument("--format", default="yolov11", help="Roboflow export format (YOLOv11 PyTorch TXT)")
    ap.add_argument("--output", type=Path, default=Path("datasets_raw/roboflow"))
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--overwrite", action="store_true", help="delete an existing extracted copy first")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    enable(args.debug)

    if (args.output / "data.yaml").is_file() and not args.overwrite:
        print(f"Already downloaded: {args.output}/data.yaml (use --overwrite to fetch again)")
        return 0

    key = get_api_key(args.env_file)
    version = resolve_version(args.workspace, args.project, args.version, key)
    print(f"Fetching {args.workspace}/{args.project} version {version} as {args.format}")
    link = get_export_link(args.workspace, args.project, version, args.format, key)

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    zip_path = args.output.parent / f"{args.output.name}_v{version}.zip"
    download_file(link, zip_path, key)
    n = safe_extract(zip_path, args.output)
    zip_path.unlink(missing_ok=True)

    if not (args.output / "data.yaml").is_file():
        raise SystemExit(f"Extracted {n} files but no data.yaml found in {args.output}. "
                         f"Check that --format is a YOLO format.")
    print(f"Done: {n} files in {args.output}. Next: python scripts/merge_datasets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
