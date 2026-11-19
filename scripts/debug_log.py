"""Switchable debug logging shared by the pipeline scripts.

Turn on with the environment variable GD_DEBUG=1 (or by calling enable(True),
which every script does for its --debug flag). Every line is prefixed with
[DEBUG][function_name] so it is easy to grep for or strip out.
"""
import os
import sys

_ENABLED = os.environ.get("GD_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def enable(flag: bool = True) -> None:
    """Switch debug output on or off at runtime."""
    global _ENABLED
    _ENABLED = bool(flag) or _ENABLED


def is_enabled() -> bool:
    return _ENABLED


def dbg(func: str, msg: str) -> None:
    """Print one debug line as `[DEBUG][func] msg` to stderr when enabled."""
    if _ENABLED:
        print(f"[DEBUG][{func}] {msg}", file=sys.stderr, flush=True)
