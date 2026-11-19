"""Shared helpers for the test suite (makes scripts/ importable, prints steps, builds fake data)."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def step(msg: str) -> None:
    print(f"  [STEP] {msg}", flush=True)


def make_image(path: Path, seed: int, size=(96, 96)) -> Path:
    """Write a random-pattern JPEG that has a distinct perceptual hash per seed."""
    from PIL import Image, ImageDraw

    rnd = random.Random(seed)
    im = Image.new("RGB", size, tuple(rnd.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x1, y1 = rnd.randint(0, size[0] - 10), rnd.randint(0, size[1] - 10)
        d.rectangle([x1, y1, x1 + rnd.randint(5, 40), y1 + rnd.randint(5, 40)],
                    fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95)
    return path
