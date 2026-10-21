#!/usr/bin/env python3
"""Generate a SYNTHETIC dataset for smoke-testing the training pipeline.

The images are procedurally drawn shapes. They are **not photographs of
people** and carry no detection value whatsoever: their only purpose is to
prove that the plumbing works -- that YOLOv5 accepts the generated
``data.yaml``, that a training epoch completes, that checkpoints are
written, and that the experiment metadata is recorded.

For safety this script **refuses to write inside the repository's
``datasets/`` directory**, so synthetic data can never be mistaken for
real, reviewed data.

Usage:
    venv/bin/python scripts/make_synthetic_fixture.py --out /tmp/ml-fixture \
        --images 24 --seed 7

Exit codes:
    0  fixture generated
    1  generation failed
    2  bad usage (including a refusal to write into datasets/)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

#: Splits and their share of the generated images.
SPLIT_SHARES = (("train", 0.6), ("val", 0.2), ("test", 0.2))


def _draw_person(draw, box, colour) -> None:
    """Draw a crude standing figure inside ``box`` (x, y, w, h)."""
    x, y, w, h = box
    head_r = max(2, int(min(w, h) * 0.18))
    draw.ellipse(
        (x + w // 2 - head_r, y, x + w // 2 + head_r, y + 2 * head_r),
        fill=colour,
    )
    body_top = y + 2 * head_r
    draw.rectangle(
        (x + w // 4, body_top, x + 3 * w // 4, y + h),
        fill=colour,
    )


def generate(out_dir: Path, images: int, seed: int, width: int, height: int) -> dict:
    """Create the synthetic dataset and return a summary."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict = {}
    index = 0
    for split, share in SPLIT_SHARES:
        split_images = max(1, round(images * share))
        counts[split] = split_images
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(split_images):
            index += 1
            background = (rng.randint(20, 90), rng.randint(20, 90), rng.randint(20, 90))
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            rows = []
            for _ in range(rng.randint(0, 5)):
                person_w = rng.randint(width // 12, width // 5)
                person_h = rng.randint(height // 4, height // 2)
                x = rng.randint(0, max(0, width - person_w))
                y = rng.randint(0, max(0, height - person_h))
                colour = (rng.randint(120, 250), rng.randint(120, 250), rng.randint(120, 250))
                _draw_person(draw, (x, y, person_w, person_h), colour)
                xc = (x + person_w / 2) / width
                yc = (y + person_h / 2) / height
                rows.append((0, xc, yc, person_w / width, person_h / height))
            image.save(image_dir / f"syn_{index:05d}.jpg", quality=90)
            label_dir.joinpath(f"syn_{index:05d}.txt").write_text(
                "".join("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row) for row in rows),
                encoding="utf-8",
            )
    return {"out_dir": str(out_dir), "images": index, "splits": counts, "seed": seed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory (must be outside datasets/)")
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    protected = (PROJECT_ROOT / "datasets").resolve()
    if protected in out_dir.parents or out_dir == protected:
        print("[ERROR] refusing to write synthetic data into the real dataset tree: "
              f"{protected}", file=sys.stderr)
        print("[INFO] choose a directory outside the repository, e.g. /tmp/ml-fixture",
              file=sys.stderr)
        return EXIT_USAGE
    if out_dir.exists():
        if not args.force:
            print(f"[ERROR] output directory already exists: {out_dir} (use --force)",
                  file=sys.stderr)
            return EXIT_USAGE
        shutil.rmtree(out_dir)

    try:
        summary = generate(out_dir, args.images, args.seed, args.width, args.height)
    except Exception as exc:
        print(f"[ERROR] fixture generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print("[WARN] SYNTHETIC DATA -- procedurally drawn shapes, not photographs, "
          "no detection value")
    for key, value in summary.items():
        print(f"[INFO] {key:<10}: {value}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
