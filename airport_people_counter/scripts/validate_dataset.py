#!/usr/bin/env python3
"""Validate a YOLOv5 people-detection dataset and write a machine-readable report.

Checks images, labels, class ids, box geometry, duplicates and split
consistency. Nothing in the dataset is modified.

Usage:
    venv/bin/python scripts/validate_dataset.py --root datasets/people_v1 \
        [--report datasets/reports/people_v1.json] [--preview-out datasets/reports/preview]

Exit codes:
    0  no error-level findings
    1  the dataset has error-level findings
    2  the dataset root does not exist or is not a dataset directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.contract import DEFAULT_CLASS_SCHEMA, AnnotationStatus  # noqa: E402
from ml.validate import validate_dataset  # noqa: E402

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="dataset root containing images/ and labels/")
    parser.add_argument("--splits", default="train,val,test", help="comma-separated split names")
    parser.add_argument("--report", default=None, help="write the JSON report to this path")
    parser.add_argument("--annotation-status", default=AnnotationStatus.UNREVIEWED.value,
                        help="status recorded for images that have a label file")
    parser.add_argument("--no-image-check", action="store_true",
                        help="skip reading image pixels (faster, less thorough)")
    parser.add_argument("--no-near-duplicate", action="store_true",
                        help="skip perceptual-hash near-duplicate detection")
    parser.add_argument("--preview-out", default=None,
                        help="render annotation previews into this directory")
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] dataset root not found: {root}", file=sys.stderr)
        return EXIT_USAGE

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report, records = validate_dataset(
        root,
        class_schema=DEFAULT_CLASS_SCHEMA,
        splits=splits,
        check_images=not args.no_image_check,
        check_near_duplicates=not args.no_near_duplicate,
        annotation_status=args.annotation_status,
    )

    if not args.quiet:
        print(report.render_text())

    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"[INFO] report written: {target}")

    if args.preview_out:
        from ml.preview import preview_dataset

        for split in splits:
            written = preview_dataset(root, split, Path(args.preview_out) / split,
                                      limit=args.preview_limit)
            if written and not args.quiet:
                print(f"[INFO] {len(written)} preview(s) for split {split}")

    if not report.ok:
        print(f"[FAIL] {len(report.errors)} error-level finding(s)", file=sys.stderr)
        return EXIT_FINDINGS
    print("[OK] dataset passed validation")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
