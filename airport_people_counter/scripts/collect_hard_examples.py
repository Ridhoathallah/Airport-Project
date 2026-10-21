#!/usr/bin/env python3
"""Mine hard examples from source material for human review.

Scans a video or an image directory with the current checkpoint, flags
frames whose detection statistics match a known failure mode, and writes
a review sheet. The output is **machine-proposed** material: it is never
ground truth and never enters a dataset version without human review.

Privacy: frames of a camera contain identifiable people, so pixels are
only written when BOTH ``--retain-frames`` and ``--consent-acknowledged``
are given. Otherwise only derived statistics are produced.

Usage:
    venv/bin/python scripts/collect_hard_examples.py \
        --source input/clip.mp4 --weights models/yolov5s.pt \
        --out datasets/collected/run_001

Exit codes:
    0  mining completed
    1  mining failed
    2  bad usage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.collect import PRIVACY_NOTICE, mine_source  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="video file or directory of images")
    parser.add_argument("--weights", default="models/yolov5s.pt")
    parser.add_argument("--out", default=None, help="output directory for candidates")
    parser.add_argument("--stride", type=int, default=15,
                        help="inspect every Nth frame of a video (default 15)")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crowd-threshold", type=int, default=8,
                        help="detection count above which a frame is 'crowd_dense'")
    parser.add_argument("--small-area-ratio", type=float, default=0.005,
                        help="median box area ratio below which a frame is 'small_person'")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--retain-frames", action="store_true",
                        help="write the candidate frames as JPEG (requires --consent-acknowledged)")
    parser.add_argument("--consent-acknowledged", action="store_true",
                        help="assert the source may be processed and frames retained")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] source not found: {source}", file=sys.stderr)
        return EXIT_USAGE
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] checkpoint not found: {weights}", file=sys.stderr)
        return EXIT_USAGE

    print(PRIVACY_NOTICE)
    print()

    try:
        result = mine_source(
            source=source,
            weights=str(weights),
            out_dir=args.out,
            stride=args.stride,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            device=args.device,
            imgsz=args.imgsz,
            crowd_threshold=args.crowd_threshold,
            small_area_ratio=args.small_area_ratio,
            retain_frames=args.retain_frames,
            consent_acknowledged=args.consent_acknowledged,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        print(f"[ERROR] mining failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(f"[INFO] run id        : {result.run_id}")
    print(f"[INFO] frames examined: {result.frames_examined}")
    print(f"[INFO] candidates     : {len(result.candidates)}")
    print(f"[INFO] frames retained: {result.frames_retained}")
    if result.out_dir:
        print(f"[INFO] output         : {result.out_dir}")
        print(f"[INFO] review sheet   : {Path(result.out_dir) / 'REVIEW.md'}")
    print("[INFO] candidates are machine proposals -- a human must review them "
          "before they can be used as labels")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
