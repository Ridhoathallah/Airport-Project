#!/usr/bin/env python3
"""Evaluate a checkpoint with the vendored YOLOv5 validator and report the result.

Runs ``yolov5/val.py`` on a dataset split, parses the resulting metrics
using YOLOv5's own definitions, writes a JSON report, and -- when a
baseline is supplied -- compares the two under the acceptance criteria.

The candidate and the baseline are always evaluated on the *same* dataset
version and split; the report records both so a comparison can be audited
later.

Usage:
    venv/bin/python scripts/evaluate_model.py \
        --weights runs/train/people/weights/best.pt \
        --baseline models/yolov5s.pt \
        --data datasets/people_v1/data.yaml --split test

Exit codes:
    0  evaluation completed (a comparison may still be REJECT)
    1  evaluation or the comparison failed
    2  bad usage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog, DEFAULT_CATALOG_PATH  # noqa: E402
from ml.contract import sha256_file, utc_now_iso  # noqa: E402
from ml.evaluate import (  # noqa: E402
    DEFAULT_ACCEPTANCE_CRITERIA,
    EvaluationReport,
    compare,
    parse_results_csv,
    parse_val_stdout,
    write_report,
)
from ml.experiment import YOLOV5_ROOT  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="candidate checkpoint")
    parser.add_argument("--baseline", default=None, help="baseline checkpoint to compare against")
    parser.add_argument("--data", required=True, help="dataset YAML (the split is read from it)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--report", default=None, help="write the JSON report here")
    parser.add_argument("--dataset-version-id", default=None)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--experiment-id", default=None,
                        help="experiment these metrics belong to (catalog)")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--results-csv", default=None,
                        help="parse an existing results.csv instead of running validation")
    parser.add_argument("--no-single-cls", action="store_true",
                        help="do not pass --single-cls to the validator")
    return parser


def run_val(weights: str, args) -> tuple[str, str]:
    """Run the vendored validator; returns ``(stdout_text, run_dir)``.

    ``--single-cls`` is passed by default because the dataset schema is
    single-class (``person``). It is *required* to evaluate the stock
    80-class COCO checkpoint against it: YOLOv5 asserts that the
    checkpoint's class count matches the dataset, and with ``--single-cls``
    it maps every prediction to class 0 and uses class-agnostic NMS --
    which is exactly the comparison we want against a ``person``-only
    dataset.
    """
    command = [
        sys.executable,
        str(YOLOV5_ROOT / "val.py"),
        "--data", str(args.data),
        "--weights", str(weights),
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch_size),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--project", str(args.project),
        "--name", str(args.name),
        "--task", str(args.split),
    ]
    if not args.no_single_cls:
        command.append("--single-cls")
    result = subprocess.run(
        command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        print(output[-4000:], file=sys.stderr)
        raise RuntimeError(f"yolov5/val.py exited with code {result.returncode}")
    return output, str(Path(args.project) / args.name)


def evaluate(weights: str, args, tag: str) -> EvaluationReport:
    """Produce an evaluation report for one checkpoint."""
    if args.results_csv:
        metrics = parse_results_csv(args.results_csv)
    else:
        output, run_dir = run_val(weights, args)
        metrics = parse_val_stdout(output)
        csv_path = Path(run_dir) / "results.csv"
        if csv_path.is_file():
            from_csv = parse_results_csv(csv_path)
            if from_csv.map_50_95 is not None:
                metrics = from_csv
    path = Path(weights)
    report = EvaluationReport(
        model_path=str(path),
        model_sha256=sha256_file(path) if path.is_file() else None,
        dataset_version_id=args.dataset_version_id,
        dataset_version=args.dataset_version,
        split=args.split,
        metrics=metrics,
        evaluated_at_utc=utc_now_iso(),
        provenance={"role": tag, "evaluator": "yolov5/val.py", "imgsz": args.imgsz},
    )
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.data).exists():
        print(f"[ERROR] dataset YAML not found: {args.data}", file=sys.stderr)
        return EXIT_USAGE
    if not Path(args.weights).exists():
        print(f"[ERROR] checkpoint not found: {args.weights}", file=sys.stderr)
        return EXIT_USAGE

    try:
        candidate = evaluate(args.weights, args, "candidate")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(candidate.render_text())

    comparison = None
    if args.baseline:
        if not Path(args.baseline).exists():
            print(f"[ERROR] baseline checkpoint not found: {args.baseline}", file=sys.stderr)
            return EXIT_USAGE
        baseline = evaluate(args.baseline, args, "baseline")
        print()
        print(baseline.render_text())
        comparison = compare(baseline, candidate, DEFAULT_ACCEPTANCE_CRITERIA)
        print()
        print(comparison.render_text())
        if comparison.criteria.get("provisional"):
            print("[INFO] acceptance criteria are PROVISIONAL defaults, not an "
                  "agreed product requirement")

    payload = {
        "candidate": candidate.to_dict(),
        "comparison": comparison.to_dict() if comparison else None,
    }
    if args.report:
        write_report(args.report, payload)
        print(f"[INFO] report written: {args.report}")

    if not args.no_catalog and args.experiment_id:
        try:
            with Catalog(args.catalog) as catalog:
                stamp = candidate.evaluated_at_utc.replace("-", "").replace(":", "").replace(".", "")
                for name in ("precision", "recall", "map_50", "map_50_95"):
                    catalog.record_metric(
                        metric_id=f"met_{args.experiment_id}_{name}_{stamp}",
                        experiment_id=args.experiment_id,
                        metric_name=name,
                        value=candidate.metrics.get(name),
                        dataset_version_id=args.dataset_version_id,
                        split=args.split,
                    )
        except Exception as exc:
            print(f"[WARN] could not update the catalog: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    if comparison is not None and not comparison.passed:
        print("[REJECT] the candidate does not meet the acceptance criteria")
        return EXIT_OK
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
