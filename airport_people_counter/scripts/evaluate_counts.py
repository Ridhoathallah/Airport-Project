#!/usr/bin/env python3
"""Offline evaluation of recorded queue occupancy against manual ground truth.

Compares a samples JSONL written by the runtime
(`python app/detect_people.py --record-samples output/samples.jsonl`)
against manually annotated ground-truth intervals (CSV), and reports:

    MAE            mean absolute error, people
    +/-1 accuracy  fraction of pairs with |error| <= 1 (inclusive)
    +/-2 accuracy  fraction of pairs with |error| <= 2 (inclusive)
    Bias           signed mean error (pred - gt), people

Timeline: ``t_rel_s`` (seconds since the pipeline's first processed frame)
is the canonical evaluation timeline. Ground truth is annotated on the same
video timeline; each CSV row is a [start_s, end_s) interval whose ``count``
is the spot-count observed at the interval midpoint. Each interval is matched
to the prediction sample nearest its midpoint; the match must lie within
[start_s - tolerance, end_s + tolerance].

The evaluator never silently discards data: excluded intervals are counted
and reported with their reason (recording gap / no matching sample /
unconfigured ROI). If coverage falls below --min-coverage no accuracy verdict
is emitted and the process exits non-zero.

Exit codes:
    0  evaluation completed
    1  input parsed but data insufficient for a trustworthy verdict
    2  malformed input (bad JSON/CSV, missing fields, invalid values)

Usage:
    venv/bin/python scripts/evaluate_counts.py \
        --samples output/samples.jsonl \
        --ground-truth output/ground_truth.csv \
        [--tolerance-s 1.0] [--gap-s 10.0] [--min-coverage 0.8] \
        [--json output/eval_report.json]

This is an offline measurement tool only. It does not run detection,
modify thresholds, or touch the camera pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from bisect import bisect_left
from pathlib import Path

EXIT_OK = 0
EXIT_INSUFFICIENT_DATA = 1
EXIT_INPUT_ERROR = 2

SAMPLES_REQUIRED_FIELDS = ("record_type", "t_rel_s", "people_in_queue", "roi_configured")
GT_REQUIRED_FIELDS = ("start_s", "end_s", "count")


class InputError(Exception):
    """Malformed or invalid input. Maps to exit code 2."""


class InsufficientDataError(Exception):
    """Input parsed but coverage too low for a trustworthy verdict. Exit code 1."""


# ---------------------------------------------------------------------
#  LOADERS
# ---------------------------------------------------------------------


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def load_samples(path):
    """Load an occupancy-samples JSONL file.

    Returns (rows, warnings). Rows are dicts {t, count, roi} sorted by time.
    Duplicate timestamps keep the first occurrence and produce a warning.
    """
    path = Path(path)
    rows = []
    warnings = []
    try:
        text = path.read_text()
    except OSError as exc:
        raise InputError(f"cannot read samples file {path}: {exc}")

    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{lineno}: malformed JSON ({exc})")
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{lineno}: expected a JSON object")
        missing = [f for f in SAMPLES_REQUIRED_FIELDS if f not in obj]
        if missing:
            raise InputError(f"{path}:{lineno}: missing field(s): {', '.join(missing)}")
        if obj["record_type"] != "occupancy_sample":
            raise InputError(
                f"{path}:{lineno}: unexpected record_type {obj['record_type']!r} "
                "(expected 'occupancy_sample')"
            )
        t = obj["t_rel_s"]
        if not _is_number(t):
            raise InputError(f"{path}:{lineno}: invalid t_rel_s {t!r}")
        count = obj["people_in_queue"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise InputError(f"{path}:{lineno}: invalid people_in_queue {count!r} (need int >= 0)")
        roi = obj["roi_configured"]
        if not isinstance(roi, bool):
            raise InputError(f"{path}:{lineno}: invalid roi_configured {roi!r} (need boolean)")
        rows.append({"t": float(t), "count": count, "roi": roi})

    if not rows:
        raise InputError(f"{path}: no samples found (file is empty)")

    rows.sort(key=lambda r: r["t"])
    deduped = []
    duplicates = 0
    for row in rows:
        if deduped and row["t"] == deduped[-1]["t"]:
            duplicates += 1
            continue
        deduped.append(row)
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate timestamp(s) in {path.name} — first occurrence kept"
        )
    return deduped, warnings


def load_ground_truth(path):
    """Load ground-truth intervals from CSV (columns: start_s,end_s,count[,note]).

    Returns a list of dicts {start, end, count} sorted by start.
    """
    path = Path(path)
    try:
        handle = path.open(newline="")
    except OSError as exc:
        raise InputError(f"cannot read ground-truth file {path}: {exc}")

    intervals = []
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputError(f"{path}: empty ground-truth file (no header row)")
        missing = [f for f in GT_REQUIRED_FIELDS if f not in reader.fieldnames]
        if missing:
            raise InputError(
                f"{path}: missing column(s): {', '.join(missing)} "
                f"(found: {', '.join(reader.fieldnames)})"
            )
        for lineno, row in enumerate(reader, 2):
            start = _parse_csv_float(row["start_s"], "start_s", path, lineno, min_value=0.0)
            end = _parse_csv_float(row["end_s"], "end_s", path, lineno, min_value=0.0)
            count_raw = (row["count"] or "").strip()
            try:
                count = int(count_raw)
            except ValueError:
                raise InputError(
                    f"{path}:{lineno}: invalid count {row['count']!r} (need integer >= 0)"
                )
            if count < 0:
                raise InputError(f"{path}:{lineno}: negative count {count}")
            if end <= start:
                raise InputError(f"{path}:{lineno}: end_s ({end}) must be > start_s ({start})")
            intervals.append({"start": start, "end": end, "count": count})

    if not intervals:
        raise InputError(f"{path}: no ground-truth intervals found")

    intervals.sort(key=lambda iv: iv["start"])
    for prev, cur in zip(intervals, intervals[1:]):
        if cur["start"] < prev["end"]:
            raise InputError(
                f"{path}: overlapping intervals [{prev['start']}, {prev['end']}) and "
                f"[{cur['start']}, {cur['end']})"
            )
    return intervals


def _parse_csv_float(raw, field, path, lineno, min_value=None):
    raw = (raw or "").strip()
    try:
        value = float(raw)
    except ValueError:
        raise InputError(f"{path}:{lineno}: invalid {field} {raw!r}")
    if not math.isfinite(value):
        raise InputError(f"{path}:{lineno}: non-finite {field} {raw!r}")
    if min_value is not None and value < min_value:
        raise InputError(f"{path}:{lineno}: {field} ({value}) must be >= {min_value}")
    return value


# ---------------------------------------------------------------------
#  ALIGNMENT + SCORING
# ---------------------------------------------------------------------


def detect_gaps(samples, gap_s):
    """Return [(start_t, end_t)] recording gaps longer than the threshold.

    Threshold = max(gap_s, 5 x median inter-sample delta), so a coarse but
    regular sample rate is not flagged while real dropouts are.
    """
    if len(samples) < 2:
        return []
    deltas = [b["t"] - a["t"] for a, b in zip(samples, samples[1:])]
    threshold = max(gap_s, 5.0 * statistics.median(deltas))
    return [
        (a["t"], b["t"])
        for a, b in zip(samples, samples[1:])
        if b["t"] - a["t"] > threshold
    ]


def nearest_index(times, target):
    """Index of the time nearest to target; ties resolve to the earlier sample."""
    if not times:
        raise InputError("no samples available for alignment")
    i = bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    return min(candidates, key=lambda j: abs(times[j] - target))


def evaluate(samples, intervals, tolerance_s=1.0, gap_s=10.0, min_coverage=0.8,
             warnings=None):
    """Align ground-truth intervals to samples and compute metrics.

    Raises InsufficientDataError when coverage < min_coverage. Never drops
    an interval silently — every exclusion is counted with its reason.
    """
    if tolerance_s < 0:
        raise InputError(f"--tolerance-s must be >= 0 (got {tolerance_s})")
    if gap_s <= 0:
        raise InputError(f"--gap-s must be > 0 (got {gap_s})")
    if not (0.0 < min_coverage <= 1.0):
        raise InputError(f"--min-coverage must be in (0, 1] (got {min_coverage})")

    times = [s["t"] for s in samples]
    gaps = detect_gaps(samples, gap_s)

    def inside_gap(t):
        return any(a <= t <= b for a, b in gaps)

    excluded_gap = excluded_no_match = excluded_unconfigured = 0
    pairs = []
    for iv in intervals:
        mid = (iv["start"] + iv["end"]) / 2.0
        if inside_gap(mid):
            excluded_gap += 1
            continue
        idx = nearest_index(times, mid)
        sample_t = times[idx]
        if sample_t < iv["start"] - tolerance_s or sample_t > iv["end"] + tolerance_s:
            excluded_no_match += 1
            continue
        sample = samples[idx]
        if not sample["roi"]:
            excluded_unconfigured += 1
            continue
        error = sample["count"] - iv["count"]
        pairs.append({
            "interval": [iv["start"], iv["end"]],
            "gt": iv["count"],
            "pred": sample["count"],
            "error": error,
            "sample_t_rel_s": sample_t,
        })

    total = len(intervals)
    evaluated = len(pairs)
    coverage = evaluated / total

    if coverage < min_coverage:
        lines = [
            f"coverage {coverage:.1%} below required minimum {min_coverage:.0%}.",
            f"  ground-truth intervals : {total}",
            f"  evaluated              : {evaluated}",
            f"  excluded (recording gap): {excluded_gap}",
            f"  excluded (no match within ±{tolerance_s}s of midpoint): {excluded_no_match}",
            f"  excluded (ROI unconfigured): {excluded_unconfigured}",
            "No accuracy verdict emitted.",
        ]
        raise InsufficientDataError("\n".join(lines))

    errors = [p["error"] for p in pairs]
    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n
    within_1 = sum(1 for e in errors if abs(e) <= 1) / n
    within_2 = sum(1 for e in errors if abs(e) <= 2) / n

    return {
        "metrics": {
            "pairs": n,
            "mae": round(mae, 3),
            "within_1_accuracy": round(within_1, 4),
            "within_2_accuracy": round(within_2, 4),
            "bias": round(bias, 3),
        },
        "accuracy_percent": {
            "within_1": round(within_1 * 100.0, 1),
            "within_2": round(within_2 * 100.0, 1),
        },
        "coverage": {
            "total_intervals": total,
            "evaluated": evaluated,
            "excluded_recording_gap": excluded_gap,
            "excluded_no_match": excluded_no_match,
            "excluded_roi_unconfigured": excluded_unconfigured,
            "ratio": round(coverage, 4),
        },
        "gaps": [
            {"start_s": round(a, 3), "end_s": round(b, 3), "duration_s": round(b - a, 3)}
            for a, b in gaps
        ],
        "parameters": {
            "timeline": "t_rel_s",
            "tolerance_s": tolerance_s,
            "gap_s": gap_s,
            "min_coverage": min_coverage,
        },
        "warnings": list(warnings or []),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------
#  OUTPUT
# ---------------------------------------------------------------------


def format_report(report) -> str:
    m = report["metrics"]
    c = report["coverage"]
    lines = [
        "=" * 50,
        "  OCCUPANCY EVALUATION",
        "=" * 50,
        "",
        f"Pairs evaluated : {c['evaluated']} / {c['total_intervals']}"
        f"   (coverage {c['ratio']:.1%})",
        f"MAE             : {m['mae']:.3f} people",
        f"+/-1 accuracy   : {report['accuracy_percent']['within_1']:.1f}%",
        f"+/-2 accuracy   : {report['accuracy_percent']['within_2']:.1f}%",
        f"Bias (pred-gt)  : {m['bias']:+.3f} people",
        "",
        f"Recording gaps  : {len(report['gaps'])}",
        f"Excluded        : {c['excluded_recording_gap']} gap,"
        f" {c['excluded_no_match']} no-match,"
        f" {c['excluded_roi_unconfigured']} ROI-unconfigured",
    ]
    for warning in report["warnings"]:
        lines.append(f"[WARN] {warning}")
    return "\n".join(lines)


def write_json_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate recorded queue occupancy against manual ground truth "
                    "(offline; MAE / +/-1 / +/-2 / bias)"
    )
    parser.add_argument("--samples", required=True,
                        help="Samples JSONL from `detect_people.py --record-samples`")
    parser.add_argument("--ground-truth", required=True, dest="ground_truth",
                        help="Ground-truth CSV (columns: start_s,end_s,count[,note])")
    parser.add_argument("--tolerance-s", type=float, default=1.0,
                        help="Match window around each interval midpoint, seconds (default: 1.0)")
    parser.add_argument("--gap-s", type=float, default=10.0,
                        help="Minimum recording-gap duration to flag, seconds (default: 10.0)")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="Required evaluated-interval fraction, 0-1 (default: 0.8)")
    parser.add_argument("--json", default=None, dest="json_out",
                        help="Write the full report as JSON to this path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        samples, warnings = load_samples(args.samples)
        intervals = load_ground_truth(args.ground_truth)
        all_unconfigured = samples and all(not s["roi"] for s in samples)
        if all_unconfigured:
            raise InputError(
                "all samples have roi_configured=false — queue occupancy is "
                "undefined until a ROI is configured (--define-roi)"
            )
        report = evaluate(
            samples, intervals,
            tolerance_s=args.tolerance_s,
            gap_s=args.gap_s,
            min_coverage=args.min_coverage,
            warnings=warnings,
        )
    except InputError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except InsufficientDataError as exc:
        print("[ERROR] INSUFFICIENT DATA — refusing to emit accuracy numbers.",
              file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_INSUFFICIENT_DATA

    print(format_report(report))
    if args.json_out:
        write_json_report(report, args.json_out)
        print(f"[INFO] JSON report written: {args.json_out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
