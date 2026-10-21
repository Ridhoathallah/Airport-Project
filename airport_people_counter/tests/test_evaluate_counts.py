"""Tests for scripts/evaluate_counts.py (offline occupancy evaluation).

Loads the script by path so the tests do not depend on package layout.
All fixtures are written to temporary directories; nothing touches
config/, output/, or the runtime modules.
"""

import contextlib
import csv
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "evaluate_counts.py"
_spec = importlib.util.spec_from_file_location("evaluate_counts", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)


def sample_row(t, count, roi=True):
    return {
        "record_type": "occupancy_sample",
        "t_rel_s": t,
        "event_time": "2026-08-26T09:00:00.000Z",
        "source_frame": None,
        "source_fps": None,
        "people_in_queue": count,
        "state": "NORMAL",
        "roi_configured": roi,
    }


def write_samples(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def write_gt(path, intervals):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start_s", "end_s", "count", "note"])
        for start, end, count in intervals:
            writer.writerow([start, end, count, ""])
    return path


def uniform_samples(t0=0.0, step=1.0, n=20, counts=None, roi=True):
    counts = counts if counts is not None else [0] * n
    return [
        sample_row(round(t0 + i * step, 3), counts[i % len(counts)], roi)
        for i in range(n)
    ]


def norm(rows):
    """Reduce full sample rows to the {t, count, roi} shape load_samples returns."""
    return [{"t": float(r["t_rel_s"]), "count": r["people_in_queue"],
             "roi": r["roi_configured"]} for r in rows]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestLoadSamples(Base):
    def test_parses_valid_rows_sorted_by_time(self):
        path = write_samples(self.dir / "s.jsonl", [
            sample_row(5.0, 2), sample_row(1.0, 0),
        ])
        rows, warnings = ec.load_samples(path)
        self.assertEqual([r["t"] for r in rows], [1.0, 5.0])
        self.assertEqual(warnings, [])

    def test_malformed_json_reports_line_number(self):
        path = self.dir / "bad.jsonl"
        path.write_text('{"record_type": "occupancy_sample"\n')
        with self.assertRaises(ec.InputError) as ctx:
            ec.load_samples(path)
        self.assertIn("line 1", str(ctx.exception))

    def test_missing_required_field(self):
        row = sample_row(0.0, 1)
        del row["people_in_queue"]
        path = write_samples(self.dir / "s.jsonl", [row])
        with self.assertRaises(ec.InputError) as ctx:
            ec.load_samples(path)
        self.assertIn("people_in_queue", str(ctx.exception))

    def test_wrong_record_type_rejected(self):
        row = sample_row(0.0, 1)
        row["record_type"] = "something_else"
        path = write_samples(self.dir / "s.jsonl", [row])
        with self.assertRaises(ec.InputError) as ctx:
            ec.load_samples(path)
        self.assertIn("record_type", str(ctx.exception))

    def test_negative_count_rejected(self):
        path = write_samples(self.dir / "s.jsonl", [sample_row(0.0, -1)])
        with self.assertRaises(ec.InputError):
            ec.load_samples(path)

    def test_non_integer_count_rejected(self):
        path = write_samples(self.dir / "s.jsonl", [sample_row(0.0, 1.5)])
        with self.assertRaises(ec.InputError):
            ec.load_samples(path)

    def test_invalid_t_rel_s_rejected(self):
        for bad in ("x", None, True, float("inf")):
            path = write_samples(self.dir / "s.jsonl", [sample_row(bad, 1)])
            with self.assertRaises(ec.InputError):
                ec.load_samples(path)

    def test_duplicate_timestamps_warn_and_keep_first(self):
        path = write_samples(self.dir / "s.jsonl", [
            sample_row(1.0, 3), sample_row(1.0, 9), sample_row(2.0, 4),
        ])
        rows, warnings = ec.load_samples(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["count"], 3)
        self.assertEqual(len(warnings), 1)

    def test_empty_file_rejected(self):
        path = self.dir / "empty.jsonl"
        path.write_text("")
        with self.assertRaises(ec.InputError):
            ec.load_samples(path)


class TestLoadGroundTruth(Base):
    def test_loads_intervals_with_optional_note_column(self):
        path = write_gt(self.dir / "gt.csv", [(0, 10, 3), (10, 20, 5)])
        intervals = ec.load_ground_truth(path)
        self.assertEqual(intervals, [
            {"start": 0.0, "end": 10.0, "count": 3},
            {"start": 10.0, "end": 20.0, "count": 5},
        ])

    def test_unordered_input_is_sorted(self):
        path = write_gt(self.dir / "gt.csv", [(10, 20, 1), (0, 10, 2)])
        intervals = ec.load_ground_truth(path)
        self.assertEqual(intervals[0]["start"], 0.0)

    def test_touching_intervals_allowed(self):
        path = write_gt(self.dir / "gt.csv", [(0, 10, 1), (10, 20, 1)])
        self.assertEqual(len(ec.load_ground_truth(path)), 2)

    def test_overlapping_intervals_rejected(self):
        path = write_gt(self.dir / "gt.csv", [(0, 12, 1), (10, 20, 1)])
        with self.assertRaises(ec.InputError) as ctx:
            ec.load_ground_truth(path)
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_missing_column_rejected(self):
        path = self.dir / "gt.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(["start_s", "end_s"])
        with self.assertRaises(ec.InputError) as ctx:
            ec.load_ground_truth(path)
        self.assertIn("count", str(ctx.exception))

    def test_invalid_number_rejected(self):
        path = self.dir / "gt.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(["start_s", "end_s", "count"])
            csv.writer(fh).writerow(["zero", "10", "3"])
        with self.assertRaises(ec.InputError):
            ec.load_ground_truth(path)

    def test_non_integer_count_rejected(self):
        path = self.dir / "gt.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(["start_s", "end_s", "count"])
            csv.writer(fh).writerow(["0", "10", "2.5"])
        with self.assertRaises(ec.InputError):
            ec.load_ground_truth(path)

    def test_zero_length_interval_rejected(self):
        path = write_gt(self.dir / "gt.csv", [(5, 5, 1)])
        with self.assertRaises(ec.InputError):
            ec.load_ground_truth(path)

    def test_empty_file_rejected(self):
        path = self.dir / "gt.csv"
        path.write_text("")
        with self.assertRaises(ec.InputError):
            ec.load_ground_truth(path)


class TestGapDetection(Base):
    def test_detects_jump_above_threshold(self):
        samples = uniform_samples(n=10) + uniform_samples(t0=30.0, n=10)
        gaps = ec.detect_gaps(norm(samples), gap_s=10.0)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0][0], 9.0)
        self.assertAlmostEqual(gaps[0][1], 30.0)

    def test_no_gap_at_regular_rate(self):
        samples = norm(uniform_samples(n=30))
        self.assertEqual(ec.detect_gaps(samples, gap_s=10.0), [])

    def test_custom_gap_threshold(self):
        # deltas of 1 s except one 8 s dropout; median delta is 1 s so the
        # adaptive floor is 5*1 = 5 s.
        ts = [float(i) for i in range(8)] + [15.0 + i for i in range(8)]
        samples = [{"t": t, "count": 0, "roi": True} for t in ts]
        self.assertEqual(ec.detect_gaps(samples, gap_s=10.0), [])   # threshold 10
        self.assertEqual(len(ec.detect_gaps(samples, gap_s=3.0)), 1)  # threshold 5


class TestEvaluate(Base):
    def evaluate(self, samples, intervals, **kwargs):
        kwargs.setdefault("min_coverage", 0.5)
        return ec.evaluate(norm(samples), intervals, **kwargs)

    def test_perfect_prediction(self):
        report = self.evaluate(
            uniform_samples(counts=[3]),
            [{"start": 0, "end": 10, "count": 3}, {"start": 10, "end": 20, "count": 3}],
        )
        m = report["metrics"]
        self.assertEqual(m["mae"], 0.0)
        self.assertEqual(m["within_1_accuracy"], 1.0)
        self.assertEqual(m["within_2_accuracy"], 1.0)
        self.assertEqual(m["bias"], 0.0)
        self.assertEqual(m["pairs"], 2)

    def test_constant_over_count(self):
        report = self.evaluate(
            uniform_samples(counts=[5]),
            [{"start": 0, "end": 10, "count": 3}],
        )
        m = report["metrics"]
        self.assertEqual(m["mae"], 2.0)
        self.assertEqual(m["bias"], 2.0)
        self.assertEqual(m["within_1_accuracy"], 0.0)
        self.assertEqual(m["within_2_accuracy"], 1.0)

    def test_constant_under_count(self):
        report = self.evaluate(
            uniform_samples(counts=[1]),
            [{"start": 0, "end": 10, "count": 3}],
        )
        m = report["metrics"]
        self.assertEqual(m["mae"], 2.0)
        self.assertEqual(m["bias"], -2.0)

    def test_mixed_errors_aggregate_correctly(self):
        gt = [(0, 10, 3), (10, 20, 3), (20, 30, 3), (30, 40, 3)]
        counts_by_mid = {5.0: 4, 15.0: 2, 25.0: 3, 35.0: 7}  # errors +1 -1 0 +4
        samples = [sample_row(t, c) for t, c in sorted(counts_by_mid.items())]
        report = self.evaluate(samples, [{"start": s, "end": e, "count": c}
                                         for s, e, c in gt])
        m = report["metrics"]
        self.assertEqual(m["mae"], 1.5)
        self.assertEqual(m["bias"], 1.0)
        self.assertEqual(m["within_1_accuracy"], 0.75)
        self.assertEqual(m["within_2_accuracy"], 0.75)

    def test_error_boundary_inclusive(self):
        gt = [{"start": 0, "end": 10, "count": 3}, {"start": 10, "end": 20, "count": 3}]
        samples = [sample_row(5.0, 4), sample_row(15.0, 5)]  # errors +1 and +2
        report = self.evaluate(samples, gt)
        m = report["metrics"]
        self.assertEqual(m["within_1_accuracy"], 0.5)
        self.assertEqual(m["within_2_accuracy"], 1.0)

    def test_nearest_sample_to_midpoint_tie_prefers_earlier(self):
        samples = [sample_row(4.0, 1), sample_row(6.0, 9)]
        report = self.evaluate(samples, [{"start": 0, "end": 10, "count": 0}])
        pair = report["pairs"][0]
        self.assertEqual(pair["sample_t_rel_s"], 4.0)
        self.assertEqual(pair["pred"], 1)

    def test_match_window_is_interval_plus_tolerance(self):
        # Sample at t=11 lies outside [0,10] but within tolerance 1.0 -> matched.
        samples = [sample_row(11.0, 2)]
        report = self.evaluate(samples, [{"start": 0, "end": 10, "count": 2}])
        self.assertEqual(report["coverage"]["evaluated"], 1)

    def test_sample_outside_window_excluded_as_no_match(self):
        # one matchable pair keeps coverage above the floor; the second
        # interval has its only nearby sample outside the tolerance window
        samples = [sample_row(5.0, 2), sample_row(22.0, 2)]
        report = self.evaluate(
            samples,
            [{"start": 0, "end": 10, "count": 2}, {"start": 10, "end": 20, "count": 2}],
            tolerance_s=1.0,
        )
        self.assertEqual(report["coverage"]["excluded_no_match"], 1)
        self.assertEqual(report["coverage"]["evaluated"], 1)

    def test_missing_interval_counted_not_dropped(self):
        samples = uniform_samples(t0=100.0, n=10)  # covers only the second interval
        report = self.evaluate(
            samples,
            [{"start": 0, "end": 10, "count": 0}, {"start": 100, "end": 110, "count": 0}],
        )
        c = report["coverage"]
        self.assertEqual(c["total_intervals"], 2)
        self.assertEqual(c["evaluated"], 1)
        self.assertEqual(c["excluded_no_match"], 1)
        self.assertAlmostEqual(c["ratio"], 0.5)

    def test_interval_inside_recording_gap_excluded(self):
        samples = uniform_samples(n=10) + uniform_samples(t0=40.0, n=10)
        report = self.evaluate(
            samples,
            [{"start": 14, "end": 16, "count": 0}, {"start": 44, "end": 46, "count": 0}],
        )
        c = report["coverage"]
        self.assertEqual(c["excluded_recording_gap"], 1)
        self.assertEqual(c["evaluated"], 1)
        self.assertEqual(len(report["gaps"]), 1)

    def test_unconfigured_roi_sample_excluded(self):
        samples = [sample_row(5.0, 2), sample_row(15.0, 2, roi=False)]
        report = self.evaluate(
            samples,
            [{"start": 0, "end": 10, "count": 2}, {"start": 10, "end": 20, "count": 2}],
        )
        self.assertEqual(report["coverage"]["excluded_roi_unconfigured"], 1)
        self.assertEqual(report["metrics"]["pairs"], 1)

    def test_coverage_below_minimum_raises(self):
        samples = uniform_samples(n=10)
        intervals = [
            {"start": 0, "end": 10, "count": 0},
            {"start": 100, "end": 110, "count": 0},
        ]
        with self.assertRaises(ec.InsufficientDataError) as ctx:
            ec.evaluate(norm(samples), intervals, tolerance_s=1.0, gap_s=10.0,
                        min_coverage=0.8)
        self.assertIn("80%", str(ctx.exception))

    def test_zero_overlap_raises_insufficient_data(self):
        samples = norm(uniform_samples(n=10))  # t = 0..9
        intervals = [{"start": 100, "end": 110, "count": 1}]
        with self.assertRaises(ec.InsufficientDataError):
            ec.evaluate(samples, intervals, tolerance_s=1.0, gap_s=10.0,
                        min_coverage=0.8)

    def test_invalid_parameters_raise_input_error(self):
        samples = uniform_samples(n=5)
        intervals = [{"start": 0, "end": 10, "count": 0}]
        for kwargs in (
            {"tolerance_s": -1.0},
            {"gap_s": 0.0},
            {"min_coverage": 0.0},
            {"min_coverage": 1.5},
        ):
            with self.assertRaises(ec.InputError):
                ec.evaluate(samples, intervals, **kwargs)


class TestMain(Base):
    def run_main(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ec.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_success_and_json_report(self):
        samples_path = write_samples(self.dir / "samples.jsonl",
                                     uniform_samples(counts=[3], n=25))
        gt_path = write_gt(self.dir / "gt.csv",
                           [(0, 10, 3), (10, 20, 3), (20, 30, 3)])
        json_path = self.dir / "report" / "eval.json"
        code, out, _ = self.run_main([
            "--samples", str(samples_path),
            "--ground-truth", str(gt_path),
            "--json", str(json_path),
        ])
        self.assertEqual(code, ec.EXIT_OK)
        self.assertIn("MAE", out)
        self.assertTrue(json_path.exists())
        payload = json.loads(json_path.read_text())
        self.assertEqual(payload["metrics"]["mae"], 0.0)
        self.assertEqual(payload["parameters"]["timeline"], "t_rel_s")

    def test_insufficient_data_exit_code_1(self):
        samples_path = write_samples(self.dir / "samples.jsonl",
                                     uniform_samples(n=10))
        gt_path = write_gt(self.dir / "gt.csv", [(500, 510, 2)])  # no overlap
        code, _, err = self.run_main([
            "--samples", str(samples_path), "--ground-truth", str(gt_path),
        ])
        self.assertEqual(code, ec.EXIT_INSUFFICIENT_DATA)
        self.assertIn("INSUFFICIENT DATA", err)

    def test_malformed_input_exit_code_2(self):
        bad = self.dir / "bad.jsonl"
        bad.write_text("not json\n")
        gt_path = write_gt(self.dir / "gt.csv", [(0, 10, 1)])
        code, _, err = self.run_main([
            "--samples", str(bad), "--ground-truth", str(gt_path),
        ])
        self.assertEqual(code, ec.EXIT_INPUT_ERROR)
        self.assertIn("[ERROR]", err)

    def test_all_unconfigured_samples_exit_code_2(self):
        samples_path = write_samples(self.dir / "samples.jsonl",
                                     uniform_samples(n=10, roi=False))
        gt_path = write_gt(self.dir / "gt.csv", [(0, 10, 1)])
        code, _, err = self.run_main([
            "--samples", str(samples_path), "--ground-truth", str(gt_path),
        ])
        self.assertEqual(code, ec.EXIT_INPUT_ERROR)
        self.assertIn("roi_configured=false", err)


if __name__ == "__main__":
    unittest.main()
