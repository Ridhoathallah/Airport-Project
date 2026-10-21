"""Tests for the ML dataset engineering toolkit.

Everything here runs against temporary directories and synthetic fixtures:
no test ever touches ``models/``, ``datasets/`` or any real imagery, and
no test needs to download anything.

The tests are written as ``unittest.TestCase`` classes so that they are
collected by both runners used in this project:

* ``make test``  -> ``python -m unittest discover -s tests``
* ``pytest tests/``
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog  # noqa: E402
from ml.contract import (  # noqa: E402
    AnnotationStatus,
    ClassDef,
    DEFAULT_CLASS_SCHEMA,
    ImageRecord,
    canonical_json,
    class_schema_to_dict,
    make_annotation_id,
    make_image_id,
    sha256_file,
    sha256_json,
    utc_now_iso,
    validate_class_schema,
)
from ml.evaluate import (  # noqa: E402
    DEFAULT_ACCEPTANCE_CRITERIA,
    EvaluationReport,
    MetricSet,
    compare,
    parse_results_csv,
    parse_val_stdout,
)
from ml.experiment import (  # noqa: E402
    ExperimentConfig,
    TrainingRun,
    create_experiment_id,
    find_checkpoints,
    hash_checkpoint,
)
from ml.manifest import (  # noqa: E402
    Manifest,
    ManifestError,
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)
from ml.split import SplitPolicy, check_leakage, plan_splits  # noqa: E402
from ml.validate import (  # noqa: E402
    Severity,
    parse_label_file,
    perceptual_hash,
    validate_dataset,
)


# ─────────────────────────────────────────────────────────────
#  FIXTURE HELPERS
# ─────────────────────────────────────────────────────────────


def make_image(path, width: int = 64, height: int = 48, colour=(120, 130, 140)) -> Path:
    """Write a small synthetic image."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), colour).save(path)
    return path


def write_label(path, rows) -> Path:
    """Write a YOLO label file from ``(class_id, xc, yc, w, h)`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join("{} {} {} {} {}\n".format(*row) for row in rows)
    path.write_text(body, encoding="utf-8")
    return path


def make_record(tmp: Path, name: str, group=None, digest=None, phash=None) -> ImageRecord:
    return ImageRecord(
        image_id=make_image_id(digest or f"{name}-digest"),
        relpath=f"images/train/{name}.jpg",
        sha256=digest or f"{name}-digest",
        width=64,
        height=48,
        source_group=group,
        phash=phash,
    )


class TempDirTestCase(unittest.TestCase):
    """Base class providing a throwaway directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlkit-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  CONTRACT
# ─────────────────────────────────────────────────────────────


class TestContract(unittest.TestCase):
    def test_canonical_json_is_order_independent(self):
        self.assertEqual(canonical_json({"a": 1, "b": 2}), canonical_json({"b": 2, "a": 1}))

    def test_json_hash_changes_with_content(self):
        self.assertNotEqual(sha256_json({"a": 1}), sha256_json({"a": 2}))

    def test_image_id_is_content_addressed(self):
        self.assertEqual(make_image_id("a" * 64), make_image_id("a" * 64))
        self.assertNotEqual(make_image_id("a" * 64), make_image_id("b" * 64))

    def test_annotation_id_is_stable_and_box_sensitive(self):
        first = make_annotation_id("img_1", 0, [0.5, 0.5, 0.2, 0.2])
        same = make_annotation_id("img_1", 0, [0.5, 0.5, 0.2, 0.2])
        other = make_annotation_id("img_1", 0, [0.4, 0.5, 0.2, 0.2])
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)

    def test_utc_timestamp_format(self):
        stamp = utc_now_iso()
        self.assertTrue(stamp.endswith("Z"), stamp)
        self.assertIn("T", stamp)

    def test_class_schema_validation(self):
        self.assertEqual(validate_class_schema(DEFAULT_CLASS_SCHEMA), [])
        self.assertTrue(validate_class_schema([]))
        self.assertTrue(validate_class_schema((ClassDef(0, "person"), ClassDef(0, "other"))))

    def test_class_schema_round_trip(self):
        payload = class_schema_to_dict(DEFAULT_CLASS_SCHEMA)
        self.assertEqual(payload["classes"], [{"id": 0, "name": "person"}])

    def test_machine_proposed_is_not_ground_truth(self):
        self.assertFalse(AnnotationStatus.MACHINE_PROPOSED.is_ground_truth)
        self.assertFalse(AnnotationStatus.UNREVIEWED.is_ground_truth)
        self.assertTrue(AnnotationStatus.VERIFIED.is_ground_truth)
        self.assertTrue(AnnotationStatus.NEGATIVE_REVIEWED.is_ground_truth)

    def test_signed64_round_trips_the_bit_pattern(self):
        from ml.contract import to_signed64

        self.assertEqual(to_signed64(0), 0)
        self.assertEqual(to_signed64((1 << 64) - 1), -1)
        self.assertEqual(to_signed64(1 << 63), -(1 << 63))
        self.assertEqual(to_signed64(5), 5)
        for value in (0, 1, 1 << 62, 1 << 63, (1 << 64) - 1):
            self.assertEqual(to_signed64(value) & ((1 << 64) - 1), value)


# ─────────────────────────────────────────────────────────────
#  ANNOTATION PARSING AND VALIDATION
# ─────────────────────────────────────────────────────────────


class TestLabelParsing(TempDirTestCase):
    def test_valid_label_parses(self):
        path = write_label(self.tmp / "a.txt", [(0, 0.5, 0.5, 0.2, 0.4)])
        boxes, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(findings, [])
        self.assertAlmostEqual(boxes[0].x_center, 0.5)
        self.assertAlmostEqual(boxes[0].area(), 0.08)

    def test_class_id_out_of_range_is_an_error(self):
        path = write_label(self.tmp / "a.txt", [(7, 0.5, 0.5, 0.2, 0.2)])
        boxes, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(boxes, [])
        self.assertEqual(findings[0].code, "label_class_out_of_range")
        self.assertIs(findings[0].severity, Severity.ERROR)

    def test_coordinates_out_of_bounds_are_an_error(self):
        path = write_label(self.tmp / "a.txt", [(0, 1.4, 0.5, 0.2, 0.2)])
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_coords_out_of_bounds")

    def test_zero_area_box_is_an_error(self):
        path = write_label(self.tmp / "a.txt", [(0, 0.5, 0.5, 0.0, 0.2)])
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_zero_area_box")

    def test_malformed_row_is_reported(self):
        path = self.tmp / "a.txt"
        path.write_text("0 0.5 0.5\n", encoding="utf-8")
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_malformed_row")

    def test_non_numeric_row_is_reported(self):
        path = self.tmp / "a.txt"
        path.write_text("0 abc 0.5 0.2 0.2\n", encoding="utf-8")
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_non_numeric")

    def test_non_integer_class_is_reported(self):
        path = self.tmp / "a.txt"
        path.write_text("0.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_non_integer_class")

    def test_box_outside_image_is_reported(self):
        path = write_label(self.tmp / "a.txt", [(0, 0.05, 0.5, 0.4, 0.2)])
        _, findings = parse_label_file(path, DEFAULT_CLASS_SCHEMA)
        self.assertEqual(findings[0].code, "label_box_outside_image")


class TestDatasetValidation(TempDirTestCase):
    def _dataset(self) -> Path:
        root = self.tmp / "ds"
        make_image(root / "images" / "train" / "a.jpg")
        write_label(root / "labels" / "train" / "a.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        return root

    def test_valid_dataset_passes(self):
        root = self._dataset()
        report, records = validate_dataset(root)
        self.assertTrue(report.ok, report.render_text())
        self.assertEqual(len(records), 1)
        self.assertEqual(report.counts["images"], 1)

    def test_missing_label_is_an_error(self):
        root = self._dataset()
        make_image(root / "images" / "train" / "orphan.jpg")
        report, _ = validate_dataset(root)
        self.assertFalse(report.ok)
        self.assertIn("missing_label", [f.code for f in report.findings])

    def test_missing_image_is_an_error(self):
        root = self._dataset()
        write_label(root / "labels" / "train" / "ghost.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        report, _ = validate_dataset(root)
        self.assertFalse(report.ok)
        self.assertIn("missing_image", [f.code for f in report.findings])

    def test_corrupt_image_is_an_error(self):
        root = self._dataset()
        (root / "images" / "train" / "broken.jpg").write_bytes(b"not an image")
        write_label(root / "labels" / "train" / "broken.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        report, _ = validate_dataset(root)
        self.assertFalse(report.ok)
        self.assertIn("image_unreadable", [f.code for f in report.findings])

    def test_empty_label_is_not_an_error_but_is_flagged(self):
        root = self._dataset()
        (root / "labels" / "train" / "a.txt").write_text("", encoding="utf-8")
        report, _ = validate_dataset(root)
        self.assertTrue(report.ok, report.render_text())
        self.assertIn("empty_label_unclassified", [f.code for f in report.findings])

    def test_empty_label_declared_as_reviewed_negative_is_clean(self):
        root = self._dataset()
        (root / "labels" / "train" / "a.txt").write_text("", encoding="utf-8")
        report, _ = validate_dataset(
            root, annotation_status=AnnotationStatus.NEGATIVE_REVIEWED.value
        )
        self.assertTrue(report.ok, report.render_text())
        codes = [f.code for f in report.findings]
        self.assertIn("empty_label_reviewed_negative", codes)
        self.assertNotIn("empty_label_unclassified", codes)

    def test_duplicate_images_are_detected(self):
        root = self._dataset()
        shutil.copy(root / "images" / "train" / "a.jpg", root / "images" / "train" / "b.jpg")
        write_label(root / "labels" / "train" / "b.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        report, _ = validate_dataset(root)
        self.assertFalse(report.ok)
        self.assertIn("duplicate_image", [f.code for f in report.findings])

    def test_report_is_serialisable(self):
        root = self._dataset()
        report, _ = validate_dataset(root)
        payload = report.to_dict()
        json.dumps(payload)
        self.assertIn("summary", payload)
        self.assertIn("findings", payload)
        self.assertIn("DATASET", report.render_text().upper())

    def test_perceptual_hash_is_stable_for_identical_images(self):
        first = make_image(self.tmp / "one.png", colour=(10, 20, 30))
        second = self.tmp / "two.png"
        shutil.copy(first, second)
        self.assertEqual(perceptual_hash(first), perceptual_hash(second))

    def test_inconsistent_dimensions_are_reported(self):
        root = self._dataset()
        make_image(root / "images" / "train" / "wide.jpg", width=200, height=50)
        write_label(root / "labels" / "train" / "wide.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        report, _ = validate_dataset(root)
        self.assertIn("inconsistent_dimensions", [f.code for f in report.findings])


# ─────────────────────────────────────────────────────────────
#  SPLITTING
# ─────────────────────────────────────────────────────────────


class TestSplitting(unittest.TestCase):
    def _records(self, groups: int = 6, per_group: int = 2):
        records = []
        for g in range(groups):
            for i in range(per_group):
                records.append(make_record(Path("."), f"g{g}_{i}", group=f"group{g}"))
        return records

    def test_split_is_deterministic_for_a_seed(self):
        records = self._records()
        policy = SplitPolicy(seed=42, ratios={"train": 0.7, "val": 0.2, "test": 0.1})
        first = plan_splits(records, policy)
        second = plan_splits(records, policy)
        self.assertEqual(first.assignments, second.assignments)

    def test_different_seeds_can_differ(self):
        records = self._records(groups=10, per_group=1)
        a = plan_splits(records, SplitPolicy(seed=1, ratios={"train": 0.7, "val": 0.2, "test": 0.1}))
        b = plan_splits(records, SplitPolicy(seed=2, ratios={"train": 0.7, "val": 0.2, "test": 0.1}))
        self.assertNotEqual(a.assignments, b.assignments)

    def test_groups_are_never_split_across_partitions(self):
        records = self._records()
        result = plan_splits(records, SplitPolicy(seed=7))
        by_group: dict = {}
        for record in records:
            split = result.assignments[record.image_id]
            by_group.setdefault(record.source_group, set()).add(split)
        for group, splits in by_group.items():
            self.assertEqual(len(splits), 1, f"group {group} leaked across {splits}")

    def test_leakage_checker_detects_a_group_that_was_split(self):
        records = [make_record(Path("."), "a", group="g"), make_record(Path("."), "b", group="g")]
        findings = check_leakage(records, {"img_": "train"} | {
            records[0].image_id: "train", records[1].image_id: "val"
        })
        self.assertIn("group_leakage", [f.code for f in findings])

    def test_leakage_checker_detects_duplicate_content_across_splits(self):
        # Two nominally different images (distinct ids) that carry identical
        # bytes -- e.g. a manifest edited by hand, or the same picture
        # re-registered under a second name.
        digest = "f" * 64
        records = [
            ImageRecord(image_id="img_a", relpath="images/train/a.jpg", sha256=digest),
            ImageRecord(image_id="img_b", relpath="images/val/b.jpg", sha256=digest),
        ]
        findings = check_leakage(records, {"img_a": "train", "img_b": "val"})
        self.assertIn("duplicate_leakage", [f.code for f in findings])

    def test_orphan_images_are_reported(self):
        records = [make_record(Path("."), "a", group=None)]
        result = plan_splits(records, SplitPolicy(seed=1))
        self.assertIn("orphan_images", [f.code for f in result.findings])

    def test_orphan_policy_error_blocks_the_split(self):
        records = [make_record(Path("."), "a", group=None)]
        result = plan_splits(records, SplitPolicy(seed=1, orphan_policy="error"))
        self.assertFalse(result.ok)

    def test_all_images_are_assigned(self):
        records = self._records()
        result = plan_splits(records, SplitPolicy(seed=3))
        self.assertEqual(len(result.assignments), len(records))

    def test_split_counts_sum_to_the_total(self):
        records = self._records(groups=5, per_group=3)
        result = plan_splits(records, SplitPolicy(seed=3))
        splits = result.splits()
        self.assertEqual(sum(len(v) for v in splits.values()), len(records))


# ─────────────────────────────────────────────────────────────
#  MANIFEST
# ─────────────────────────────────────────────────────────────


class TestManifest(TempDirTestCase):
    def _manifest(self) -> Manifest:
        records = [make_record(self.tmp, "a", group="g1"), make_record(self.tmp, "b", group="g2")]
        return build_manifest(
            dataset_version="people_test",
            images=records,
            splits={"train": [records[0].image_id], "val": [], "test": [records[1].image_id]},
            class_schema=DEFAULT_CLASS_SCHEMA,
            seed=123,
            split_policy={"seed": 123},
            root="ds",
        )

    def test_manifest_hash_is_stable(self):
        first = self._manifest()
        second = self._manifest()
        self.assertEqual(first.compute_sha256(), second.compute_sha256())
        self.assertEqual(first.dataset_version_id, second.dataset_version_id)

    def test_creation_timestamp_does_not_change_the_hash(self):
        first = self._manifest()
        second = self._manifest()
        second.created_at_utc = "2099-01-01T00:00:00.000Z"
        self.assertEqual(first.compute_sha256(), second.compute_sha256())
        # ... but the timestamp is still recorded in the written manifest
        self.assertIn("created_at_utc", second.to_dict())

    def test_manifest_hash_changes_when_content_changes(self):
        first = self._manifest()
        second = self._manifest()
        second.images[0].sha256 = "0" * 64
        self.assertNotEqual(first.compute_sha256(), second.compute_sha256())

    def test_write_and_load_round_trip(self):
        manifest = self._manifest()
        path = write_manifest(self.tmp / "manifests" / "people_test.json", manifest)
        loaded = load_manifest(path)
        self.assertEqual(loaded.dataset_version, manifest.dataset_version)
        self.assertEqual(loaded.manifest_sha256, manifest.manifest_sha256)
        self.assertEqual(len(loaded.images), 2)

    def test_verify_detects_tampering(self):
        manifest = self._manifest()
        path = write_manifest(self.tmp / "manifests" / "people_test.json", manifest)
        ok, problems = verify_manifest(path, check_files=False)
        self.assertTrue(ok, problems)

        data = json.loads(path.read_text())
        data["seed"] = 999
        path.write_text(json.dumps(data), encoding="utf-8")
        ok, problems = verify_manifest(path, check_files=False)
        self.assertFalse(ok)
        self.assertTrue(any("hash mismatch" in p for p in problems))

    def test_verify_detects_a_modified_image_file(self):
        root = self.tmp / "ds"
        image = make_image(root / "images" / "train" / "a.jpg")
        record = ImageRecord(
            image_id=make_image_id(sha256_file(image)),
            relpath="images/train/a.jpg",
            sha256=sha256_file(image),
        )
        manifest = build_manifest(
            dataset_version="people_test",
            images=[record],
            splits={"train": [record.image_id], "val": [], "test": []},
            class_schema=DEFAULT_CLASS_SCHEMA,
            root="../ds",          # relative to the directory holding the manifest
        )
        path = write_manifest(self.tmp / "manifests" / "people_test.json", manifest)
        make_image(image, width=70, height=48, colour=(1, 2, 3))
        ok, problems = verify_manifest(path, check_files=True)
        self.assertFalse(ok)
        self.assertTrue(any("hash mismatch" in p for p in problems))

    def test_verify_reports_a_missing_image_file(self):
        root = self.tmp / "ds"
        image = make_image(root / "images" / "train" / "a.jpg")
        record = ImageRecord(
            image_id=make_image_id(sha256_file(image)),
            relpath="images/train/a.jpg",
            sha256=sha256_file(image),
        )
        manifest = build_manifest(
            dataset_version="people_test",
            images=[record],
            splits={"train": [record.image_id], "val": [], "test": []},
            class_schema=DEFAULT_CLASS_SCHEMA,
            root="../ds",
        )
        path = write_manifest(self.tmp / "manifests" / "people_test.json", manifest)
        image.unlink()
        ok, problems = verify_manifest(path, check_files=True)
        self.assertFalse(ok)
        self.assertTrue(any("missing image file" in p for p in problems))

    def test_loading_a_broken_manifest_raises(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ManifestError):
            load_manifest(bad)


# ─────────────────────────────────────────────────────────────
#  CATALOG
# ─────────────────────────────────────────────────────────────


class TestCatalog(TempDirTestCase):
    def test_fresh_catalog_has_the_expected_schema_version(self):
        with Catalog(self.tmp / "cat.db") as catalog:
            self.assertEqual(catalog.version, 1)

    def test_reopening_a_catalog_is_idempotent(self):
        path = self.tmp / "cat.db"
        with Catalog(path) as catalog:
            catalog.migrate()
        with Catalog(path) as catalog:
            self.assertEqual(catalog.version, 1)

    def test_dataset_registration_is_idempotent(self):
        with Catalog(self.tmp / "cat.db") as catalog:
            for _ in range(3):
                catalog.register_dataset_version(
                    dataset_version_id="dsv_1",
                    name="people_v1",
                    contract_version="1.0.0",
                    class_schema_version="1.0.0",
                    manifest_path="datasets/manifests/people_v1.json",
                    manifest_sha256="a" * 64,
                    seed=1,
                )
            self.assertEqual(len(catalog.list_dataset_versions()), 1)

    def test_images_and_splits_are_registered(self):
        records = [make_record(self.tmp, "a", group="g1")]
        with Catalog(self.tmp / "cat.db") as catalog:
            catalog.register_dataset_version(
                dataset_version_id="dsv_1", name="v1", contract_version="1.0.0",
                class_schema_version="1.0.0", manifest_path="m.json",
                manifest_sha256="a" * 64,
            )
            catalog.register_images("dsv_1", records)
            catalog.register_splits("dsv_1", {records[0].image_id: "train"},
                                    {records[0].image_id: records[0]})
            self.assertEqual(len(catalog.list_images("dsv_1")), 1)
            self.assertEqual(catalog.stats()["split_assignment"], 1)

    def test_experiment_and_artifact_registration(self):
        with Catalog(self.tmp / "cat.db") as catalog:
            catalog.register_experiment(
                experiment_id="exp_1", dataset_version_id=None, manifest_sha256=None,
                base_checkpoint="models/yolov5s.pt", base_checkpoint_sha256="b" * 64,
                config_sha256="c" * 64, resolved_config={"epochs": 1}, status="completed",
            )
            catalog.register_artifact(
                artifact_id="art_1", experiment_id="exp_1", role="best",
                path="runs/train/x/weights/best.pt", sha256="d" * 64, size_bytes=10,
            )
            catalog.record_metric(
                metric_id="met_1", experiment_id="exp_1", metric_name="map_50", value=0.5,
                split="test",
            )
            self.assertEqual(len(catalog.list_experiments()), 1)
            self.assertEqual(len(catalog.list_artifacts("exp_1")), 1)
            self.assertEqual(len(catalog.list_metrics("exp_1")), 1)

    def test_verify_dataset_reports_missing_files(self):
        with Catalog(self.tmp / "cat.db") as catalog:
            catalog.register_dataset_version(
                dataset_version_id="dsv_1", name="v1", contract_version="1.0.0",
                class_schema_version="1.0.0", manifest_path=str(self.tmp / "m.json"),
                manifest_sha256="a" * 64,
            )
            record = make_record(self.tmp, "a")
            record.relpath = "images/train/absent.jpg"
            catalog.register_images("dsv_1", [record])
            problems = catalog.verify_dataset("dsv_1")
            self.assertTrue(problems)

    def test_images_with_a_high_perceptual_hash_can_be_stored(self):
        record = make_record(self.tmp, "a")
        record.phash = (1 << 64) - 1          # exceeds SQLite's signed INTEGER range
        with Catalog(self.tmp / "cat.db") as catalog:
            catalog.register_dataset_version(
                dataset_version_id="dsv_1", name="v1", contract_version="1.0.0",
                class_schema_version="1.0.0", manifest_path="m.json",
                manifest_sha256="a" * 64,
            )
            catalog.register_images("dsv_1", [record])
            self.assertEqual(len(catalog.list_images("dsv_1")), 1)

    def test_catalog_is_separate_from_the_runtime_database(self):
        self.assertTrue(Path("datasets/metadata/dataset_catalog.db").as_posix().endswith(
            "datasets/metadata/dataset_catalog.db"))


# ─────────────────────────────────────────────────────────────
#  EXPERIMENTS
# ─────────────────────────────────────────────────────────────


class TestExperiment(unittest.TestCase):
    def test_config_hash_is_stable_and_content_sensitive(self):
        first = ExperimentConfig(data="d.yaml", weights="w.pt", epochs=10)
        second = ExperimentConfig(data="d.yaml", weights="w.pt", epochs=10)
        third = ExperimentConfig(data="d.yaml", weights="w.pt", epochs=11)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, third.sha256)

    def test_resolved_config_is_json_serialisable(self):
        config = ExperimentConfig(data="d.yaml", weights="w.pt")
        json.dumps(config.to_dict())

    def test_command_targets_the_vendored_train_script(self):
        config = ExperimentConfig(data="d.yaml", weights="w.pt")
        command = config.build_command()
        self.assertTrue(command[1].endswith("train.py"))
        self.assertIn("--data", command)
        self.assertIn("--weights", command)
        self.assertIn("--single-cls", command)

    def test_command_contains_only_flags_the_vendored_trainer_accepts(self):
        config = ExperimentConfig(data="d.yaml", weights="w.pt")
        command = config.build_command()
        train_py = Path(__file__).resolve().parent.parent / "yolov5" / "train.py"
        if train_py.is_file():
            source = train_py.read_text(encoding="utf-8")
            for flag in command:
                if flag.startswith("--"):
                    self.assertIn(f'"{flag}"', source,
                                  f"vendored train.py does not accept {flag}")

    def test_experiment_id_is_stable_for_the_same_config(self):
        config = ExperimentConfig(data="d.yaml", weights="w.pt")
        first = create_experiment_id(config, "2026-10-20T00:00:00.000Z")
        second = create_experiment_id(config, "2026-10-20T00:00:00.000Z")
        self.assertEqual(first, second)

    def test_checkpoint_hashing_detects_a_missing_file(self):
        self.assertIsNone(hash_checkpoint("/nonexistent/checkpoint.pt"))

    def test_find_checkpoints_locates_best_and_last(self):
        run = Path(tempfile.mkdtemp(prefix="mlkit-run-"))
        self.addCleanup(shutil.rmtree, run, ignore_errors=True)
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"best")
        (run / "weights" / "last.pt").write_bytes(b"last")
        found = find_checkpoints(run)
        self.assertEqual(set(found), {"best", "last"})

    def test_incomplete_run_is_recorded_as_failed(self):
        run = TrainingRun(experiment_id="exp_x", status="failed", error="boom")
        payload = run.to_dict()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "boom")

    def test_original_checkpoint_is_not_modified(self):
        checkpoint = Path("models/yolov5s.pt")
        if not checkpoint.is_file():
            self.skipTest("models/yolov5s.pt is not present in this checkout")
        before = sha256_file(checkpoint)
        ExperimentConfig(data="d.yaml", weights=str(checkpoint)).build_command()
        after = sha256_file(checkpoint)
        self.assertEqual(before, after)


# ─────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────


class TestEvaluation(TempDirTestCase):
    def _results_csv(self) -> Path:
        path = self.tmp / "results.csv"
        header = ("epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,"
                  "metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss")
        path.write_text(
            header + "\n"
            "0,1.0,1.0,1.0,0.5000,0.4000,0.4500,0.2500,1.0\n"
            "1,0.9,0.9,0.9,0.7000,0.6000,0.6500,0.3500,0.9\n",
            encoding="utf-8",
        )
        return path

    def test_results_csv_is_parsed(self):
        metrics = parse_results_csv(self._results_csv())
        self.assertAlmostEqual(metrics.precision, 0.7)
        self.assertAlmostEqual(metrics.map_50, 0.65)
        self.assertAlmostEqual(metrics.map_50_95, 0.35)

    def test_val_stdout_is_parsed(self):
        text = ("Class Names (only 'person' is used):\n"
                "                   all        10        25      0.8000      0.7000      0.7500      0.4000\n")
        metrics = parse_val_stdout(text)
        self.assertAlmostEqual(metrics.precision, 0.8)
        self.assertAlmostEqual(metrics.recall, 0.7)
        self.assertEqual(metrics.images, 10)

    def test_empty_stdout_yields_no_metrics(self):
        metrics = parse_val_stdout("nothing here")
        self.assertIsNone(metrics.map_50)

    def test_report_serialisation_round_trip(self):
        report = EvaluationReport(
            model_path="best.pt", model_sha256="a" * 64, split="test",
            metrics=MetricSet(precision=0.8, recall=0.7, map_50=0.75, map_50_95=0.4),
        )
        payload = report.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["metrics"]["map_50"], 0.75)
        self.assertIn("mAP@0.5", report.render_text())

    def test_comparison_reports_deltas(self):
        baseline = EvaluationReport(model_path="base.pt", split="test",
                                    metrics=MetricSet(precision=0.7, recall=0.6,
                                                      map_50=0.6, map_50_95=0.3))
        candidate = EvaluationReport(model_path="cand.pt", split="test",
                                     metrics=MetricSet(precision=0.8, recall=0.7,
                                                       map_50=0.7, map_50_95=0.35))
        comparison = compare(baseline, candidate)
        self.assertAlmostEqual(comparison.deltas["map_50"], 0.1, places=6)
        self.assertTrue(comparison.passed)

    def test_comparison_rejects_a_candidate_that_regresses(self):
        baseline = EvaluationReport(model_path="base.pt", split="test",
                                    metrics=MetricSet(precision=0.9, recall=0.9,
                                                      map_50=0.8, map_50_95=0.5))
        candidate = EvaluationReport(model_path="cand.pt", split="test",
                                     metrics=MetricSet(precision=0.5, recall=0.4,
                                                       map_50=0.4, map_50_95=0.2))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.passed)

    def test_comparison_is_not_comparable_across_dataset_versions(self):
        baseline = EvaluationReport(model_path="base.pt", split="test",
                                    dataset_version_id="dsv_1",
                                    metrics=MetricSet(map_50=0.6, map_50_95=0.3,
                                                      precision=0.7, recall=0.6))
        candidate = EvaluationReport(model_path="cand.pt", split="test",
                                     dataset_version_id="dsv_2",
                                     metrics=MetricSet(map_50=0.9, map_50_95=0.5,
                                                       precision=0.9, recall=0.8))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.comparable)
        self.assertFalse(comparison.passed)

    def test_acceptance_criteria_are_labelled_provisional(self):
        self.assertTrue(DEFAULT_ACCEPTANCE_CRITERIA["provisional"])


if __name__ == "__main__":
    unittest.main()
