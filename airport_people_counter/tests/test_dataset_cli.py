"""Tests for the dataset CLIs, the catalog lifecycle and the training preflight.

These tests exercise the real command-line entry points end to end, but
strictly inside temporary directories: no test touches the repository's
``datasets/``, ``runs/`` or metadata catalog, and no test modifies
``models/yolov5s.pt``.

The tests are ``unittest.TestCase`` classes so that both runners used in
this project collect them:

* ``make test``  -> ``python -m unittest discover -s tests``
* ``pytest tests/``
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog  # noqa: E402
from ml.manifest import STATUS_EMPTY, STATUS_READY  # noqa: E402
from ml.preflight import run_preflight  # noqa: E402


def load_script(name: str):
    """Import a CLI script by path under a unique module name."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_mlkit_cli_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_image(path, width: int = 64, height: int = 48, colour=(120, 130, 140),
               seed: int | None = None) -> Path:
    """Write a small synthetic image.

    When ``seed`` is given the image gets a deterministic noise pattern, so
    two images with different seeds differ both in their bytes and in
    their perceptual hash. (Solid colours are not enough: JPEG
    quantisation makes two nearby colours produce identical files, which
    the duplicate detector would then correctly flag.)
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if seed is None:
        Image.new("RGB", (width, height), colour).save(path)
        return path
    import random

    rng = random.Random(seed)
    pixels = bytes(rng.randrange(256) for _ in range(width * height * 3))
    image = Image.frombytes("RGB", (width, height), pixels)
    image.save(path)
    return path


def write_label(path, rows) -> Path:
    """Write a YOLO label file from ``(class_id, xc, yc, w, h)`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("{} {} {} {} {}\n".format(*row) for row in rows), encoding="utf-8")
    return path


def make_dataset(root: Path, per_split: int = 3, group: str = "session_a") -> Path:
    """Create a small but valid single-class dataset with source groups.

    Each image gets its own deterministic noise pattern, so no two images
    are byte-identical and no two are perceptual near-duplicates.
    """
    seed = 1000
    for index, split in enumerate(("train", "val", "test")):
        group_dir = f"{group}_{split}"
        for i in range(per_split):
            seed += 1
            name = f"{split}_{i:03d}"
            make_image(root / "images" / split / group_dir / f"{name}.jpg", seed=seed)
            write_label(root / "labels" / split / group_dir / f"{name}.txt",
                        [(0, 0.5, 0.5, 0.3, 0.4)])
    return root


class DatasetWorkspace:
    """A throwaway workspace with its own dataset, manifests and catalog."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="mlkit-cli-"))
        self.dataset = self.root / "dataset"
        self.manifests = self.root / "manifests"
        self.reports = self.root / "reports"
        self.catalog = self.root / "catalog.db"
        for directory in (self.dataset, self.manifests, self.reports):
            directory.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def build_argv(self, name="v1", root=None, extra=None):
        argv = [
            "--root", str(root or self.dataset),
            "--name", name,
            "--manifest-dir", str(self.manifests),
            "--report-dir", str(self.reports),
            "--catalog", str(self.catalog),
        ]
        return argv + list(extra or [])


# ─────────────────────────────────────────────────────────────
#  CATALOG INITIALISATION AND MIGRATIONS
# ─────────────────────────────────────────────────────────────


class TestCatalogLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlkit-catalog-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.catalog_path = self.tmp / "nested" / "dir" / "catalog.db"
        self.cli = load_script("catalog")

    def test_init_creates_the_catalog_and_its_parent_directory(self):
        self.assertFalse(self.catalog_path.parent.exists())
        code = self.cli.main(["--catalog", str(self.catalog_path), "init"])
        self.assertEqual(code, 0)
        self.assertTrue(self.catalog_path.is_file())
        with Catalog(self.catalog_path) as catalog:
            self.assertEqual(catalog.version, 1)

    def test_init_is_idempotent(self):
        for _ in range(3):
            self.assertEqual(self.cli.main(["--catalog", str(self.catalog_path), "init"]), 0)
        with Catalog(self.catalog_path) as catalog:
            self.assertEqual(catalog.version, 1)
            self.assertEqual(catalog.stats()["dataset_version"], 0)

    def test_init_preserves_existing_records(self):
        self.cli.main(["--catalog", str(self.catalog_path), "init"])
        with Catalog(self.catalog_path) as catalog:
            catalog.register_dataset_version(
                dataset_version_id="dsv_keep", name="keep", contract_version="1.0.0",
                class_schema_version="1.0.0", manifest_path="m.json",
                manifest_sha256="a" * 64,
            )
        self.cli.main(["--catalog", str(self.catalog_path), "init"])
        with Catalog(self.catalog_path) as catalog:
            self.assertEqual(len(catalog.list_dataset_versions()), 1)
            self.assertEqual(catalog.list_dataset_versions()[0]["name"], "keep")

    def test_migration_is_forward_only_and_reports_its_version(self):
        with Catalog(self.catalog_path) as catalog:
            self.assertEqual(catalog.migrate(), 1)
            self.assertEqual(catalog.version, 1)
            # running it again is a no-op
            self.assertEqual(catalog.migrate(), 1)

    def test_migration_applies_to_an_older_database(self):
        with Catalog(self.catalog_path) as catalog:
            catalog.migrate()
        # simulate a database written by an older version of the tooling
        conn = sqlite3.connect(str(self.catalog_path))
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
        conn.close()
        with Catalog(self.catalog_path) as catalog:
            self.assertEqual(catalog.migrate(), 1)
            self.assertEqual(catalog.version, 1)

    def test_inspection_on_a_missing_catalog_reports_clearly(self):
        for command in ("stats", "datasets", "experiments", "artifacts", "metrics"):
            code = self.cli.main(["--catalog", str(self.catalog_path), command])
            self.assertEqual(code, 3, f"{command} should report a missing catalog")

    def test_inspection_on_a_missing_catalog_explains_how_to_fix_it(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.cli.main(["--catalog", str(self.catalog_path), "stats"])
        output = buffer.getvalue()
        self.assertIn("catalog.py init", output)
        self.assertIn(str(self.catalog_path), output)

    def test_inspection_on_a_missing_catalog_has_a_json_form(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self.cli.main(["--catalog", str(self.catalog_path), "datasets", "--json"])
        self.assertEqual(code, 3)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["exists"])
        self.assertEqual(payload["error"], "catalog_not_initialized")
        self.assertIn("init", payload["hint"])

    def test_inspection_on_an_initialised_catalog_succeeds(self):
        self.cli.main(["--catalog", str(self.catalog_path), "init"])
        for command in ("stats", "datasets", "experiments", "artifacts", "metrics"):
            self.assertEqual(self.cli.main(["--catalog", str(self.catalog_path), command]), 0)

    def test_verify_reports_a_missing_dataset_version(self):
        self.cli.main(["--catalog", str(self.catalog_path), "init"])
        code = self.cli.main(["--catalog", str(self.catalog_path), "verify",
                              "--dataset-version", "dsv_missing"])
        self.assertEqual(code, 1)


# ─────────────────────────────────────────────────────────────
#  BUILD: VALID, EMPTY, IDEMPOTENT, FAILURE
# ─────────────────────────────────────────────────────────────


class TestBuildDataset(unittest.TestCase):
    def setUp(self):
        self.workspace = DatasetWorkspace()
        self.addCleanup(self.workspace.cleanup)
        self.cli = load_script("build_dataset")

    def _catalog_versions(self):
        if not self.workspace.catalog.exists():
            return []
        with Catalog(self.workspace.catalog) as catalog:
            return catalog.list_dataset_versions()

    def test_build_of_a_valid_dataset_succeeds(self):
        make_dataset(self.workspace.dataset)
        code = self.cli.main(self.workspace.build_argv())
        self.assertEqual(code, 0)

        manifest_path = self.workspace.manifests / "v1.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], STATUS_READY)
        self.assertEqual(len(manifest["images"]), 9)

        data_yaml = self.workspace.dataset / "data.yaml"
        self.assertTrue(data_yaml.is_file())
        self.assertIn("0: person", data_yaml.read_text(encoding="utf-8"))

    def test_build_registers_the_version_images_and_splits(self):
        make_dataset(self.workspace.dataset)
        self.cli.main(self.workspace.build_argv())
        with Catalog(self.workspace.catalog) as catalog:
            versions = catalog.list_dataset_versions()
            self.assertEqual(len(versions), 1)
            version_id = versions[0]["dataset_version_id"]
            self.assertEqual(len(catalog.list_images(version_id)), 9)
            self.assertEqual(catalog.stats()["split_assignment"], 9)
            self.assertEqual(versions[0]["status"], "active")

    def test_build_creates_the_catalog_when_it_does_not_exist(self):
        make_dataset(self.workspace.dataset)
        self.assertFalse(self.workspace.catalog.exists())
        self.cli.main(self.workspace.build_argv())
        self.assertTrue(self.workspace.catalog.exists())

    def test_rebuilding_is_idempotent(self):
        make_dataset(self.workspace.dataset)
        for _ in range(3):
            self.assertEqual(self.cli.main(self.workspace.build_argv()), 0)
        with Catalog(self.workspace.catalog) as catalog:
            stats = catalog.stats()
            self.assertEqual(stats["dataset_version"], 1)
            self.assertEqual(stats["image"], 9)
            self.assertEqual(stats["split_assignment"], 9)
            self.assertEqual(stats["validation_report"], 1)

    def test_building_an_empty_dataset_is_refused(self):
        code = self.cli.main(self.workspace.build_argv())
        self.assertEqual(code, 3)
        self.assertEqual(self._catalog_versions(), [])

    def test_a_refused_build_still_keeps_the_validation_report(self):
        self.cli.main(self.workspace.build_argv())
        report = self.workspace.reports / "v1.validation.json"
        self.assertTrue(report.is_file())
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"]["images"], 0)
        with Catalog(self.workspace.catalog) as catalog:
            self.assertEqual(catalog.stats()["validation_report"], 1)
            self.assertEqual(catalog.stats()["dataset_version"], 0)

    def test_allow_empty_registers_an_explicitly_empty_version(self):
        code = self.cli.main(self.workspace.build_argv(extra=["--allow-empty"]))
        self.assertEqual(code, 0)
        versions = self._catalog_versions()
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["status"], "empty")
        manifest = json.loads((self.workspace.manifests / "v1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], STATUS_EMPTY)

    def test_a_failed_validation_registers_no_dataset_version(self):
        root = self.workspace.dataset
        make_image(root / "images" / "train" / "s" / "broken.jpg")
        write_label(root / "labels" / "train" / "s" / "broken.txt", [(9, 0.5, 0.5, 0.2, 0.2)])
        code = self.cli.main(self.workspace.build_argv())
        self.assertEqual(code, 1)
        self.assertEqual(self._catalog_versions(), [])
        self.assertTrue((self.workspace.reports / "v1.validation.json").is_file())

    def test_registration_failure_leaves_no_dataset_version(self):
        make_dataset(self.workspace.dataset)
        with Catalog(self.workspace.catalog) as catalog:
            catalog.migrate()

        original = Catalog.register_dataset

        def exploding_register(self, **kwargs):
            raise RuntimeError("simulated registration failure")

        Catalog.register_dataset = exploding_register
        self.addCleanup(setattr, Catalog, "register_dataset", original)

        code = self.cli.main(self.workspace.build_argv())
        self.assertEqual(code, 1)
        with Catalog(self.workspace.catalog) as catalog:
            self.assertEqual(catalog.stats()["dataset_version"], 0)
            self.assertEqual(catalog.stats()["image"], 0)
            self.assertEqual(catalog.stats()["split_assignment"], 0)

    def test_a_failed_write_inside_the_transaction_rolls_back(self):
        """A failing image insert must undo the version row written before it."""
        from ml.contract import ImageRecord, make_image_id

        broken = ImageRecord(
            image_id=make_image_id("0" * 64),
            relpath="images/train/broken.jpg",
            sha256=None,          # violates the NOT NULL constraint on image.sha256
        )
        with Catalog(self.workspace.catalog) as catalog:
            catalog.migrate()
            with self.assertRaises(sqlite3.IntegrityError):
                catalog.register_dataset(
                    dataset_version_id="dsv_boom", name="boom", contract_version="1.0.0",
                    class_schema_version="1.0.0", manifest_path="m.json",
                    manifest_sha256="a" * 64, dataset_root=str(self.workspace.dataset),
                    images=[broken], assignments={}, records_by_id={},
                )
        with Catalog(self.workspace.catalog) as catalog:
            self.assertEqual(catalog.stats()["dataset_version"], 0)
            self.assertEqual(catalog.stats()["image"], 0)

    def test_build_rejects_a_missing_root(self):
        code = self.cli.main(self.workspace.build_argv(root=self.workspace.root / "nope"))
        self.assertEqual(code, 2)


# ─────────────────────────────────────────────────────────────
#  VALIDATION CLI
# ─────────────────────────────────────────────────────────────


class TestValidateCli(unittest.TestCase):
    def setUp(self):
        self.workspace = DatasetWorkspace()
        self.addCleanup(self.workspace.cleanup)
        self.cli = load_script("validate_dataset")

    def test_valid_dataset_exits_zero(self):
        make_dataset(self.workspace.dataset)
        code = self.cli.main(["--root", str(self.workspace.dataset),
                              "--report", str(self.workspace.reports / "r.json")])
        self.assertEqual(code, 0)
        self.assertTrue((self.workspace.reports / "r.json").is_file())

    def test_invalid_dataset_exits_one(self):
        root = self.workspace.dataset
        make_image(root / "images" / "train" / "s" / "a.jpg")
        write_label(root / "labels" / "train" / "s" / "a.txt", [(0, 1.5, 0.5, 0.2, 0.2)])
        code = self.cli.main(["--root", str(root)])
        self.assertEqual(code, 1)

    def test_missing_root_exits_two(self):
        code = self.cli.main(["--root", str(self.workspace.root / "nope")])
        self.assertEqual(code, 2)

    def test_empty_dataset_passes_validation_but_reports_empty_splits(self):
        import contextlib
        import io

        for split in ("train", "val", "test"):
            (self.workspace.dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self.cli.main(["--root", str(self.workspace.dataset)])
        self.assertEqual(code, 0)
        self.assertIn("split_empty", buffer.getvalue())


# ─────────────────────────────────────────────────────────────
#  TRAINING PREFLIGHT
# ─────────────────────────────────────────────────────────────


class TestTrainingPreflight(unittest.TestCase):
    def setUp(self):
        self.workspace = DatasetWorkspace()
        self.addCleanup(self.workspace.cleanup)
        self.build_cli = load_script("build_dataset")
        self.train_cli = load_script("train_person")

    def _data_yaml(self) -> Path:
        return self.workspace.dataset / "data.yaml"

    def _build(self):
        self.build_cli.main(self.workspace.build_argv())

    def test_preflight_passes_for_a_valid_built_dataset(self):
        make_dataset(self.workspace.dataset)
        self._build()
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertTrue(result.ok, result.render_text())
        self.assertEqual(result.split_counts["train"], 3)
        self.assertEqual(result.split_counts["val"], 3)

    def test_preflight_fails_for_an_empty_dataset(self):
        self.workspace.dataset.mkdir(parents=True, exist_ok=True)
        (self._data_yaml()).write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("train_images", [c.name for c in result.errors])

    def test_preflight_fails_when_a_label_file_is_missing(self):
        make_dataset(self.workspace.dataset)
        self._build()
        label = next((self.workspace.dataset / "labels" / "train").rglob("*.txt"))
        label.unlink()
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("train_labels", [c.name for c in result.errors])

    def test_preflight_fails_on_an_invalid_label(self):
        make_dataset(self.workspace.dataset)
        self._build()
        label = next((self.workspace.dataset / "labels" / "train").rglob("*.txt"))
        label.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("train_labels", [c.name for c in result.errors])

    def test_preflight_fails_when_no_manifest_exists(self):
        make_dataset(self.workspace.dataset)
        self._build()
        (self.workspace.manifests / "v1.json").unlink()
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("manifest", [c.name for c in result.errors])

    def test_preflight_fails_when_the_manifest_does_not_match_the_files(self):
        make_dataset(self.workspace.dataset)
        self._build()
        image = next((self.workspace.dataset / "images" / "train").rglob("*.jpg"))
        make_image(image, width=70, height=48, colour=(9, 9, 9))
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("manifest", [c.name for c in result.errors])

    def test_preflight_fails_for_an_empty_manifest_status(self):
        self.build_cli.main(self.workspace.build_argv(extra=["--allow-empty"]))
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        self.assertFalse(result.ok)
        self.assertIn("manifest", [c.name for c in result.errors])

    def test_training_is_refused_on_an_empty_dataset(self):
        self.workspace.dataset.mkdir(parents=True, exist_ok=True)
        (self._data_yaml()).write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        code = self.train_cli.main([
            "--data", str(self._data_yaml()),
            "--weights", str(PROJECT_ROOT / "models" / "yolov5s.pt"),
            "--project", str(self.workspace.root / "runs"),
            "--name", "nope",
            "--catalog", str(self.workspace.catalog),
        ])
        self.assertEqual(code, 4)
        self.assertFalse((self.workspace.root / "runs").exists())

    def test_training_is_refused_when_the_dataset_yaml_is_missing(self):
        code = self.train_cli.main([
            "--data", str(self.workspace.root / "missing.yaml"),
            "--weights", str(PROJECT_ROOT / "models" / "yolov5s.pt"),
        ])
        self.assertEqual(code, 2)

    def test_training_is_refused_when_the_checkpoint_is_missing(self):
        make_dataset(self.workspace.dataset)
        self._build()
        code = self.train_cli.main([
            "--data", str(self._data_yaml()),
            "--weights", str(self.workspace.root / "missing.pt"),
        ])
        self.assertEqual(code, 2)

    def test_preflight_result_is_serialisable(self):
        make_dataset(self.workspace.dataset)
        self._build()
        result = run_preflight(self._data_yaml(), manifest_dir=self.workspace.manifests)
        payload = result.to_dict()
        json.dumps(payload)
        self.assertTrue(payload["ok"])
        self.assertIn("checks", payload)


if __name__ == "__main__":
    unittest.main()
