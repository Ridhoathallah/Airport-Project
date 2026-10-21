#!/usr/bin/env python3
"""Build a versioned dataset: validate, split, write the manifest, register it.

The output is three artefacts:

* a **manifest** (JSON) -- the authoritative, hash-verified description of
  the dataset version, stored under ``datasets/manifests/``;
* a **YOLOv5 ``data.yaml``** -- what ``yolov5/train.py`` consumes;
* **catalog rows** -- the metadata index that ties experiments to data.

The split is group-first and deterministic: all frames from one source
video or session stay together, and the same ``--seed`` always produces
the same assignment.

An **empty dataset is rejected** with exit code 3. Building a version
that contains no images would create a record that looks usable but
cannot train anything, so the build refuses. Pass ``--allow-empty`` to
register it explicitly as an ``empty`` (not training-ready) version.

The validation report is written to disk and registered in **every**
outcome -- including a failed validation -- so the evidence of what went
wrong is never lost.

Usage:
    venv/bin/python scripts/build_dataset.py --root datasets/people_v1 \
        --name people_v1 --seed 20261020 --ratios 0.7/0.2/0.1

    # build a version and register it in a specific catalog
    venv/bin/python scripts/build_dataset.py --root <root> --name <version> \
        --catalog datasets/metadata/dataset_catalog.db

Exit codes:
    0  the dataset version was built and registered
    1  validation or leakage checks failed (no dataset version was registered)
    2  bad usage
    3  the dataset contains no images (no training-ready version was registered)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog, DEFAULT_CATALOG_PATH  # noqa: E402
from ml.contract import DEFAULT_CLASS_SCHEMA, AnnotationStatus, sha256_json  # noqa: E402
from ml.manifest import (  # noqa: E402
    STATUS_EMPTY,
    STATUS_READY,
    build_manifest,
    write_manifest,
)
from ml.split import SplitPolicy, plan_splits  # noqa: E402
from ml.validate import validate_dataset  # noqa: E402

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_EMPTY = 3

MANIFEST_DIR = PROJECT_ROOT / "datasets" / "manifests"
REPORT_DIR = PROJECT_ROOT / "datasets" / "reports"


def parse_ratios(text: str) -> dict:
    parts = [p.strip() for p in str(text).split("/") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must look like '0.7/0.2/0.1'")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ratios must be numeric: {exc}") from exc
    if any(v < 0 for v in values) or sum(values) <= 0:
        raise argparse.ArgumentTypeError("ratios must be non-negative and sum to > 0")
    return {"train": values[0], "val": values[1], "test": values[2]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "workflow:\n"
            "  1. put images in <root>/images/{train,val,test} and matching YOLO\n"
            "     label files in <root>/labels/{train,val,test}\n"
            "  2. validate:  scripts/validate_dataset.py --root <root>\n"
            "  3. build:     scripts/build_dataset.py --root <root> --name <version>\n"
            "  4. train:     scripts/train_person.py --data <root>/data.yaml "
            "--weights models/yolov5s.pt\n"
            "  5. evaluate:  scripts/evaluate_model.py --weights <best.pt> "
            "--baseline models/yolov5s.pt --data <root>/data.yaml --split test\n"
        ),
    )
    parser.add_argument("--root", required=True,
                        help="dataset root containing images/ and labels/")
    parser.add_argument("--name", required=True,
                        help="dataset version name, e.g. people_v1")
    parser.add_argument("--seed", type=int, default=20261020,
                        help="split seed; the same seed always reproduces the same split")
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("0.7/0.2/0.1"),
                        help="train/val/test ratios, e.g. 0.7/0.2/0.1")
    parser.add_argument("--manifest-dir", default=str(MANIFEST_DIR),
                        help="directory the manifest is written to")
    parser.add_argument("--report-dir", default=str(REPORT_DIR),
                        help="directory the validation report is written to "
                             "(kept even when the build fails)")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH),
                        help="metadata catalog to register the version in")
    parser.add_argument("--no-catalog", action="store_true",
                        help="do not touch the catalog (the manifest is still written)")
    parser.add_argument("--orphan-policy", default="singleton", choices=("singleton", "error"),
                        help="what to do with images that have no source group: "
                             "'singleton' treats each as its own group (and warns), "
                             "'error' refuses the split")
    parser.add_argument("--annotation-status", default=AnnotationStatus.UNREVIEWED.value,
                        help="annotation status recorded for images that have a label file "
                             "(use 'negative_reviewed' for reviewed images with no person)")
    parser.add_argument("--notes", default="", help="free-text note stored with the version")
    parser.add_argument("--allow-empty", action="store_true",
                        help="register an explicitly empty version instead of refusing")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute everything but write nothing")
    return parser


def write_data_yaml(root: Path, name: str) -> Path:
    """Write the YOLOv5 ``data.yaml`` describing the split layout.

    ``path:`` is written relative to the vendored YOLOv5 root, because that
    is where ``yolov5/utils/general.check_dataset`` resolves a relative
    path from (the same convention ``yolov5/data/coco128.yaml`` uses).
    """
    yolov5_root = PROJECT_ROOT / "yolov5"
    try:
        path_value = os.path.relpath(root.resolve(), yolov5_root.resolve())
    except ValueError:  # pragma: no cover - different drives on Windows
        path_value = root.resolve().as_posix()
    lines = [
        f"# YOLOv5 data description for dataset version '{name}'.",
        "# Generated by scripts/build_dataset.py -- the split directories are the",
        "# authority; this file only tells YOLOv5 where to find them.",
        "#",
        "# 'path' is relative to the vendored yolov5/ root, which is where",
        "# yolov5/utils/general.check_dataset resolves a relative path from.",
        "",
        f"path: {Path(path_value).as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "# Single-class schema: the runtime consumes COCO class 0 only.",
        "names:",
        "  0: person",
        "",
    ]
    target = root / "data.yaml"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] dataset root not found: {root}", file=sys.stderr)
        return EXIT_USAGE

    manifest_dir = Path(args.manifest_dir)
    report_dir = Path(args.report_dir)
    report_path = report_dir / f"{args.name}.validation.json"

    # ── 1. validate ──────────────────────────────────────────
    print(f"[INFO] validating {root} ...")
    report, records = validate_dataset(
        root,
        class_schema=DEFAULT_CLASS_SCHEMA,
        annotation_status=args.annotation_status,
    )
    print(report.render_text())

    # The report is preserved in every outcome, including a failure.
    if not args.dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] validation report: {report_path}")

    if not report.ok:
        _register_report_only(args, report, report_path)
        print("[FAIL] validation produced errors -- fix them before building a version",
              file=sys.stderr)
        return EXIT_FINDINGS

    # ── 2. an empty dataset is not a training-ready version ──
    if not records and not args.allow_empty:
        _register_report_only(args, report, report_path)
        print("[FAIL] the dataset contains no images", file=sys.stderr)
        print("[FAIL] refusing to register a dataset version that cannot be trained on",
              file=sys.stderr)
        print("[INFO] add images and label files under "
              f"{root}/images and {root}/labels, or pass --allow-empty to register "
              "an explicitly empty version", file=sys.stderr)
        return EXIT_EMPTY

    # ── 3. split ─────────────────────────────────────────────
    policy = SplitPolicy(seed=args.seed, ratios=args.ratios, orphan_policy=args.orphan_policy)
    result = plan_splits(records, policy)
    print(f"[INFO] split plan: {result.counts}")
    for finding in result.findings:
        print(f"    - {finding.severity.value.upper():<7} {finding.code}: {finding.message}")
    if not result.ok:
        _register_report_only(args, report, report_path)
        print("[FAIL] leakage checks failed -- refusing to build a dataset version",
              file=sys.stderr)
        return EXIT_FINDINGS

    for record in records:
        record.split = result.assignments.get(record.image_id)

    # ── 4. manifest ──────────────────────────────────────────
    root_relative = os.path.relpath(root.resolve(), manifest_dir.resolve())
    status = STATUS_READY if records else STATUS_EMPTY
    manifest = build_manifest(
        dataset_version=args.name,
        images=records,
        splits=result.splits(),
        class_schema=DEFAULT_CLASS_SCHEMA,
        seed=args.seed,
        split_policy=policy.to_dict(),
        root=root_relative,
        notes=args.notes,
        status=status,
    )
    manifest_path = manifest_dir / f"{args.name}.json"
    data_yaml = root / "data.yaml"

    if args.dry_run:
        print("[INFO] dry run -- nothing written")
        print(f"[INFO] would write manifest : {manifest_path}")
        print(f"[INFO] would write data.yaml: {data_yaml}")
        return EXIT_OK

    write_manifest(manifest_path, manifest)
    write_data_yaml(root, args.name)
    print(f"[INFO] manifest : {manifest_path}")
    print(f"[INFO] data.yaml: {data_yaml}")
    print(f"[INFO] dataset version id: {manifest.dataset_version_id}")
    print(f"[INFO] manifest sha256   : {manifest.manifest_sha256}")

    if status == STATUS_EMPTY:
        print("[WARN] this version contains no images and is NOT training-ready",
              file=sys.stderr)

    # ── 5. register atomically ───────────────────────────────
    if not args.no_catalog:
        try:
            with Catalog(args.catalog) as catalog:
                catalog.register_dataset(
                    dataset_version_id=manifest.dataset_version_id,
                    name=args.name,
                    contract_version=manifest.contract_version,
                    class_schema_version=manifest.class_schema_version,
                    manifest_path=str(manifest_path),
                    manifest_sha256=manifest.manifest_sha256,
                    dataset_root=str(root.resolve()),
                    images=records,
                    assignments=result.assignments,
                    records_by_id={r.image_id: r for r in records},
                    seed=args.seed,
                    split_policy=policy.to_dict(),
                    status="active" if status == STATUS_READY else "empty",
                    validation_report=report.to_dict(),
                    validation_report_id=report_id(report.to_dict()),
                    validation_report_path=str(report_path),
                    notes=args.notes,
                )
            print(f"[INFO] catalog  : {args.catalog}")
        except Exception as exc:
            # The registration is a single transaction, so a failure here
            # leaves no dataset version behind. Report it cleanly instead
            # of surfacing a traceback.
            print(f"[FAIL] could not register the dataset version: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("[FAIL] the catalog was not modified; the manifest on disk is "
                  "not registered and must not be trained on", file=sys.stderr)
            return EXIT_FINDINGS

    if status == STATUS_EMPTY:
        print("[WARN] registered as an EMPTY version -- the training preflight will "
              "refuse it", file=sys.stderr)
    return EXIT_OK


def report_id(report: dict) -> str:
    """A stable id for a validation report.

    The report's creation timestamp is excluded, so re-running the same
    validation updates the same catalog row instead of accumulating a new
    one. A genuinely different set of findings produces a different id.
    """
    body = {k: v for k, v in report.items() if k != "created_at_utc"}
    return "rep_" + sha256_json(body)[:16]


def _register_report_only(args, report, report_path) -> None:
    """Record a validation report even when no dataset version is created.

    A failed build must not leave a half-registered version behind, but the
    evidence of *why* it failed is worth keeping.
    """
    if args.no_catalog or args.dry_run:
        return
    try:
        with Catalog(args.catalog) as catalog:
            catalog.register_validation_report(
                report_id=report_id(report.to_dict()),
                dataset_version_id=None,
                report=report.to_dict(),
                path=str(report_path),
            )
        print(f"[INFO] validation report registered in {args.catalog}")
    except Exception as exc:
        print(f"[WARN] could not register the validation report: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
