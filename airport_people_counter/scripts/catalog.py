#!/usr/bin/env python3
"""Inspect the dataset metadata catalog.

The catalog is a small SQLite index of dataset versions, image metadata,
split assignments, validation reports, experiments and model artifacts.
It never stores image bytes, and it never claims a dataset is immutable --
``verify`` recomputes the recorded hashes and reports any drift.

The catalog is created explicitly with ``init``; the inspection commands
never create it implicitly, because an auto-created catalog would look
like a populated one.

Usage:
    venv/bin/python scripts/catalog.py init
    venv/bin/python scripts/catalog.py stats
    venv/bin/python scripts/catalog.py datasets
    venv/bin/python scripts/catalog.py experiments
    venv/bin/python scripts/catalog.py artifacts --experiment <id>
    venv/bin/python scripts/catalog.py metrics --experiment <id>
    venv/bin/python scripts/catalog.py verify --dataset-version <id>

Exit codes:
    0  the command succeeded
    1  the command failed (e.g. a hash mismatch)
    2  bad usage
    3  the catalog does not exist yet -- run "catalog.py init"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog, DEFAULT_CATALOG_PATH  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CATALOG_MISSING = 3

#: Printed when the catalog has not been initialised yet.
MISSING_HINT = (
    "The dataset catalog does not exist yet.\n"
    "  Create it with:  venv/bin/python scripts/catalog.py init\n"
    "  (a dataset build also creates it automatically)"
)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit JSON instead of text")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH),
                        help="path to the metadata catalog SQLite file")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", parents=[common],
                   help="create the catalog and apply schema migrations (idempotent)")
    sub.add_parser("stats", parents=[common], help="row counts per table")
    sub.add_parser("datasets", parents=[common], help="registered dataset versions")
    sub.add_parser("experiments", parents=[common], help="registered experiments")
    artifacts = sub.add_parser("artifacts", parents=[common], help="model artifacts")
    artifacts.add_argument("--experiment", default=None,
                           help="only artifacts of this experiment")
    metrics = sub.add_parser("metrics", parents=[common], help="recorded metrics")
    metrics.add_argument("--experiment", default=None,
                         help="only metrics of this experiment")
    verify = sub.add_parser("verify", parents=[common],
                             help="re-check recorded dataset hashes")
    verify.add_argument("--dataset-version", required=True, help="dataset version id")
    return parser


def _emit(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key:<22}: {value}")
    else:
        print(json.dumps(payload, indent=2, default=str))


def _catalog_missing(args) -> int:
    """Report a missing catalog clearly, in text or JSON form."""
    if getattr(args, "json", False):
        print(json.dumps({
            "catalog": str(args.catalog),
            "exists": False,
            "error": "catalog_not_initialized",
            "hint": MISSING_HINT,
        }, indent=2))
    else:
        print(f"[ERROR] catalog not found: {args.catalog}")
        print(MISSING_HINT)
    return EXIT_CATALOG_MISSING


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    catalog_path = Path(args.catalog)

    # ── init: create the catalog and run the migrations ─────
    if args.command == "init":
        with Catalog(catalog_path) as catalog:
            version = catalog.version
            stats = catalog.stats()
        if getattr(args, "json", False):
            print(json.dumps({"catalog": str(catalog_path), "created": True,
                              "schema_version": version}, indent=2))
        else:
            print(f"[OK] catalog ready: {catalog_path}")
            print(f"[INFO] schema version: {version}")
        return EXIT_OK

    # ── every other command needs an existing catalog ───────
    if not catalog_path.exists():
        return _catalog_missing(args)

    with Catalog(catalog_path) as catalog:
        if args.command == "stats":
            _emit(catalog.stats(), args.json)
        elif args.command == "datasets":
            rows = catalog.list_dataset_versions()
            if not rows and not getattr(args, "json", False):
                print("[INFO] the catalog is empty -- no dataset versions registered yet")
                print("[INFO] build one with: scripts/build_dataset.py --root <root> --name <version>")
            _emit(rows, args.json)
        elif args.command == "experiments":
            rows = catalog.list_experiments()
            if not rows and not getattr(args, "json", False):
                print("[INFO] the catalog is empty -- no training experiments registered yet")
            _emit(rows, args.json)
        elif args.command == "artifacts":
            _emit(catalog.list_artifacts(args.experiment), args.json)
        elif args.command == "metrics":
            _emit(catalog.list_metrics(args.experiment), args.json)
        elif args.command == "verify":
            problems = catalog.verify_dataset(args.dataset_version)
            if problems:
                print(f"[FAIL] {len(problems)} problem(s) found:")
                for problem in problems:
                    print(f"    - {problem}")
                return EXIT_FAILED
            print(f"[OK] dataset version {args.dataset_version} matches its recorded hashes")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
