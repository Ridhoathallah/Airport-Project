#!/usr/bin/env python3
"""Launch a fine-tuning run with full experiment provenance.

The run is described by a *resolved* configuration that is hashed and
stored, so a later experiment can be matched to the exact data, weights
and hyper-parameters it used. The base checkpoint is never modified: the
vendored ``yolov5/train.py`` writes its outputs to a separate run
directory and the original ``models/yolov5s.pt`` is only read.

Usage:
    venv/bin/python scripts/train_person.py --data datasets/people_v1/data.yaml \
        --weights models/yolov5s.pt --epochs 50 --imgsz 640 --batch 8 --device mps

    # show what would run, without running it
    venv/bin/python scripts/train_person.py --data ... --dry-run

Exit codes:
    0  training completed and checkpoints were produced
    1  training failed or produced no checkpoint
    2  bad usage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.catalog import Catalog, DEFAULT_CATALOG_PATH  # noqa: E402
from ml.contract import sha256_file, utc_now_iso  # noqa: E402
from ml.experiment import (  # noqa: E402
    ExperimentConfig,
    TrainingRun,
    capture_environment,
    create_experiment_id,
    find_checkpoints,
)
from ml.preflight import run_preflight  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PREFLIGHT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="dataset YAML consumed by yolov5/train.py")
    parser.add_argument("--weights", default="models/yolov5s.pt",
                        help="base checkpoint (read-only; never modified)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer", default="SGD", choices=("SGD", "Adam", "AdamW"))
    parser.add_argument("--device", default="cpu", help="cpu | mps | 0,1 (cuda)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hyp", default="", help="optional hyper-parameter YAML")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="people")
    parser.add_argument("--no-single-cls", action="store_true",
                        help="do not pass --single-cls (the dataset schema is single-class)")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--dataset-version-id", default=None,
                        help="dataset version id this run trains on (for traceability)")
    parser.add_argument("--manifest-sha256", default=None,
                        help="manifest hash of the dataset version (for traceability)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configuration and command, then exit")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="train without the readiness checks (recorded in the "
                             "experiment metadata; use with care)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra yolov5 argument, e.g. --set cache=disk")
    return parser


def _extra_from_pairs(pairs) -> dict:
    extra: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if value.lower() in ("true", "false"):
            extra[key] = value.lower() == "true"
        else:
            extra[key] = value
    return extra


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] dataset YAML not found: {data_path}", file=sys.stderr)
        return EXIT_USAGE
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] base checkpoint not found: {weights_path}", file=sys.stderr)
        return EXIT_USAGE

    # ── training preflight ──────────────────────────────────
    # Refuse to invoke the trainer on data that cannot produce a result:
    # an empty dataset, missing or invalid labels, or a dataset that was
    # never built into a verified version.
    preflight = run_preflight(data_path)
    print(preflight.render_text())
    if not preflight.ok:
        if not args.skip_preflight:
            print("[FAIL] training preflight failed -- refusing to start", file=sys.stderr)
            for check in preflight.errors:
                print(f"    - {check.name}: {check.detail}", file=sys.stderr)
            print("[INFO] build and verify the dataset version first: "
                  "scripts/build_dataset.py --root <root> --name <version>", file=sys.stderr)
            return EXIT_PREFLIGHT
        print("[WARN] --skip-preflight given: training despite failed checks",
              file=sys.stderr)

    try:
        extra = _extra_from_pairs(args.set)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_USAGE

    config = ExperimentConfig(
        data=str(data_path),
        weights=str(weights_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        hyp=args.hyp,
        project=args.project,
        name=args.name,
        single_cls=not args.no_single_cls,
        extra=extra,
    )
    command = config.build_command()
    env = capture_environment()
    created_at = utc_now_iso()
    experiment_id = create_experiment_id(config, created_at)
    base_sha = sha256_file(weights_path)
    preflight_payload = preflight.to_dict()
    preflight_payload["skipped"] = bool(args.skip_preflight)

    print(f"[INFO] experiment id : {experiment_id}")
    print(f"[INFO] config sha256  : {config.sha256}")
    print(f"[INFO] base checkpoint : {weights_path} ({base_sha[:12]}…)")
    print(f"[INFO] command        : {' '.join(command)}")

    if args.dry_run:
        print("[INFO] dry run -- training not started")
        return EXIT_OK

    run_dir = Path(args.project) / args.name
    log_dir = Path(args.project) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    run = TrainingRun(
        experiment_id=experiment_id,
        status="running",
        started_at_utc=utc_now_iso(),
        command=command,
        log_path=str(log_path),
    )

    if not args.no_catalog:
        with Catalog(args.catalog) as catalog:
            catalog.register_experiment(
                experiment_id=experiment_id,
                dataset_version_id=args.dataset_version_id,
                manifest_sha256=args.manifest_sha256,
                base_checkpoint=str(weights_path),
                base_checkpoint_sha256=base_sha,
                config_sha256=config.sha256,
                resolved_config=config.to_dict(),
                status="running",
                device=args.device,
                env=env,
                log_path=str(log_path),
                notes=("preflight skipped" if args.skip_preflight else ""),
                created_at_utc=created_at,
            )

    print(f"[INFO] starting training, log -> {log_path}")
    return_code = 1
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
            return_code = process.wait()
    except KeyboardInterrupt:
        run.status = "failed"
        run.error = "interrupted by user"
        _finalise(args, run, run_dir, experiment_id, config, base_sha, env, created_at,
                  preflight_payload)
        print("\n[FAIL] training interrupted", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        _finalise(args, run, run_dir, experiment_id, config, base_sha, env, created_at,
                  preflight_payload)
        print(f"[FAIL] could not start training: {exc}", file=sys.stderr)
        return EXIT_FAILED

    run.ended_at_utc = utc_now_iso()
    run.checkpoints = find_checkpoints(run_dir)
    if return_code != 0:
        run.status = "failed"
        run.error = f"yolov5/train.py exited with code {return_code}"
    elif not run.checkpoints:
        run.status = "failed"
        run.error = "training finished but no checkpoint was produced"
    else:
        run.status = "completed"

    _finalise(args, run, run_dir, experiment_id, config, base_sha, env, created_at,
              preflight_payload)

    if run.status == "completed":
        print(f"[INFO] training complete; checkpoints: {run.checkpoints}")
        print("[INFO] next: scripts/evaluate_model.py --weights <best.pt> "
              "--baseline models/yolov5s.pt")
        return EXIT_OK
    print(f"[FAIL] {run.error}", file=sys.stderr)
    return EXIT_FAILED


def _finalise(args, run: TrainingRun, run_dir: Path, experiment_id: str,
              config: ExperimentConfig, base_sha, env: dict, created_at: str,
              preflight_payload: Optional[dict] = None) -> None:
    """Persist the final experiment state and its artifacts."""
    if args.no_catalog:
        return
    try:
        with Catalog(args.catalog) as catalog:
            catalog.register_experiment(
                experiment_id=experiment_id,
                dataset_version_id=args.dataset_version_id,
                manifest_sha256=args.manifest_sha256,
                base_checkpoint=config.weights,
                base_checkpoint_sha256=base_sha,
                config_sha256=config.sha256,
                resolved_config=config.to_dict(),
                status=run.status,
                device=config.device,
                env=env,
                log_path=run.log_path,
                notes=("preflight skipped" if args.skip_preflight else ""),
                created_at_utc=created_at,
            )
            for role, path in run.checkpoints.items():
                target = Path(path)
                catalog.register_artifact(
                    artifact_id=f"art_{experiment_id}_{role}",
                    experiment_id=experiment_id,
                    role=role,
                    path=str(target),
                    sha256=sha256_file(target) if target.is_file() else None,
                    size_bytes=target.stat().st_size if target.is_file() else None,
                )
            run_summary = run_dir / "experiment.json"
            try:
                run_summary.parent.mkdir(parents=True, exist_ok=True)
                run_summary.write_text(
                    json.dumps({"run": run.to_dict(), "config": config.to_dict(),
                                "environment": env,
                                "preflight": preflight_payload or {}}, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
    except Exception as exc:  # catalog failures must not mask the training outcome
        print(f"[WARN] could not update the catalog: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
