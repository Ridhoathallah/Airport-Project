"""Versioned SQLite catalog for dataset and experiment metadata.

The catalog is an *index*, not the source of truth. The authoritative
record of a dataset is its JSON manifest (see :mod:`ml.manifest`); the
catalog stores manifests, their hashes, image metadata, split assignments,
validation reports, experiments and model artifacts so that an experiment
can be traced back to the exact data it was trained on.

Deliberate constraints:

* this database is **separate** from the runtime monitoring database and
  shares no schema, connection or lifecycle with it;
* no image bytes or video files are stored here -- only paths and hashes;
* registering the same dataset version twice is idempotent;
* a dataset is never *declared* immutable -- :meth:`Catalog.verify_dataset`
  recomputes the recorded hashes and reports drift.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .contract import sha256_json, to_signed64, utc_now_iso

#: Current schema version. Bump when a migration is added.
CATALOG_VERSION = 1

DEFAULT_CATALOG_PATH = Path("datasets/metadata/dataset_catalog.db")


class CatalogError(Exception):
    """Raised for catalog misuse (unknown dataset version, bad arguments)."""


_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS dataset_version (
    dataset_version_id   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    contract_version     TEXT NOT NULL,
    class_schema_version TEXT NOT NULL,
    created_at_utc       TEXT NOT NULL,
    manifest_path        TEXT NOT NULL,
    manifest_sha256      TEXT NOT NULL,
    dataset_root         TEXT,
    seed                 INTEGER,
    split_policy_json    TEXT,
    status               TEXT NOT NULL DEFAULT 'active',
    notes                TEXT,
    UNIQUE (name, manifest_sha256)
);

CREATE TABLE IF NOT EXISTS source_asset (
    source_id          TEXT PRIMARY KEY,
    kind               TEXT NOT NULL,
    ref                TEXT,
    sha256             TEXT,
    source_group       TEXT,
    captured_at_utc    TEXT,
    license            TEXT,
    usage_restrictions TEXT,
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS image (
    image_id           TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_version(dataset_version_id)
                       ON DELETE CASCADE,
    relpath            TEXT NOT NULL,
    sha256             TEXT NOT NULL,
    width              INTEGER NOT NULL DEFAULT 0,
    height             INTEGER NOT NULL DEFAULT 0,
    source_id          TEXT REFERENCES source_asset(source_id),
    source_group       TEXT,
    captured_at_utc    TEXT,
    annotation_status  TEXT NOT NULL DEFAULT 'unreviewed',
    class_schema_version TEXT NOT NULL,
    phash              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_image_hash ON image(sha256);
CREATE INDEX IF NOT EXISTS idx_image_group ON image(source_group);
CREATE INDEX IF NOT EXISTS idx_image_dataset ON image(dataset_version_id);

CREATE TABLE IF NOT EXISTS split_assignment (
    image_id           TEXT NOT NULL REFERENCES image(image_id) ON DELETE CASCADE,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_version(dataset_version_id)
                       ON DELETE CASCADE,
    split              TEXT NOT NULL,
    source_group       TEXT,
    assigned_at_utc    TEXT NOT NULL,
    PRIMARY KEY (image_id, dataset_version_id)
);
CREATE INDEX IF NOT EXISTS idx_split_name ON split_assignment(dataset_version_id, split);

CREATE TABLE IF NOT EXISTS validation_report (
    report_id          TEXT PRIMARY KEY,
    dataset_version_id TEXT REFERENCES dataset_version(dataset_version_id) ON DELETE CASCADE,
    created_at_utc     TEXT NOT NULL,
    report_sha256      TEXT NOT NULL,
    status             TEXT NOT NULL,
    counts_json        TEXT,
    findings_json      TEXT,
    path               TEXT
);

CREATE TABLE IF NOT EXISTS experiment (
    experiment_id       TEXT PRIMARY KEY,
    created_at_utc      TEXT NOT NULL,
    dataset_version_id  TEXT REFERENCES dataset_version(dataset_version_id),
    manifest_sha256     TEXT,
    base_checkpoint     TEXT,
    base_checkpoint_sha256 TEXT,
    config_sha256       TEXT NOT NULL,
    resolved_config_json TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'planned',
    device              TEXT,
    env_json            TEXT,
    log_path            TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiment_dataset ON experiment(dataset_version_id);

CREATE TABLE IF NOT EXISTS model_artifact (
    artifact_id   TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment(experiment_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        TEXT,
    bytes         INTEGER,
    created_at_utc TEXT NOT NULL,
    UNIQUE (experiment_id, role)
);

CREATE TABLE IF NOT EXISTS metric (
    metric_id           TEXT PRIMARY KEY,
    experiment_id       TEXT NOT NULL REFERENCES experiment(experiment_id) ON DELETE CASCADE,
    dataset_version_id  TEXT REFERENCES dataset_version(dataset_version_id),
    split               TEXT,
    metric_name         TEXT NOT NULL,
    value               REAL,
    threshold           REAL,
    evaluated_at_utc    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metric_experiment ON metric(experiment_id);
"""


class Catalog:
    """Small, documented API over the metadata catalog."""

    def __init__(self, path=DEFAULT_CATALOG_PATH):
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── lifecycle ───────────────────────────────────────────
    def __enter__(self) -> "Catalog":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> "Catalog":
        """Open (creating if needed) the catalog and apply pending migrations."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._conn = conn
        self.migrate()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise CatalogError("catalog is not open; use Catalog(...) as a context manager")
        return self._conn

    def migrate(self) -> int:
        """Apply forward-only migrations; returns the resulting schema version."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self.conn.executescript(_MIGRATION_1)
            version = 1
        self.conn.execute(f"PRAGMA user_version = {int(version)}")
        self.conn.commit()
        return int(version)

    @property
    def version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    # ── dataset versions ────────────────────────────────────
    def register_dataset_version(
        self,
        *,
        dataset_version_id: str,
        name: str,
        contract_version: str,
        class_schema_version: str,
        manifest_path: str,
        manifest_sha256: str,
        dataset_root: Optional[str] = None,
        seed: Optional[int] = None,
        split_policy: Optional[dict] = None,
        created_at_utc: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """Register a dataset version (idempotent) and return its id."""
        payload = (
            dataset_version_id,
            name,
            contract_version,
            class_schema_version,
            created_at_utc or utc_now_iso(),
            str(manifest_path),
            manifest_sha256,
            str(dataset_root) if dataset_root else None,
            seed,
            json.dumps(split_policy or {}, sort_keys=True),
            notes,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO dataset_version (dataset_version_id, name, contract_version,
                    class_schema_version, created_at_utc, manifest_path, manifest_sha256,
                    dataset_root, seed, split_policy_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version_id) DO UPDATE SET
                    name=excluded.name,
                    manifest_path=excluded.manifest_path,
                    manifest_sha256=excluded.manifest_sha256,
                    dataset_root=excluded.dataset_root,
                    seed=excluded.seed,
                    split_policy_json=excluded.split_policy_json,
                    notes=excluded.notes
                """,
                payload,
            )
        return dataset_version_id

    def register_images(self, dataset_version_id: str, images: list) -> int:
        """Register image metadata for a dataset version (idempotent)."""
        rows = [
            (
                image.image_id,
                dataset_version_id,
                image.relpath,
                image.sha256,
                int(image.width or 0),
                int(image.height or 0),
                image.source_group,
                image.captured_at_utc,
                image.annotation_status,
                image.class_schema_version,
                to_signed64(image.phash) if image.phash is not None else None,
            )
            for image in images
        ]
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO image (image_id, dataset_version_id, relpath, sha256, width,
                    height, source_group, captured_at_utc, annotation_status,
                    class_schema_version, phash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    relpath=excluded.relpath,
                    sha256=excluded.sha256,
                    width=excluded.width,
                    height=excluded.height,
                    source_group=excluded.source_group,
                    captured_at_utc=excluded.captured_at_utc,
                    annotation_status=excluded.annotation_status,
                    phash=excluded.phash
                """,
                rows,
            )
        return len(rows)

    def register_splits(self, dataset_version_id: str, assignments: dict, records: dict) -> int:
        """Register split assignments (idempotent)."""
        now = utc_now_iso()
        rows = []
        for image_id, split in assignments.items():
            record = records.get(image_id)
            rows.append((
                image_id,
                dataset_version_id,
                split,
                getattr(record, "source_group", None),
                now,
            ))
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO split_assignment (image_id, dataset_version_id, split,
                    source_group, assigned_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id, dataset_version_id) DO UPDATE SET
                    split=excluded.split,
                    source_group=excluded.source_group
                """,
                rows,
            )
        return len(rows)

    def register_dataset(
        self,
        *,
        dataset_version_id: str,
        name: str,
        contract_version: str,
        class_schema_version: str,
        manifest_path: str,
        manifest_sha256: str,
        dataset_root: str,
        images: list,
        assignments: dict,
        records_by_id: dict,
        seed: Optional[int] = None,
        split_policy: Optional[dict] = None,
        status: str = "active",
        validation_report: Optional[dict] = None,
        validation_report_id: Optional[str] = None,
        validation_report_path: Optional[str] = None,
        created_at_utc: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """Register a complete dataset version atomically.

        The version row, its images, its split assignments and the
        validation report are written in **one transaction**. Either the
        whole dataset version becomes visible, or nothing does -- a failure
        half-way through can never leave an apparently valid version
        behind. Re-running with the same content updates the existing
        rows instead of duplicating them.
        """
        now = created_at_utc or utc_now_iso()
        image_rows = [
            (
                image.image_id,
                dataset_version_id,
                image.relpath,
                image.sha256,
                int(image.width or 0),
                int(image.height or 0),
                image.source_group,
                image.captured_at_utc,
                image.annotation_status,
                image.class_schema_version,
                to_signed64(image.phash) if image.phash is not None else None,
            )
            for image in images
        ]
        split_rows = [
            (
                image_id,
                dataset_version_id,
                split,
                getattr(records_by_id.get(image_id), "source_group", None),
                now,
            )
            for image_id, split in assignments.items()
        ]
        with self.conn:                      # one transaction for the whole version
            self.conn.execute(
                """
                INSERT INTO dataset_version (dataset_version_id, name, contract_version,
                    class_schema_version, created_at_utc, manifest_path, manifest_sha256,
                    dataset_root, seed, split_policy_json, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version_id) DO UPDATE SET
                    name=excluded.name,
                    manifest_path=excluded.manifest_path,
                    manifest_sha256=excluded.manifest_sha256,
                    dataset_root=excluded.dataset_root,
                    seed=excluded.seed,
                    split_policy_json=excluded.split_policy_json,
                    status=excluded.status,
                    notes=excluded.notes
                """,
                (
                    dataset_version_id,
                    name,
                    contract_version,
                    class_schema_version,
                    now,
                    str(manifest_path),
                    manifest_sha256,
                    str(dataset_root),
                    seed,
                    json.dumps(split_policy or {}, sort_keys=True),
                    status,
                    notes,
                ),
            )
            if image_rows:
                self.conn.executemany(
                    """
                    INSERT INTO image (image_id, dataset_version_id, relpath, sha256, width,
                        height, source_group, captured_at_utc, annotation_status,
                        class_schema_version, phash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        relpath=excluded.relpath,
                        sha256=excluded.sha256,
                        width=excluded.width,
                        height=excluded.height,
                        source_group=excluded.source_group,
                        captured_at_utc=excluded.captured_at_utc,
                        annotation_status=excluded.annotation_status,
                        phash=excluded.phash
                    """,
                    image_rows,
                )
            if split_rows:
                self.conn.executemany(
                    """
                    INSERT INTO split_assignment (image_id, dataset_version_id, split,
                        source_group, assigned_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_id, dataset_version_id) DO UPDATE SET
                        split=excluded.split,
                        source_group=excluded.source_group
                    """,
                    split_rows,
                )
            if validation_report is not None and validation_report_id:
                self.conn.execute(
                    """
                    INSERT INTO validation_report (report_id, dataset_version_id,
                        created_at_utc, report_sha256, status, counts_json, findings_json, path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(report_id) DO UPDATE SET
                        status=excluded.status,
                        counts_json=excluded.counts_json,
                        findings_json=excluded.findings_json,
                        path=excluded.path
                    """,
                    (
                        validation_report_id,
                        dataset_version_id,
                        validation_report.get("created_at_utc") or now,
                        sha256_json(validation_report),
                        "pass" if validation_report.get("ok") else "fail",
                        json.dumps(validation_report.get("counts") or {}, sort_keys=True),
                        json.dumps(validation_report.get("findings") or [], sort_keys=True),
                        validation_report_path,
                    ),
                )
        return dataset_version_id

    def register_validation_report(
        self,
        *,
        report_id: str,
        dataset_version_id: Optional[str],
        report: dict,
        path: Optional[str] = None,
    ) -> str:
        """Register a validation report (idempotent)."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO validation_report (report_id, dataset_version_id, created_at_utc,
                    report_sha256, status, counts_json, findings_json, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    status=excluded.status,
                    counts_json=excluded.counts_json,
                    findings_json=excluded.findings_json,
                    path=excluded.path
                """,
                (
                    report_id,
                    dataset_version_id,
                    report.get("created_at_utc") or utc_now_iso(),
                    sha256_json(report),
                    "pass" if report.get("ok") else "fail",
                    json.dumps(report.get("counts") or {}, sort_keys=True),
                    json.dumps(report.get("findings") or [], sort_keys=True),
                    path,
                ),
            )
        return report_id

    def list_dataset_versions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM dataset_version ORDER BY created_at_utc DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_dataset_version(self, dataset_version_id: str) -> Optional[dict]:
        """One dataset version row, or ``None`` when it is not registered."""
        row = self.conn.execute(
            "SELECT * FROM dataset_version WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def find_dataset_version_by_root(self, dataset_root: str) -> Optional[dict]:
        """The most recent dataset version registered for a dataset root."""
        row = self.conn.execute(
            "SELECT * FROM dataset_version WHERE dataset_root = ? "
            "ORDER BY created_at_utc DESC LIMIT 1",
            (str(dataset_root),),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_images(self, dataset_version_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM image WHERE dataset_version_id = ? ORDER BY relpath",
            (dataset_version_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def verify_dataset(self, dataset_version_id: str) -> list[str]:
        """Recompute stored image hashes and report any drift.

        Returns a list of human-readable problems; empty means the dataset
        on disk still matches what the catalog recorded.
        """
        from .contract import sha256_file

        problems: list[str] = []
        version = self.conn.execute(
            "SELECT * FROM dataset_version WHERE dataset_version_id = ?", (dataset_version_id,)
        ).fetchone()
        if version is None:
            return [f"unknown dataset version: {dataset_version_id}"]
        dataset_root = version["dataset_root"]
        if not dataset_root:
            return [f"dataset version {dataset_version_id} has no recorded dataset root"]
        root = Path(dataset_root)
        for row in self.conn.execute(
            "SELECT relpath, sha256 FROM image WHERE dataset_version_id = ?", (dataset_version_id,)
        ):
            target = root / row["relpath"]
            if not target.exists():
                problems.append(f"missing file: {row['relpath']}")
                continue
            digest = sha256_file(target)
            if digest != row["sha256"]:
                problems.append(f"hash mismatch: {row['relpath']}")
        return problems

    # ── experiments ─────────────────────────────────────────
    def register_experiment(
        self,
        *,
        experiment_id: str,
        dataset_version_id: Optional[str],
        manifest_sha256: Optional[str],
        base_checkpoint: Optional[str],
        base_checkpoint_sha256: Optional[str],
        config_sha256: str,
        resolved_config: dict,
        status: str = "planned",
        device: Optional[str] = None,
        env: Optional[dict] = None,
        log_path: Optional[str] = None,
        notes: str = "",
        created_at_utc: Optional[str] = None,
    ) -> str:
        """Register a training experiment (idempotent)."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO experiment (experiment_id, created_at_utc, dataset_version_id,
                    manifest_sha256, base_checkpoint, base_checkpoint_sha256, config_sha256,
                    resolved_config_json, status, device, env_json, log_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    status=excluded.status,
                    device=excluded.device,
                    env_json=excluded.env_json,
                    log_path=excluded.log_path,
                    notes=excluded.notes,
                    resolved_config_json=excluded.resolved_config_json
                """,
                (
                    experiment_id,
                    created_at_utc or utc_now_iso(),
                    dataset_version_id,
                    manifest_sha256,
                    base_checkpoint,
                    base_checkpoint_sha256,
                    config_sha256,
                    json.dumps(resolved_config, sort_keys=True, default=str),
                    status,
                    device,
                    json.dumps(env or {}, sort_keys=True, default=str),
                    log_path,
                    notes,
                ),
            )
        return experiment_id

    def register_artifact(
        self,
        *,
        artifact_id: str,
        experiment_id: str,
        role: str,
        path: str,
        sha256: Optional[str] = None,
        size_bytes: Optional[int] = None,
        created_at_utc: Optional[str] = None,
    ) -> str:
        """Register a model artifact such as ``best.pt`` or ``last.pt``."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO model_artifact (artifact_id, experiment_id, role, path, sha256,
                    bytes, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id, role) DO UPDATE SET
                    path=excluded.path,
                    sha256=excluded.sha256,
                    bytes=excluded.bytes,
                    created_at_utc=excluded.created_at_utc
                """,
                (
                    artifact_id,
                    experiment_id,
                    role,
                    path,
                    sha256,
                    size_bytes,
                    created_at_utc or utc_now_iso(),
                ),
            )
        return artifact_id

    def record_metric(
        self,
        *,
        metric_id: str,
        experiment_id: str,
        metric_name: str,
        value: Optional[float],
        dataset_version_id: Optional[str] = None,
        split: Optional[str] = None,
        threshold: Optional[float] = None,
        evaluated_at_utc: Optional[str] = None,
    ) -> str:
        """Record one evaluation metric."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO metric (metric_id, experiment_id, dataset_version_id, split,
                    metric_name, value, threshold, evaluated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(metric_id) DO UPDATE SET
                    value=excluded.value,
                    threshold=excluded.threshold,
                    evaluated_at_utc=excluded.evaluated_at_utc
                """,
                (
                    metric_id,
                    experiment_id,
                    dataset_version_id,
                    split,
                    metric_name,
                    value,
                    threshold,
                    evaluated_at_utc or utc_now_iso(),
                ),
            )
        return metric_id

    def list_experiments(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiment ORDER BY created_at_utc DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self, experiment_id: Optional[str] = None) -> list[dict]:
        if experiment_id:
            rows = self.conn.execute(
                "SELECT * FROM model_artifact WHERE experiment_id = ? ORDER BY role",
                (experiment_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM model_artifact ORDER BY role").fetchall()
        return [dict(row) for row in rows]

    def list_metrics(self, experiment_id: Optional[str] = None) -> list[dict]:
        if experiment_id:
            rows = self.conn.execute(
                "SELECT * FROM metric WHERE experiment_id = ? ORDER BY metric_name",
                (experiment_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM metric ORDER BY metric_name").fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict:
        """Row counts per table, for the ``catalog stats`` command."""
        tables = (
            "dataset_version",
            "source_asset",
            "image",
            "split_assignment",
            "validation_report",
            "experiment",
            "model_artifact",
            "metric",
        )
        out: dict[str, Any] = {"catalog_path": str(self.path), "schema_version": self.version}
        for table in tables:
            out[table] = int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return out
