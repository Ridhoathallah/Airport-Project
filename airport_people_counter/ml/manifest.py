"""Dataset manifests: the authoritative, versioned description of a dataset.

A manifest is a single JSON document that fully describes one dataset
version: every image (with its content hash and provenance), the split
assignment, the class schema, the seed and split policy, and a hash of
all of that.

The manifest -- not the directory layout, and not the metadata catalog --
is the source of truth. The catalog (see :mod:`ml.catalog`) indexes
manifests so experiments can be looked up; it never asserts that a
dataset is immutable. Immutability is *verified* by recomputing hashes
with :func:`verify_manifest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .contract import (
    CLASS_SCHEMA_VERSION,
    CONTRACT_VERSION,
    ImageRecord,
    class_schema_from_dict,
    class_schema_to_dict,
    make_dataset_version_id,
    sha256_json,
    utc_now_iso,
)

MANIFEST_VERSION = "1.0.0"

#: Readiness status of a dataset version.
#:
#: ``ready``   -- the version contains images and can be trained on.
#: ``empty``   -- the version contains no images. It exists so that an
#:                empty dataset is *visible* rather than silently missing,
#:                but it is NOT training-ready: the training preflight
#:                refuses it.
STATUS_READY = "ready"
STATUS_EMPTY = "empty"

#: Fields that are recorded in the manifest but deliberately excluded from
#: its hash. ``created_at_utc`` is excluded so that the dataset version id
#: is *content-derived*: building the same dataset twice with the same seed
#: yields the same id, which is what makes an experiment reproducible. The
#: timestamp is still written to the file for audit.
HASH_EXCLUDED_FIELDS = ("created_at_utc",)


class ManifestError(Exception):
    """Raised when a manifest cannot be parsed or fails verification."""


@dataclass
class Manifest:
    """One immutable-by-verification dataset version."""

    dataset_version: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    contract_version: str = CONTRACT_VERSION
    class_schema_version: str = CLASS_SCHEMA_VERSION
    class_schema: dict = field(default_factory=dict)
    #: Dataset root, expressed relative to the directory holding the manifest.
    root: str = "."
    seed: Optional[int] = None
    split_policy: dict = field(default_factory=dict)
    images: list[ImageRecord] = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    notes: str = ""
    manifest_sha256: Optional[str] = None
    #: ``ready`` or ``empty`` -- see :data:`STATUS_READY` / :data:`STATUS_EMPTY`.
    status: str = STATUS_READY

    @property
    def is_training_ready(self) -> bool:
        """True only when the version holds images and is marked ready."""
        return self.status == STATUS_READY and bool(self.images)

    # ── identity ────────────────────────────────────────────
    def body(self) -> dict:
        """The full manifest body, as written to disk."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "contract_version": self.contract_version,
            "class_schema_version": self.class_schema_version,
            "class_schema": self.class_schema,
            "dataset_version": self.dataset_version,
            "created_at_utc": self.created_at_utc,
            "root": self.root,
            "seed": self.seed,
            "split_policy": self.split_policy,
            "images": [record.to_dict() for record in self.images],
            "splits": self.splits,
            "notes": self.notes,
            "status": self.status,
        }

    def hash_body(self) -> dict:
        """The subset of the body the manifest hash is computed over."""
        return {k: v for k, v in self.body().items() if k not in HASH_EXCLUDED_FIELDS}

    def compute_sha256(self) -> str:
        """Recompute the manifest hash from its hashable body."""
        return sha256_json(self.hash_body())

    @property
    def dataset_version_id(self) -> str:
        """Content-derived identifier for this dataset version."""
        return make_dataset_version_id(self.hash_body())

    # ── serialisation ───────────────────────────────────────
    def to_dict(self) -> dict:
        data = self.body()
        data["dataset_version_id"] = self.dataset_version_id
        data["manifest_sha256"] = self.manifest_sha256 or self.compute_sha256()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if "dataset_version" not in data:
            raise ManifestError("manifest is missing 'dataset_version'")
        manifest = cls(
            dataset_version=str(data["dataset_version"]),
            created_at_utc=str(data.get("created_at_utc") or utc_now_iso()),
            contract_version=str(data.get("contract_version") or CONTRACT_VERSION),
            class_schema_version=str(data.get("class_schema_version") or CLASS_SCHEMA_VERSION),
            class_schema=data.get("class_schema") or {},
            root=str(data.get("root") or "."),
            seed=data.get("seed"),
            split_policy=data.get("split_policy") or {},
            images=[ImageRecord.from_dict(item) for item in data.get("images") or []],
            splits=data.get("splits") or {},
            notes=str(data.get("notes") or ""),
            manifest_sha256=data.get("manifest_sha256"),
            status=str(data.get("status") or STATUS_READY),
        )
        return manifest

    # ── queries ─────────────────────────────────────────────
    def records_by_split(self, split: str) -> list[ImageRecord]:
        ids = set(self.splits.get(split) or [])
        return [r for r in self.images if r.image_id in ids]

    def groups(self) -> dict:
        """Map ``source_group`` -> list of image ids (``None`` key for orphans)."""
        grouped: dict = {}
        for record in self.images:
            grouped.setdefault(record.source_group, []).append(record.image_id)
        return grouped


def build_manifest(
    dataset_version: str,
    images: list[ImageRecord],
    splits: dict,
    class_schema: tuple,
    seed: Optional[int] = None,
    split_policy: Optional[dict] = None,
    root: str = ".",
    notes: str = "",
    status: Optional[str] = None,
) -> Manifest:
    """Assemble a manifest from records and a split assignment.

    ``status`` defaults to :data:`STATUS_EMPTY` when there are no images,
    so an empty dataset can never be written out as a training-ready
    version by accident.
    """
    if status is None:
        status = STATUS_READY if images else STATUS_EMPTY
    manifest = Manifest(
        dataset_version=dataset_version,
        class_schema=class_schema_to_dict(class_schema),
        images=list(images),
        splits={k: list(v) for k, v in splits.items()},
        seed=seed,
        split_policy=dict(split_policy or {}),
        root=root,
        notes=notes,
        status=status,
    )
    manifest.manifest_sha256 = manifest.compute_sha256()
    return manifest


def write_manifest(path, manifest: Manifest) -> Path:
    """Write a manifest to ``path`` as pretty-printed JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def load_manifest(path) -> Manifest:
    """Read a manifest from ``path``."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {source}: {exc}") from exc
    return Manifest.from_dict(data)


def verify_manifest(path, check_files: bool = True) -> tuple[bool, list[str]]:
    """Verify a manifest against its recorded hash and (optionally) its files.

    Returns ``(ok, problems)``. ``check_files`` re-hashes every referenced
    image on disk; a mismatch means the dataset was modified after the
    manifest was written.
    """
    problems: list[str] = []
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        return False, [str(exc)]

    expected = manifest.manifest_sha256
    actual = manifest.compute_sha256()
    if expected != actual:
        problems.append(
            f"manifest hash mismatch: recorded {expected!r}, recomputed {actual!r}"
        )

    if check_files:
        # ``root`` is the dataset root expressed relative to the directory
        # that holds this manifest.
        root = Path(path).parent / manifest.root
        for record in manifest.images:
            target = root / record.relpath
            if not target.exists():
                problems.append(f"missing image file: {record.relpath}")
                continue
            from .contract import sha256_file

            digest = sha256_file(target)
            if digest != record.sha256:
                problems.append(
                    f"content hash mismatch for {record.relpath}: "
                    f"recorded {record.sha256[:12]}…, on disk {digest[:12]}…"
                )
    return (not problems), problems


def dataset_version_id(manifest: Manifest) -> str:
    """Convenience wrapper for :attr:`Manifest.dataset_version_id`."""
    return manifest.dataset_version_id
