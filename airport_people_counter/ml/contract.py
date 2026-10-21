"""Versioned data contract for the people-detection dataset.

This module is the single source of truth for *what a dataset record means*:
identifiers, hashes, the class schema, annotation status, and timestamp
representation. Every other module in :mod:`ml` builds on the primitives
defined here, so the contract can be versioned in one place.

Design rules enforced by this module:

* **Content addressing.** An image is identified by the SHA-256 of its
  bytes. The same picture therefore has the same ``image_id`` no matter
  where it lives on disk, and two files with the same ``image_id`` *are*
  duplicates by definition.
* **UTC everywhere.** Timestamps are ISO-8601 with a trailing ``Z`` and
  millisecond precision, matching the runtime convention in
  ``core/models.py`` (``utc_now_iso``) and ``app/queue_prototype.py``
  (``iso_from_unix``).
* **Never invent provenance.** Unknown values are ``None`` or an explicit
  ``"unknown"``; they are never guessed, defaulted to a plausible value,
  or silently back-filled.
* **Machine output is not ground truth.** ``AnnotationStatus`` keeps
  human-reviewed annotations strictly separate from model predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

#: Version of the data contract itself (this file's semantics).
CONTRACT_VERSION = "1.0.0"

#: Version of the class schema. Bump when class ids/names change.
CLASS_SCHEMA_VERSION = "1.0.0"

#: Canonical hash chunk length used by all generated identifiers.
_ID_HASH_LEN = 16

#: Number of bytes in a perceptual-hash comparison window (see ml.validate).
PHASH_SIDE = 8


class AnnotationStatus(str, Enum):
    """Provenance of an annotation.

    Only ``VERIFIED`` and ``NEGATIVE_REVIEWED`` are ground truth. A
    ``MACHINE_PROPOSED`` label is a model prediction that has not been
    checked by a human; it must never be treated as ground truth.
    """

    VERIFIED = "verified"
    NEGATIVE_REVIEWED = "negative_reviewed"
    MACHINE_PROPOSED = "machine_proposed"
    UNREVIEWED = "unreviewed"

    @property
    def is_ground_truth(self) -> bool:
        return self in (AnnotationStatus.VERIFIED, AnnotationStatus.NEGATIVE_REVIEWED)


class SourceKind(str, Enum):
    """Where a raw asset came from."""

    VIDEO = "video"
    IMAGE = "image"
    CAMERA = "camera"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class SplitName(str, Enum):
    """Canonical split names used by the manifest and the YOLOv5 data.yaml."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class ClassDef:
    """A single class in the dataset's class schema."""

    id: int
    name: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}


#: The project detects exactly one class: COCO class 0, ``person``. The
#: runtime filters on ``PERSON_CLASS_ID = 0`` (app/detect_people.py), so the
#: dataset uses a single-class schema and the model is trained with
#: ``nc=1``. This keeps mAP measured on the task that actually matters.
DEFAULT_CLASS_SCHEMA: tuple[ClassDef, ...] = (ClassDef(0, "person"),)


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing ``Z`` (millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (``Z`` suffix accepted) into an aware datetime."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to a canonical JSON string (stable key order, no spaces).

    Two structurally equal objects always produce the same string, which is
    what makes content hashes of manifests and configurations meaningful.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def to_signed64(value: int) -> int:
    """Map an unsigned 64-bit value into the signed 64-bit range.

    SQLite's ``INTEGER`` is a *signed* 64-bit integer, so a raw 64-bit
    perceptual hash with the top bit set cannot be stored. The conversion is
    a pure bit reinterpretation, so XOR-based comparisons (Hamming
    distance) are unaffected.
    """
    masked = int(value) & ((1 << 64) - 1)
    return masked - (1 << 64) if masked >= (1 << 63) else masked


def sha256_json(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of ``obj``."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks so large videos are safe."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_image_id(content_sha256: str) -> str:
    """Stable content-addressed identifier for an image."""
    return f"img_{content_sha256[:_ID_HASH_LEN]}"


def make_annotation_id(image_id: str, class_id: int, box: Sequence[float]) -> str:
    """Stable identifier for one annotation (image + class + normalised box)."""
    payload = canonical_json(
        {"image_id": image_id, "class_id": int(class_id), "box": [round(float(v), 6) for v in box]}
    )
    return f"ann_{sha256_bytes(payload.encode('utf-8'))[:_ID_HASH_LEN]}"


def make_dataset_version_id(manifest_body: dict) -> str:
    """Stable identifier for a dataset version, derived from its manifest body."""
    return f"dsv_{sha256_json(manifest_body)[:_ID_HASH_LEN]}"


def make_experiment_id(config_sha256: str, created_at_utc: str) -> str:
    """Stable identifier for a training experiment."""
    stamp = created_at_utc.replace("-", "").replace(":", "").replace(".", "")
    return f"exp_{stamp}_{config_sha256[:8]}"


def make_run_id(prefix: str = "run") -> str:
    """Short unique identifier for a collection / mining run."""
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


@dataclass
class ImageRecord:
    """One curated image plus everything known about its provenance.

    ``source_group`` is the leakage-control key: all frames captured from
    the same video or recording session share a group id so they are never
    spread across train/val/test.
    """

    image_id: str
    relpath: str
    sha256: str
    width: int = 0
    height: int = 0
    source_group: Optional[str] = None
    source_ref: Optional[str] = None
    source_kind: str = SourceKind.UNKNOWN.value
    captured_at_utc: Optional[str] = None
    license: Optional[str] = None
    usage_restrictions: Optional[str] = None
    annotation_status: str = AnnotationStatus.UNREVIEWED.value
    class_schema_version: str = CLASS_SCHEMA_VERSION
    preprocessing: Optional[str] = None
    review_note: Optional[str] = None
    phash: Optional[int] = None
    split: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "relpath": self.relpath,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "source_group": self.source_group,
            "source_ref": self.source_ref,
            "source_kind": self.source_kind,
            "captured_at_utc": self.captured_at_utc,
            "license": self.license,
            "usage_restrictions": self.usage_restrictions,
            "annotation_status": self.annotation_status,
            "class_schema_version": self.class_schema_version,
            "preprocessing": self.preprocessing,
            "review_note": self.review_note,
            "phash": self.phash,
            "split": self.split,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload["extra"] = data.get("extra") or {}
        return cls(**payload)


def validate_class_schema(schema: Iterable[ClassDef]) -> list[str]:
    """Return a list of problems with a class schema (empty list == valid)."""
    problems: list[str] = []
    items = list(schema)
    if not items:
        problems.append("class schema is empty")
        return problems
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in items:
        if not isinstance(item.id, int) or item.id < 0:
            problems.append(f"class id must be a non-negative integer, got {item.id!r}")
        if item.id in seen_ids:
            problems.append(f"duplicate class id {item.id}")
        seen_ids.add(item.id)
        if not item.name or not str(item.name).strip():
            problems.append(f"class {item.id} has an empty name")
        if item.name in seen_names:
            problems.append(f"duplicate class name {item.name!r}")
        seen_names.add(item.name)
    return problems


def class_schema_to_dict(schema: Iterable[ClassDef]) -> dict:
    """Serialise a class schema for storage in manifests and the catalog."""
    items = list(schema)
    return {
        "version": CLASS_SCHEMA_VERSION,
        "classes": [item.to_dict() for item in items],
    }


def class_schema_from_dict(data: dict) -> tuple[ClassDef, ...]:
    """Parse a serialised class schema back into :class:`ClassDef` objects."""
    classes = (data or {}).get("classes") or []
    return tuple(ClassDef(int(c["id"]), str(c["name"])) for c in classes)
