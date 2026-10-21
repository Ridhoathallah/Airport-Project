"""Training preflight: refuse to train on data that cannot produce a result.

``yolov5/train.py`` is happy to train on a dataset with no images, with
missing label files or with out-of-range class ids -- it simply produces
a useless checkpoint. This module runs the checks *before* the trainer is
invoked and refuses the run with an actionable message.

The checks are:

1. the dataset YAML exists and parses, and names the train/val splits;
2. the dataset root resolves to a real directory;
3. the train and val splits contain images (an empty split is a failure,
   not a warning, at training time);
4. every image has a label file, and every label row is valid YOLO
   (class id in the schema, normalised coordinates in range, non-zero
   area);
5. a dataset manifest exists for this root and its recorded hashes still
   match the files on disk.

Nothing here writes to the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .contract import DEFAULT_CLASS_SCHEMA, ClassDef
from .validate import Severity, parse_label_file

#: Where dataset manifests live, relative to the project root.
MANIFEST_DIR = Path("datasets/manifests")


@dataclass
class Check:
    """One preflight check result."""

    name: str
    ok: bool
    detail: str
    fatal: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "fatal": self.fatal}


@dataclass
class PreflightResult:
    """The outcome of a training preflight."""

    data_yaml: str
    dataset_root: Optional[str] = None
    checks: list = field(default_factory=list)
    split_counts: dict = field(default_factory=dict)
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    dataset_version_id: Optional[str] = None
    class_schema: dict = field(default_factory=dict)

    @property
    def errors(self) -> list:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> list:
        return [c for c in self.checks if not c.ok and not c.fatal]

    @property
    def ok(self) -> bool:
        """True when no fatal check failed."""
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "data_yaml": self.data_yaml,
            "dataset_root": self.dataset_root,
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "split_counts": dict(self.split_counts),
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "dataset_version_id": self.dataset_version_id,
            "class_schema": dict(self.class_schema),
        }

    def render_text(self) -> str:
        lines = [f"Training preflight: {self.data_yaml}"]
        for check in self.checks:
            mark = "OK  " if check.ok else ("FAIL" if check.fatal else "WARN")
            lines.append(f"  [{mark}] {check.name}: {check.detail}")
        if self.split_counts:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(self.split_counts.items()))
            lines.append(f"  split counts: {counts}")
        lines.append(f"  result: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


def _resolve_data_yaml(data_yaml) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """Read and resolve a YOLOv5 ``data.yaml``.

    Returns ``(data, dataset_root, error)``. ``path`` is resolved the way
    ``yolov5/utils/general.check_dataset`` resolves it: a relative path
    is taken relative to the vendored YOLOv5 root, an absolute path is
    used as-is.
    """
    import yaml

    path = Path(data_yaml)
    if not path.is_file():
        return None, None, f"dataset YAML not found: {path}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return None, None, f"dataset YAML is not valid YAML: {exc}"
    if not isinstance(data, dict):
        return None, None, "dataset YAML does not contain a mapping"

    raw_path = data.get("path")
    if raw_path:
        root = Path(raw_path)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / "yolov5" / root).resolve()
        data["_resolved_root"] = str(root)
    else:
        data["_resolved_root"] = str(path.parent.resolve())
    return data, data["_resolved_root"], None


def _split_dir(root: str, data: dict, key: str) -> Optional[Path]:
    value = data.get(key)
    if not value:
        return None
    if isinstance(value, list):
        value = value[0]
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return candidate


def _label_root(root: str, data: dict, key: str) -> Optional[Path]:
    """The labels directory that mirrors one split.

    ``images/train`` pairs with ``labels/train``; the split name is the
    last path component of the images path, so the label layout does not
    depend on how deeply the images are nested.
    """
    directory = _split_dir(root, data, key)
    if directory is None:
        return None
    return Path(root) / "labels" / directory.name


def _class_schema_from_data(data: dict) -> tuple:
    names = data.get("names") or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not names:
        return DEFAULT_CLASS_SCHEMA
    try:
        return tuple(ClassDef(int(k), str(v)) for k, v in names.items())
    except (TypeError, ValueError):
        return DEFAULT_CLASS_SCHEMA


def _find_manifest(dataset_root: str, manifest_dir=None) -> Optional[Path]:
    """Find the manifest whose recorded root matches this dataset root."""
    directory = Path(manifest_dir) if manifest_dir else MANIFEST_DIR
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parent.parent / directory
    if not directory.is_dir():
        return None
    target = str(Path(dataset_root).resolve())
    for candidate in sorted(directory.glob("*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        root = data.get("root") or "."
        resolved = (candidate.parent / root).resolve()
        if str(resolved) == target:
            return candidate
    return None


def run_preflight(
    data_yaml,
    class_schema: Optional[tuple] = None,
    manifest_dir=None,
    require_manifest: bool = True,
) -> PreflightResult:
    """Run every training-readiness check and return the result."""
    result = PreflightResult(data_yaml=str(data_yaml))
    data, root, error = _resolve_data_yaml(data_yaml)

    if error:
        result.checks.append(Check("data_yaml", False, error))
        return result
    result.checks.append(Check("data_yaml", True, f"parsed {data_yaml}"))

    for key in ("train", "val"):
        if not data.get(key):
            result.checks.append(Check(f"data_yaml_{key}", False,
                                       f"'{key}:' is missing from the dataset YAML"))
    if not data.get("names"):
        result.checks.append(Check("data_yaml_names", False,
                                   "'names:' is missing from the dataset YAML"))
    else:
        result.checks.append(Check("data_yaml_names", True,
                                   f"{len(data.get('names') or {})} class(es) declared"))

    if not root or not Path(root).is_dir():
        result.checks.append(Check("dataset_root", False,
                                   f"dataset root does not exist: {root}"))
        return result
    result.dataset_root = root
    result.checks.append(Check("dataset_root", True, root))

    schema = class_schema or _class_schema_from_data(data)
    result.class_schema = {str(c.id): c.name for c in schema}

    # ── splits must contain images ──────────────────────────
    from .validate import _image_files, _label_path_for

    for key in ("train", "val"):
        directory = _split_dir(root, data, key)
        if directory is None:
            continue
        if not directory.is_dir():
            result.checks.append(Check(f"{key}_images", False,
                                       f"{key} directory does not exist: {directory}"))
            continue
        images = _image_files(directory)
        result.split_counts[key] = len(images)
        if not images:
            result.checks.append(Check(f"{key}_images", False,
                                       f"{key} split contains no images: {directory}"))
        else:
            result.checks.append(Check(f"{key}_images", True,
                                       f"{len(images)} image(s) in {key}"))

    # ── labels must exist and be valid ──────────────────────
    for key in ("train", "val"):
        directory = _split_dir(root, data, key)
        if directory is None or not directory.is_dir():
            continue
        images = _image_files(directory)
        label_dir = _label_root(root, data, key)
        if label_dir is None:
            continue
        missing = 0
        invalid = 0
        invalid_examples: list = []
        for image in images:
            label_path = _label_path_for(image, directory, label_dir)
            if not label_path.is_file():
                missing += 1
                continue
            boxes, findings = parse_label_file(label_path, schema)
            errors = [f for f in findings if f.severity is Severity.ERROR]
            if errors:
                invalid += 1
                if len(invalid_examples) < 3:
                    invalid_examples.append(f"{label_path.name}: {errors[0].code}")
        if not images:
            result.checks.append(Check(f"{key}_labels", False,
                                       f"{key} split has no images, so its labels "
                                       "cannot be checked", fatal=False))
        elif missing:
            result.checks.append(Check(f"{key}_labels", False,
                                       f"{missing} image(s) in {key} have no label file"))
        elif invalid:
            result.checks.append(Check(f"{key}_labels", False,
                                       f"{invalid} invalid label file(s) in {key}: "
                                       + "; ".join(invalid_examples)))
        else:
            result.checks.append(Check(f"{key}_labels", True,
                                       f"{len(images)} label file(s) parsed and valid in {key}"))

    # ── a manifest must exist and still match the files ──────
    manifest_path = _find_manifest(root, manifest_dir)
    if manifest_path is None:
        detail = ("no dataset manifest found for this root -- build the dataset "
                  "version first: scripts/build_dataset.py --root <root> --name <version>")
        result.checks.append(Check("manifest", False, detail, fatal=require_manifest))
    else:
        result.manifest_path = str(manifest_path)
        from .manifest import verify_manifest

        ok, problems = verify_manifest(manifest_path, check_files=True)
        if ok:
            data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result.manifest_sha256 = data_manifest.get("manifest_sha256")
            result.dataset_version_id = data_manifest.get("dataset_version_id")
            status = data_manifest.get("status") or "ready"
            if status != "ready":
                result.checks.append(Check(
                    "manifest", False,
                    f"manifest status is {status!r} -- this version is not training-ready",
                ))
            else:
                result.checks.append(Check("manifest", True,
                                           f"{manifest_path} verified against the files"))
        else:
            result.checks.append(Check("manifest", False,
                                       f"manifest verification failed: {problems[0]}"))

    return result
