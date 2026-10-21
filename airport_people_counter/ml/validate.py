"""Dataset and annotation validation.

Checks a directory laid out as a YOLOv5 detection dataset::

    <root>/images/{train,val,test}/*.jpg
    <root>/labels/{train,val,test}/*.txt

and reports every problem it can find, without ever modifying the data.

Two rules shape the behaviour:

* **A valid empty label file is not an error.** An empty label file means
  "this image was reviewed and contains no person". That is only treated as
  a reviewed negative when the manifest says so; an undeclared empty label
  is reported as a warning so a human decides.
* **Nothing is repaired silently.** Corrupt images, out-of-range class ids
  and malformed rows are reported, never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from .contract import (
    AnnotationStatus,
    ClassDef,
    DEFAULT_CLASS_SCHEMA,
    CLASS_SCHEMA_VERSION,
    ImageRecord,
    make_image_id,
    sha256_file,
    to_signed64,
    utc_now_iso,
)

#: Image formats accepted by the vendored YOLOv5 loader.
SUPPORTED_IMAGE_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)

#: Minimum plausible side length for a person-detection image.
MIN_IMAGE_SIDE = 32

#: Boxes smaller than this (normalised area) are treated as implausible.
MIN_BOX_AREA = 1e-6

#: Hamming distance below which two perceptual hashes count as near-duplicates.
NEAR_DUPLICATE_DISTANCE = 4


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """A single validation finding."""

    code: str
    severity: Severity
    message: str
    path: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class LabelBox:
    """One YOLO annotation row: class id plus normalised xywh centre box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int = 0

    def as_list(self) -> list[float]:
        return [self.class_id, self.x_center, self.y_center, self.width, self.height]

    def area(self) -> float:
        return self.width * self.height


@dataclass
class ValidationReport:
    """Machine-readable result of validating one dataset root."""

    root: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    class_schema_version: str = CLASS_SCHEMA_VERSION
    findings: list[Finding] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def add(self, code: str, severity: Severity, message: str, path: Optional[str] = None,
            detail: Optional[str] = None) -> None:
        self.findings.append(Finding(code, severity, message, path, detail))

    def of_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def errors(self) -> list[Finding]:
        return self.of_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[Finding]:
        return self.of_severity(Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when no error-level finding was recorded."""
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "created_at_utc": self.created_at_utc,
            "class_schema_version": self.class_schema_version,
            "ok": self.ok,
            "counts": dict(self.counts),
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.of_severity(Severity.INFO)),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def render_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Dataset validation report: {self.root}",
            f"  created_at : {self.created_at_utc}",
            f"  class schema: {self.class_schema_version}",
            f"  status     : {'PASS' if self.ok else 'FAIL'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)",
        ]
        for key, value in self.counts.items():
            lines.append(f"  {key:<12}: {value}")
        if self.findings:
            lines.append("  findings:")
            for finding in self.findings:
                location = f" [{finding.path}]" if finding.path else ""
                lines.append(f"    - {finding.severity.value.upper():<7} {finding.code}: "
                             f"{finding.message}{location}")
        return "\n".join(lines)


def parse_label_file(path, class_schema: Iterable[ClassDef]) -> tuple[list[LabelBox], list[Finding]]:
    """Parse one YOLO label file.

    Returns ``(boxes, findings)``. Malformed rows are reported and skipped;
    a row is malformed when it does not have exactly 5 whitespace-separated
    fields or when the numeric fields cannot be parsed.
    """
    findings: list[Finding] = []
    boxes: list[LabelBox] = []
    valid_ids = {cls.id for cls in class_schema}
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        findings.append(Finding("label_unreadable", Severity.ERROR, f"cannot read label file: {exc}", str(source)))
        return boxes, findings

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            findings.append(Finding(
                "label_malformed_row",
                Severity.ERROR,
                f"expected 5 whitespace-separated fields, found {len(parts)}",
                str(source),
                f"line {line_number}: {line!r}",
            ))
            continue
        try:
            class_value = float(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            findings.append(Finding(
                "label_non_numeric",
                Severity.ERROR,
                "label row contains a non-numeric value",
                str(source),
                f"line {line_number}: {line!r}",
            ))
            continue

        if class_value != int(class_value):
            findings.append(Finding(
                "label_non_integer_class",
                Severity.ERROR,
                f"class id must be an integer, got {class_value}",
                str(source),
                f"line {line_number}",
            ))
            continue
        class_id = int(class_value)
        if class_id not in valid_ids:
            findings.append(Finding(
                "label_class_out_of_range",
                Severity.ERROR,
                f"class id {class_id} is not in the class schema {sorted(valid_ids)}",
                str(source),
                f"line {line_number}",
            ))
            continue

        x_center, y_center, width, height = coords
        if any(v < 0.0 or v > 1.0 for v in coords):
            findings.append(Finding(
                "label_coords_out_of_bounds",
                Severity.ERROR,
                f"normalised coordinates must be within [0, 1], got {coords}",
                str(source),
                f"line {line_number}",
            ))
            continue

        box = LabelBox(class_id, x_center, y_center, width, height, line_number)
        if box.area() <= MIN_BOX_AREA:
            findings.append(Finding(
                "label_zero_area_box",
                Severity.ERROR,
                f"box has zero or implausible area ({box.area():.3e})",
                str(source),
                f"line {line_number}",
            ))
            continue
        if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6 \
                or y_center - height / 2 < -1e-6 or y_center + height / 2 > 1 + 1e-6:
            findings.append(Finding(
                "label_box_outside_image",
                Severity.ERROR,
                "box extends outside the image bounds",
                str(source),
                f"line {line_number}",
            ))
            continue
        boxes.append(box)
    return boxes, findings


def perceptual_hash(path) -> Optional[int]:
    """64-bit difference hash of an image, or ``None`` if it cannot be read.

    Used only to flag *possible* near-duplicates; a human confirms them.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency of the project
        return None
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(grayscale.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return to_signed64(bits)


def hamming_distance(left: Optional[int], right: Optional[int]) -> Optional[int]:
    """Bit distance between two perceptual hashes (``None`` if either is missing)."""
    if left is None or right is None:
        return None
    return bin(left ^ right).count("1")


def _image_files(directory: Path) -> list[Path]:
    """Every image under ``directory``, including source-group sub-folders.

    The walk is recursive because a dataset is expected to keep the frames
    of one recording session together in a sub-directory
    (``images/<split>/<group>/<file>.jpg``). The mirror layout under
    ``labels/`` is used to find the matching label file.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_FORMATS
    )


def _label_files(directory: Path) -> list[Path]:
    """Every label file under ``directory`` -- plain ``.txt`` files, recursively."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() == ".txt"
    )


def _label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    """The label file that belongs to ``image_path``.

    The path inside the images directory is mirrored into the labels
    directory, so ``images/train/session_a/f.jpg`` pairs with
    ``labels/train/session_a/f.txt``.
    """
    relative = image_path.relative_to(image_root)
    return label_root / relative.with_suffix(".txt")


def _read_image_meta(path: Path) -> tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Return ``(width, height, format, mode)`` for an image, or Nones on failure."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None, None, None, None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width, image.height, (image.format or "").lower(), image.mode
    except Exception:
        return None, None, None, None


def validate_dataset(
    root,
    class_schema: Iterable[ClassDef] = DEFAULT_CLASS_SCHEMA,
    splits: Iterable[str] = ("train", "val", "test"),
    check_images: bool = True,
    check_near_duplicates: bool = True,
    annotation_status: str = AnnotationStatus.UNREVIEWED.value,
) -> tuple[ValidationReport, list[ImageRecord]]:
    """Validate a dataset root and collect the image records it describes.

    Returns ``(report, records)``. The records are the curated image
    metadata that the manifest builder consumes; they carry the content
    hash, dimensions and perceptual hash when those could be determined.
    """
    schema = tuple(class_schema)
    base = Path(root)
    report = ValidationReport(root=str(base))

    records: list[ImageRecord] = []
    seen_by_hash: dict[str, ImageRecord] = {}
    split_paths: dict[str, list[Path]] = {}

    for split in splits:
        image_dir = base / "images" / split
        label_dir = base / "labels" / split
        image_files = _image_files(image_dir)
        split_paths[split] = image_files

        if not image_dir.is_dir():
            report.add("split_images_dir_missing", Severity.WARNING,
                       f"split {split!r} has no images directory", str(image_dir))
            continue
        if not image_files:
            report.add("split_empty", Severity.WARNING,
                       f"split {split!r} contains no images", str(image_dir))

        label_stems = {
            str(p.relative_to(label_dir).with_suffix(""))
            for p in _label_files(label_dir)
        }
        image_stems = {
            str(p.relative_to(image_dir).with_suffix(""))
            for p in image_files
        }

        for stem in sorted(image_stems - label_stems):
            report.add("missing_label", Severity.ERROR,
                       "image has no matching label file",
                       str(label_dir / f"{stem}.txt"))

        for stem in sorted(label_stems - image_stems):
            report.add("missing_image", Severity.ERROR,
                       "label file has no matching image",
                       str(label_dir / f"{stem}.txt"))

        for image_path in image_files:
            label_path = _label_path_for(image_path, image_dir, label_dir)
            relpath = str(image_path.relative_to(base))

            width = height = 0
            image_format = None
            mode = None
            if check_images:
                width, height, image_format, mode = _read_image_meta(image_path)
                if width is None:
                    report.add("image_unreadable", Severity.ERROR,
                               "image could not be opened or is corrupt", relpath)
                else:
                    if image_format not in SUPPORTED_IMAGE_FORMATS:
                        report.add("image_unsupported_format", Severity.ERROR,
                                   f"unsupported image format {image_format!r}", relpath)
                    if min(width, height) < MIN_IMAGE_SIDE:
                        report.add("image_too_small", Severity.WARNING,
                                   f"image is {width}x{height}, smaller than {MIN_IMAGE_SIDE}px", relpath)
                    if mode not in ("RGB", "L", "P"):
                        report.add("image_unusual_mode", Severity.INFO,
                                   f"image colour mode is {mode!r}", relpath)

            if not label_path.exists():
                status = AnnotationStatus.UNREVIEWED.value
            else:
                status = annotation_status
                try:
                    body = label_path.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    report.add("label_unreadable", Severity.ERROR,
                               f"cannot read label file: {exc}", str(label_path.relative_to(base)))
                    body = ""
                if body == "":
                    if status == AnnotationStatus.UNREVIEWED.value:
                        report.add("empty_label_unclassified", Severity.WARNING,
                                   "label file is empty; declare it as a reviewed negative "
                                   "or add annotations", str(label_path.relative_to(base)))
                    else:
                        report.add("empty_label_reviewed_negative", Severity.INFO,
                                   "empty label file recorded as a reviewed negative",
                                   str(label_path.relative_to(base)))
                else:
                    _, findings = parse_label_file(label_path, schema)
                    for finding in findings:
                        report.findings.append(finding)

            digest = sha256_file(image_path)
            record = ImageRecord(
                image_id=make_image_id(digest),
                relpath=relpath,
                sha256=digest,
                width=width or 0,
                height=height or 0,
                source_group=_group_from_path(image_path, image_dir),
                source_kind="image",
                annotation_status=status,
                class_schema_version=CLASS_SCHEMA_VERSION,
                phash=perceptual_hash(image_path) if check_near_duplicates else None,
                extra={"split": split, "format": image_format, "mode": mode},
            )
            if digest in seen_by_hash:
                report.add("duplicate_image", Severity.ERROR,
                           f"identical content already present as {seen_by_hash[digest].relpath}",
                           relpath)
            else:
                seen_by_hash[digest] = record
            records.append(record)

    # near-duplicate detection (within the whole dataset)
    if check_near_duplicates:
        hashed = [r for r in records if r.phash is not None]
        for i in range(len(hashed)):
            for j in range(i + 1, len(hashed)):
                distance = hamming_distance(hashed[i].phash, hashed[j].phash)
                if distance is not None and distance <= NEAR_DUPLICATE_DISTANCE:
                    report.add("near_duplicate_image", Severity.WARNING,
                               f"perceptual hash distance {distance} from {hashed[j].relpath}",
                               hashed[i].relpath)

    # dimension consistency
    dimensions = {(r.width, r.height) for r in records if r.width and r.height}
    if len(dimensions) > 1:
        report.add("inconsistent_dimensions", Severity.WARNING,
                   f"{len(dimensions)} distinct image sizes present: {sorted(dimensions)[:8]}")

    report.counts = {
        "splits": len(split_paths),
        "images": len(records),
        "unique_images": len(seen_by_hash),
        "errors": len(report.errors),
        "warnings": len(report.warnings),
    }
    return report, records


def _group_from_path(image_path: Path, image_root: Path) -> Optional[str]:
    """Derive a source group from the directory structure.

    ``<root>/images/<split>/<group>/<file>.jpg`` yields the group name
    ``<group>``; images sitting directly in the split directory have no
    group and are reported as orphans by the splitter.
    """
    relative = image_path.relative_to(image_root)
    parts = relative.parts
    if len(parts) >= 2:
        return parts[0]
    return None
