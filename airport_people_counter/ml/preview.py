"""Optional visual preview of annotations for human review.

Rendering a picture is often the fastest way to catch a systematic label
error that a numeric validator cannot see (a swapped class, a box drawn
around the wrong object, a mirrored image). The preview is a review
aid only -- it never changes a label file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .validate import LabelBox

#: BGR colours cycled through when a class schema has several classes.
PALETTE = (
    (0, 230, 118),
    (0, 200, 255),
    (0, 140, 255),
    (255, 0, 200),
    (0, 255, 255),
    (200, 0, 255),
)

CLASS_NAMES = {0: "person"}


def _colour(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def render_preview(
    image_path,
    boxes: Iterable[LabelBox],
    out_path,
    class_names: Optional[dict] = None,
    thickness: int = 2,
) -> Path:
    """Draw ``boxes`` onto a copy of ``image_path`` and write it to ``out_path``."""
    import cv2

    names = class_names or CLASS_NAMES
    source = Path(image_path)
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"cannot read image: {source}")
    height, width = frame.shape[:2]

    for box in boxes:
        x1 = int((box.x_center - box.width / 2) * width)
        y1 = int((box.y_center - box.height / 2) * height)
        x2 = int((box.x_center + box.width / 2) * width)
        y2 = int((box.y_center + box.height / 2) * height)
        colour = _colour(box.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
        label = names.get(box.class_id, str(box.class_id))
        cv2.putText(
            frame, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), frame)
    return target


def preview_dataset(root, split: str, out_dir, limit: int = 20,
                    class_names: Optional[dict] = None) -> list:
    """Render previews for up to ``limit`` images of one split."""
    from .validate import parse_label_file
    from .contract import DEFAULT_CLASS_SCHEMA

    base = Path(root)
    image_dir = base / "images" / split
    label_dir = base / "labels" / split
    out_root = Path(out_dir)
    written: list = []
    if not image_dir.is_dir():
        return written
    for image_path in sorted(image_dir.iterdir()):
        if len(written) >= limit:
            break
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = []
        if label_path.exists():
            boxes, _ = parse_label_file(label_path, DEFAULT_CLASS_SCHEMA)
        target = out_root / f"{image_path.stem}.jpg"
        render_preview(image_path, boxes, target, class_names=class_names)
        written.append(str(target))
    return written
