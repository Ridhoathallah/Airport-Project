"""Hard-example mining: find frames worth a human's attention.

The workflow is deliberately one-directional:

1. run the **current** checkpoint over permitted source material;
2. record *candidates* -- frames whose detection statistics look like a
   known failure mode;
3. keep the machine output strictly separate from ground truth: every
   candidate is written with ``annotation_status = machine_proposed`` and
   is never promoted to a label automatically;
4. a human reviews the candidates, corrects them, and only then are the
   approved frames added to a new dataset version (which is re-validated
   and re-split afterwards).

Privacy and authorisation
-------------------------
Frames from a camera contain identifiable people. Retaining pixels is
therefore opt-in twice over: the caller must pass both
``retain_frames=True`` **and** ``consent_acknowledged=True``. Without
those flags only derived statistics (numbers, never images) are written.
The policy notice is exposed as :data:`PRIVACY_NOTICE` so the CLI can
print it verbatim.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .contract import AnnotationStatus, make_run_id, utc_now_iso

#: Printed before any collection run; states the handling requirements.
PRIVACY_NOTICE = (
    "PRIVACY / AUTHORISATION NOTICE\n"
    "  Frames captured from a camera contain identifiable people. Retaining\n"
    "  frame pixels therefore requires BOTH --retain-frames and\n"
    "  --consent-acknowledged, which assert that:\n"
    "    - the source material is permitted for this use,\n"
    "    - people in the frames have been informed or are otherwise lawful\n"
    "      to process for model improvement,\n"
    "    - the retention period below is acceptable.\n"
    "  Without both flags only derived statistics are written -- no images.\n"
    "  Collected frames are review material, not training data: they carry\n"
    "  annotation_status=machine_proposed and are never ground truth."
)

#: Default retention window for collected frames, in days.
DEFAULT_RETENTION_DAYS = 30

#: Failure-mode categories this miner can flag.
CATEGORY_CROWD_DENSE = "crowd_dense"
CATEGORY_SMALL_PERSON = "small_person"
CATEGORY_TRUNCATED = "truncated_body"
CATEGORY_LOW_CONFIDENCE = "low_confidence"
CATEGORY_NO_DETECTION = "no_detection"


@dataclass
class Detection:
    """One detection in original-frame pixel coordinates."""

    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


@dataclass
class Candidate:
    """A frame flagged for human review."""

    source_ref: str
    frame_index: int
    captured_at_utc: str
    categories: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    retained_path: Optional[str] = None
    annotation_status: str = AnnotationStatus.MACHINE_PROPOSED.value
    review_note: str = "machine-proposed candidate; requires human review before use"

    def to_dict(self) -> dict:
        return {
            "source_ref": self.source_ref,
            "frame_index": self.frame_index,
            "captured_at_utc": self.captured_at_utc,
            "categories": list(self.categories),
            "detections": list(self.detections),
            "stats": dict(self.stats),
            "retained_path": self.retained_path,
            "annotation_status": self.annotation_status,
            "review_note": self.review_note,
        }


@dataclass
class MiningResult:
    """Outcome of one mining run."""

    run_id: str
    source: str
    weights: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    frames_examined: int = 0
    candidates: list = field(default_factory=list)
    frames_retained: int = 0
    privacy: dict = field(default_factory=dict)
    out_dir: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "weights": self.weights,
            "created_at_utc": self.created_at_utc,
            "frames_examined": self.frames_examined,
            "candidate_count": len(self.candidates),
            "frames_retained": self.frames_retained,
            "privacy": dict(self.privacy),
            "out_dir": self.out_dir,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _load_detector(weights: str, device: str, conf_thres: float, iou_thres: float, classes):
    """Build a vendored-YOLOv5 detector. Imported lazily so the rest of the
    module works without torch installed."""
    project_root = Path(__file__).resolve().parent.parent
    yolov5_root = project_root / "yolov5"
    if str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))
    import torch
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes

    backend = DetectMultiBackend(weights, device=torch.device(device), data=None, fp16=False)
    backend.warmup(imgsz=(1, 3, 640, 640))
    return backend, non_max_suppression, scale_boxes


def detect_in_frame(
    frame,
    detector,
    non_max_suppression,
    scale_boxes,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    imgsz: int = 640,
) -> list:
    """Run the detector on one BGR frame and return :class:`Detection` objects."""
    import numpy as np
    import torch
    from utils.augmentations import letterbox

    im0 = frame
    im, ratio, pad = letterbox(im0, imgsz, auto=False, scaleup=False)
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
    im = torch.from_numpy(im).to(detector.device)
    im = im.half() if detector.fp16 else im.float()
    im /= 255
    if im.ndimension() == 3:
        im = im[None]
    pred = detector(im, augment=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes)
    out: list = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                out.append(Detection(
                    class_id=int(cls),
                    confidence=float(conf),
                    x1=float(xyxy[0]), y1=float(xyxy[1]),
                    x2=float(xyxy[2]), y2=float(xyxy[3]),
                ))
    return out


def classify_frame(
    detections: list,
    frame_shape,
    crowd_threshold: int = 8,
    small_area_ratio: float = 0.005,
    border_margin_px: float = 4.0,
    low_conf_band=(0.25, 0.50),
) -> tuple[list, dict]:
    """Assign failure-mode categories to a frame.

    Returns ``(categories, stats)``. These are *candidate* categories for
    prioritisation -- they are hypotheses about where the model struggles,
    not verified defect claims.
    """
    height, width = frame_shape[:2]
    frame_area = float(height * width)
    categories: list[str] = []
    confidences = [d.confidence for d in detections]
    stats = {
        "detection_count": len(detections),
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "min_confidence": min(confidences) if confidences else None,
        "max_confidence": max(confidences) if confidences else None,
        "median_box_area_ratio": None,
    }

    if len(detections) >= crowd_threshold:
        categories.append(CATEGORY_CROWD_DENSE)

    if detections:
        ratios = sorted(d.area() / frame_area for d in detections)
        median_ratio = ratios[len(ratios) // 2]
        stats["median_box_area_ratio"] = median_ratio
        if median_ratio < small_area_ratio:
            categories.append(CATEGORY_SMALL_PERSON)

    for detection in detections:
        touches = (
            detection.x1 <= border_margin_px or detection.y1 <= border_margin_px
            or detection.x2 >= width - border_margin_px or detection.y2 >= height - border_margin_px
        )
        if touches:
            categories.append(CATEGORY_TRUNCATED)
            break

    low, high = low_conf_band
    if any(low <= d.confidence < high for d in detections):
        categories.append(CATEGORY_LOW_CONFIDENCE)

    if not detections:
        categories.append(CATEGORY_NO_DETECTION)

    return sorted(set(categories)), stats


def mine_source(
    source,
    weights: str,
    out_dir=None,
    stride: int = 15,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cpu",
    imgsz: int = 640,
    crowd_threshold: int = 8,
    small_area_ratio: float = 0.005,
    retain_frames: bool = False,
    consent_acknowledged: bool = False,
    run_id: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> MiningResult:
    """Scan a video or image directory for hard examples.

    ``retain_frames`` only has an effect when ``consent_acknowledged`` is
    also true; otherwise no pixels are written.
    """
    import cv2

    run_id = run_id or make_run_id("mine")
    source_path = Path(source)
    result = MiningResult(
        run_id=run_id,
        source=str(source_path),
        weights=str(weights),
        privacy={
            "retain_frames": bool(retain_frames and consent_acknowledged),
            "consent_acknowledged": bool(consent_acknowledged),
            "retention_days": DEFAULT_RETENTION_DAYS,
            "annotation_status": AnnotationStatus.MACHINE_PROPOSED.value,
        },
    )

    frames_retained = bool(retain_frames and consent_acknowledged)
    if retain_frames and not consent_acknowledged:
        result.privacy["retain_frames"] = False
        result.privacy["note"] = (
            "retain_frames was requested without consent_acknowledged; "
            "no image pixels were written"
        )

    out_root = Path(out_dir) if out_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        result.out_dir = str(out_root)

    detector, nms, scale = _load_detector(weights, device, conf_thres, iou_thres, None)

    def handle_frame(frame, frame_index, source_ref):
        result.frames_examined += 1
        detections = detect_in_frame(
            frame, detector, nms, scale,
            conf_thres=conf_thres, iou_thres=iou_thres, imgsz=imgsz,
        )
        categories, stats = classify_frame(
            detections, frame.shape,
            crowd_threshold=crowd_threshold,
            small_area_ratio=small_area_ratio,
        )
        if not categories:
            return
        candidate = Candidate(
            source_ref=source_ref,
            frame_index=frame_index,
            captured_at_utc=utc_now_iso(),
            categories=categories,
            detections=[d.to_dict() for d in detections],
            stats=stats,
        )
        if frames_retained and out_root is not None:
            frame_dir = out_root / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            name = f"{run_id}_{Path(source_ref).stem}_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_dir / name), frame)
            candidate.retained_path = str((frame_dir / name).relative_to(out_root))
            result.frames_retained += 1
        result.candidates.append(candidate)

    if source_path.is_dir():
        for image_path in sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            handle_frame(frame, 0, str(image_path))
    else:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source: {source_path}")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % max(1, stride) == 0:
                    handle_frame(frame, frame_index, str(source_path))
                    if max_frames is not None and result.frames_examined >= max_frames:
                        break
                frame_index += 1
        finally:
            capture.release()

    if out_root is not None:
        write_mining_output(out_root, result)
    return result


def write_mining_output(out_dir, result: MiningResult) -> Path:
    """Write the candidate CSV, the run metadata and the review sheet."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "candidates.csv"
    fieldnames = [
        "run_id", "source_ref", "frame_index", "captured_at_utc",
        "categories", "detection_count", "mean_confidence",
        "median_box_area_ratio", "retained_path", "annotation_status", "review_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in result.candidates:
            writer.writerow({
                "run_id": result.run_id,
                "source_ref": candidate.source_ref,
                "frame_index": candidate.frame_index,
                "captured_at_utc": candidate.captured_at_utc,
                "categories": ";".join(candidate.categories),
                "detection_count": candidate.stats.get("detection_count", 0),
                "mean_confidence": candidate.stats.get("mean_confidence"),
                "median_box_area_ratio": candidate.stats.get("median_box_area_ratio"),
                "retained_path": candidate.retained_path or "",
                "annotation_status": candidate.annotation_status,
                "review_note": candidate.review_note,
            })

    (root / "run.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    review = root / "REVIEW.md"
    review.write_text(
        "# Candidate review sheet\n\n"
        f"Run: `{result.run_id}`  \n"
        f"Source: `{result.source}`  \n"
        f"Frames examined: {result.frames_examined}  \n"
        f"Candidates: {len(result.candidates)}  \n"
        f"Frames retained: {result.frames_retained} (retention {DEFAULT_RETENTION_DAYS} days)\n\n"
        "Every row below is a **machine proposal**. A human must confirm or\n"
        "correct the annotations before any of it may enter a dataset version.\n"
        "Do not copy the model's predictions into a label file unchanged.\n\n"
        "| source_ref | frame | categories | detections | mean conf | retained |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {c.source_ref} | {c.frame_index} | {', '.join(c.categories)} | "
            f"{c.stats.get('detection_count', 0)} | "
            f"{_round(c.stats.get('mean_confidence'))} | {c.retained_path or '-'} |"
            for c in result.candidates
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path


def _round(value, digits: int = 3):
    return "n/a" if value is None else round(float(value), digits)
