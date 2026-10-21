"""Evaluation reports and baseline-vs-candidate comparison.

Metrics come from the vendored YOLOv5 implementation (``yolov5/val.py``),
so the numbers follow YOLOv5's definitions:

* ``precision`` / ``recall`` -- computed at the validation confidence
  threshold, over the IoU 0.50 matching used by the COCO-style evaluator;
* ``mAP@0.5`` and ``mAP@0.5:0.95`` -- averaged over the 101-point
  interpolation used by ``yolov5/utils/metrics.py``;
* per-class values are the same quantities restricted to one class.

Two reports are only ever compared when they were produced on the *same*
dataset version and split, which the report records. The acceptance
criteria shipped here are **provisional** defaults: they encode a
reasonable engineering expectation, they are not an agreed product
requirement, and they are labelled as provisional in every report.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .contract import utc_now_iso

#: Column names emitted by the vendored YOLOv5 ``results.csv``.
COL_PRECISION = "metrics/precision"
COL_RECALL = "metrics/recall"
COL_MAP50 = "metrics/mAP_0.5"
COL_MAP5095 = "metrics/mAP_0.5:0.95"

#: Provisional acceptance criteria. NOT an agreed product requirement.
DEFAULT_ACCEPTANCE_CRITERIA = {
    "provisional": True,
    "rationale": (
        "Defaults encode a conservative expectation: a candidate must clearly "
        "improve detection quality, must not regress mAP@0.5:0.95, and must "
        "not trade away recall (missed people are the operationally costly "
        "error in a crowd-safety system). Ratify before using to gate releases."
    ),
    "criteria": [
        {"metric": "map_50", "rule": "greater_or_equal", "value": 0.0,
         "delta_against": "baseline", "note": "absolute mAP@0.5 must not fall below baseline"},
        {"metric": "map_50_95", "rule": "greater_or_equal", "value": 0.0,
         "delta_against": "baseline", "note": "mAP@0.5:0.95 must not fall below baseline"},
        {"metric": "map_50", "rule": "greater_or_equal", "value": 0.02,
         "delta_against": "baseline", "note": "target improvement of at least +0.02 mAP@0.5"},
        {"metric": "recall", "rule": "greater_or_equal", "value": -0.01,
         "delta_against": "baseline", "note": "recall may not regress by more than 0.01"},
    ],
}


@dataclass
class MetricSet:
    """The metric block produced by one YOLOv5 validation run."""

    precision: Optional[float] = None
    recall: Optional[float] = None
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None
    per_class: dict = field(default_factory=dict)
    images: Optional[int] = None
    instances: Optional[int] = None
    source: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map_50": self.map_50,
            "map_50_95": self.map_50_95,
            "per_class": dict(self.per_class),
            "images": self.images,
            "instances": self.instances,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricSet":
        return cls(
            precision=data.get("precision"),
            recall=data.get("recall"),
            map_50=data.get("map_50"),
            map_50_95=data.get("map_50_95"),
            per_class=dict(data.get("per_class") or {}),
            images=data.get("images"),
            instances=data.get("instances"),
            source=data.get("source"),
        )

    def get(self, name: str) -> Optional[float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map_50": self.map_50,
            "map_50_95": self.map_50_95,
        }.get(name)


@dataclass
class EvaluationReport:
    """A normalised evaluation result for one checkpoint on one split."""

    model_path: str
    model_sha256: Optional[str] = None
    dataset_version_id: Optional[str] = None
    dataset_version: Optional[str] = None
    split: str = "test"
    metrics: MetricSet = field(default_factory=MetricSet)
    evaluated_at_utc: str = field(default_factory=utc_now_iso)
    provenance: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "dataset_version_id": self.dataset_version_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "metrics": self.metrics.to_dict(),
            "evaluated_at_utc": self.evaluated_at_utc,
            "provenance": dict(self.provenance),
            "notes": self.notes,
        }

    def render_text(self) -> str:
        m = self.metrics
        lines = [
            f"Evaluation report: {self.model_path}",
            f"  split          : {self.split}",
            f"  dataset version: {self.dataset_version or self.dataset_version_id or 'n/a'}",
            f"  precision      : {_fmt(m.precision)}",
            f"  recall         : {_fmt(m.recall)}",
            f"  mAP@0.5        : {_fmt(m.map_50)}",
            f"  mAP@0.5:0.95   : {_fmt(m.map_50_95)}",
        ]
        if m.per_class:
            lines.append("  per class:")
            for name, values in sorted(m.per_class.items()):
                lines.append(
                    f"    - {name}: p={_fmt(values.get('precision'))} "
                    f"r={_fmt(values.get('recall'))} "
                    f"ap50={_fmt(values.get('map_50'))} "
                    f"ap50_95={_fmt(values.get('map_50_95'))}"
                )
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Baseline-vs-candidate comparison with an explicit acceptance verdict."""

    baseline: EvaluationReport
    candidate: EvaluationReport
    deltas: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    comparable: bool = True
    notes: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.comparable) and all(check["passed"] for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "deltas": dict(self.deltas),
            "checks": list(self.checks),
            "criteria": dict(self.criteria),
            "comparable": self.comparable,
            "passed": self.passed,
            "notes": self.notes,
        }

    def render_text(self) -> str:
        lines = [
            "Baseline vs candidate",
            f"  baseline : {self.baseline.model_path}",
            f"  candidate: {self.candidate.model_path}",
            f"  comparable: {self.comparable}",
        ]
        for name, delta in self.deltas.items():
            lines.append(f"  delta {name:<12}: {delta:+.4f}")
        for check in self.checks:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{mark}] {check['metric']}: {check['detail']}")
        lines.append(f"  verdict  : {'ACCEPT' if self.passed else 'REJECT'}")
        if self.notes:
            lines.append(f"  notes    : {self.notes}")
        return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_results_csv(path) -> MetricSet:
    """Parse a YOLOv5 ``results.csv`` into a :class:`MetricSet`.

    The row with the highest YOLOv5 fitness (``0.1*mAP50 + 0.9*mAP5095``)
    is used, which is the checkpoint YOLOv5 itself keeps as ``best.pt``.
    """
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return MetricSet(source=str(source))
    header = [cell.strip() for cell in rows[0]]
    try:
        p_idx = header.index(COL_PRECISION)
        r_idx = header.index(COL_RECALL)
        m50_idx = header.index(COL_MAP50)
        m95_idx = header.index(COL_MAP5095)
    except ValueError:
        return MetricSet(source=str(source))

    best_row = None
    best_fitness = None
    for row in rows[1:]:
        if len(row) <= max(p_idx, r_idx, m50_idx, m95_idx):
            continue
        try:
            values = [float(row[i]) for i in (p_idx, r_idx, m50_idx, m95_idx)]
        except ValueError:
            continue
        fitness = 0.1 * values[2] + 0.9 * values[3]
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness
            best_row = values
    if best_row is None:
        return MetricSet(source=str(source))
    return MetricSet(
        precision=best_row[0],
        recall=best_row[1],
        map_50=best_row[2],
        map_50_95=best_row[3],
        source=str(source),
    )


_VAL_LINE = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def parse_val_stdout(text: str) -> MetricSet:
    """Parse the summary line of a ``yolov5/val.py`` run.

    The vendored validator prints a final line of the form::

        all  <images>  <instances>  <precision>  <recall>  <mAP@0.5>  <mAP@0.5:0.95>
    """
    for line in reversed(text.splitlines()):
        match = _VAL_LINE.match(line)
        if match:
            images, instances, precision, recall, map50, map95 = match.groups()
            return MetricSet(
                precision=float(precision),
                recall=float(recall),
                map_50=float(map50),
                map_50_95=float(map95),
                images=int(images),
                instances=int(instances),
                source="stdout",
            )
    return MetricSet(source="stdout")


def compare(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    criteria: Optional[dict] = None,
) -> ComparisonReport:
    """Compare a candidate against the baseline and apply the acceptance gates."""
    criteria = criteria or DEFAULT_ACCEPTANCE_CRITERIA
    notes = ""
    comparable = True
    if baseline.dataset_version_id and candidate.dataset_version_id \
            and baseline.dataset_version_id != candidate.dataset_version_id:
        comparable = False
        notes = "baseline and candidate were evaluated on different dataset versions"
    if baseline.split != candidate.split:
        comparable = False
        notes = (notes + "; " if notes else "") + "baseline and candidate used different splits"

    deltas: dict = {}
    for name in ("precision", "recall", "map_50", "map_50_95"):
        base_value = baseline.metrics.get(name)
        cand_value = candidate.metrics.get(name)
        if base_value is None or cand_value is None:
            deltas[name] = None
        else:
            deltas[name] = cand_value - base_value

    checks: list[dict] = []
    for criterion in criteria.get("criteria", []):
        metric = criterion["metric"]
        threshold = float(criterion["value"])
        rule = criterion.get("rule", "greater_or_equal")
        delta = deltas.get(metric)
        base_value = baseline.metrics.get(metric)
        cand_value = candidate.metrics.get(metric)
        if delta is None or base_value is None or cand_value is None:
            checks.append({
                "metric": metric,
                "passed": False,
                "detail": f"{metric} could not be evaluated (missing metric)",
                "note": criterion.get("note", ""),
            })
            continue
        if rule == "greater_or_equal":
            passed = delta >= threshold - 1e-12
            detail = f"delta {delta:+.4f} vs required >= {threshold:+.4f}"
        elif rule == "greater":
            passed = delta > threshold + 1e-12
            detail = f"delta {delta:+.4f} vs required > {threshold:+.4f}"
        else:
            passed = False
            detail = f"unknown rule {rule!r}"
        checks.append({
            "metric": metric,
            "rule": rule,
            "threshold": threshold,
            "delta": delta,
            "baseline_value": base_value,
            "candidate_value": cand_value,
            "passed": bool(passed),
            "detail": detail,
            "note": criterion.get("note", ""),
        })

    if not comparable:
        notes = (notes + "; " if notes else "") + \
            "comparison is not trustworthy because the evaluations are not comparable"

    return ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        checks=checks,
        criteria=criteria,
        comparable=comparable,
        notes=notes,
    )


def write_report(path, payload: dict) -> Path:
    """Write a report as pretty-printed JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
