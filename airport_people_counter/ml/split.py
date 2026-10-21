"""Deterministic, leakage-safe dataset splitting.

Splitting is **group-first**: all frames that come from the same video or
recording session share a ``source_group`` and are always placed in the
same split. Randomly splitting adjacent frames from one video leaks
near-identical content between train and test and inflates every metric.

The assignment is a pure function of ``(seed, ratios, group set)``:

* groups are sorted by id, then shuffled with ``random.Random(seed)``;
* each group goes to the split with the largest remaining deficit, with a
  fixed tie-break order.

Re-running the planner with the same inputs therefore reproduces the same
split byte for byte, which is what makes an experiment reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .contract import ImageRecord, SplitName
from .validate import (
    NEAR_DUPLICATE_DISTANCE,
    Finding,
    Severity,
    hamming_distance,
)

#: Order used to break ties when several splits are equally hungry.
SPLIT_ORDER = (SplitName.TRAIN.value, SplitName.VAL.value, SplitName.TEST.value)


class OrphanPolicy(str):
    """How to treat images that carry no source-group metadata."""

    SINGLETON = "singleton"
    ERROR = "error"


@dataclass
class SplitPolicy:
    """The recorded inputs of a split decision."""

    seed: int = 0
    ratios: dict = field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1})
    group_key: str = "source_group"
    orphan_policy: str = OrphanPolicy.SINGLETON
    near_duplicate_threshold: int = NEAR_DUPLICATE_DISTANCE

    def normalised_ratios(self) -> dict:
        total = sum(max(0.0, float(v)) for v in self.ratios.values())
        if total <= 0:
            raise ValueError("split ratios must sum to a positive number")
        return {name: max(0.0, float(value)) / total for name, value in self.ratios.items()}

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "group_key": self.group_key,
            "orphan_policy": self.orphan_policy,
            "near_duplicate_threshold": self.near_duplicate_threshold,
        }


@dataclass
class SplitResult:
    """Outcome of planning a split."""

    assignments: dict = field(default_factory=dict)   # image_id -> split name
    groups: dict = field(default_factory=dict)        # group id -> split name
    findings: list[Finding] = field(default_factory=list)
    policy: Optional[SplitPolicy] = None
    counts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not [f for f in self.findings if f.severity is Severity.ERROR]

    def splits(self) -> dict:
        """``{split: [image_id, ...]}`` in deterministic order."""
        out: dict = {name: [] for name in SPLIT_ORDER}
        for image_id, split in self.assignments.items():
            out.setdefault(split, []).append(image_id)
        for names in out.values():
            names.sort()
        return out

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.to_dict() if self.policy else {},
            "counts": dict(self.counts),
            "groups": dict(self.groups),
            "assignments": dict(self.assignments),
            "findings": [f.to_dict() for f in self.findings],
        }


def _group_of(record: ImageRecord, policy: SplitPolicy) -> Optional[str]:
    value = getattr(record, policy.group_key, None)
    return str(value) if value else None


def plan_splits(records: Iterable[ImageRecord], policy: Optional[SplitPolicy] = None) -> SplitResult:
    """Assign every record to a split, respecting source groups."""
    policy = policy or SplitPolicy()
    ratios = policy.normalised_ratios()
    items = list(records)
    result = SplitResult(policy=policy)

    if not items:
        result.counts = {"images": 0, "groups": 0, "orphans": 0}
        return result

    # ── group the records ───────────────────────────────────
    groups: dict = {}
    orphans: list[ImageRecord] = []
    for record in items:
        group = _group_of(record, policy)
        if group is None:
            orphans.append(record)
        else:
            groups.setdefault(group, []).append(record)

    if orphans:
        if policy.orphan_policy == OrphanPolicy.ERROR:
            result.findings.append(Finding(
                "orphan_images",
                Severity.ERROR,
                f"{len(orphans)} image(s) have no '{policy.group_key}' metadata",
                None,
                "assign a source group or switch the orphan policy to 'singleton'",
            ))
        else:
            result.findings.append(Finding(
                "orphan_images",
                Severity.WARNING,
                f"{len(orphans)} image(s) have no '{policy.group_key}' metadata and are "
                "treated as independent single-image groups; their split membership "
                "cannot be guaranteed free of leakage",
                None,
            ))
        for record in orphans:
            groups[f"__singleton__{record.image_id}"] = [record]

    # ── deterministic group order ───────────────────────────
    group_ids = sorted(groups)
    random.Random(policy.seed).shuffle(group_ids)

    # ── greedy deficit assignment ───────────────────────────
    total_images = len(items)
    targets = {name: ratios.get(name, 0.0) * total_images for name in SPLIT_ORDER}
    assigned = {name: 0 for name in SPLIT_ORDER}
    group_split: dict = {}
    assignments: dict = {}

    for group_id in group_ids:
        members = groups[group_id]
        size = len(members)
        best_split = None
        best_deficit = None
        for name in SPLIT_ORDER:
            deficit = targets[name] - assigned[name]
            if best_deficit is None or deficit > best_deficit + 1e-9:
                best_deficit = deficit
                best_split = name
        group_split[group_id] = best_split
        assigned[best_split] += size
        for record in members:
            assignments[record.image_id] = best_split

    result.assignments = assignments
    result.groups = group_split
    result.counts = {
        "images": total_images,
        "groups": len(group_ids),
        "orphans": len(orphans),
        **{f"n_{name}": assigned[name] for name in SPLIT_ORDER},
    }

    check_leakage(items, assignments, policy, result.findings)
    return result


def check_leakage(
    records: Iterable[ImageRecord],
    assignments: dict,
    policy: Optional[SplitPolicy] = None,
    findings: Optional[list] = None,
) -> list[Finding]:
    """Detect content that crosses a split boundary.

    Three kinds of leakage are checked: a group that was split across
    train/val/test, an exact duplicate (identical SHA-256) in two splits,
    and a near-duplicate (perceptual hash) in two splits.
    """
    policy = policy or SplitPolicy()
    out = findings if findings is not None else []
    items = list(records)
    by_id = {r.image_id: r for r in items}

    # 1. group crossing a split
    group_splits: dict = {}
    for record in items:
        group = _group_of(record, policy)
        if group is None:
            continue
        split = assignments.get(record.image_id)
        if group in group_splits and group_splits[group] != split:
            out.append(Finding(
                "group_leakage",
                Severity.ERROR,
                f"source group {group!r} appears in both "
                f"{group_splits[group]!r} and {split!r}",
                record.relpath,
            ))
        else:
            group_splits[group] = split

    # 2. exact duplicates across splits
    by_hash: dict = {}
    for record in items:
        split = assignments.get(record.image_id)
        if record.sha256 in by_hash and by_hash[record.sha256][1] != split:
            other_rel, other_split = by_hash[record.sha256]
            out.append(Finding(
                "duplicate_leakage",
                Severity.ERROR,
                f"identical image content also assigned to {other_split!r}",
                record.relpath,
                f"duplicate of {other_rel}",
            ))
        else:
            by_hash.setdefault(record.sha256, (record.relpath, split))

    # 3. near-duplicates across splits
    hashed = [r for r in items if r.phash is not None]
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            left, right = hashed[i], hashed[j]
            if assignments.get(left.image_id) == assignments.get(right.image_id):
                continue
            distance = hamming_distance(left.phash, right.phash)
            if distance is not None and distance <= policy.near_duplicate_threshold:
                out.append(Finding(
                    "near_duplicate_leakage",
                    Severity.WARNING,
                    f"perceptual hash distance {distance} from an image in "
                    f"{assignments.get(right.image_id)!r}",
                    left.relpath,
                    f"near-duplicate of {right.relpath}",
                ))
    return out
