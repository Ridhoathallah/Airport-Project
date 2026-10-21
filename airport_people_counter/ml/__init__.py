"""ML dataset engineering toolkit for the airport people-counter project.

This package owns the data lifecycle only: the data contract, dataset
validation, deterministic leakage-safe splitting, versioned manifests, the
metadata catalog, training-experiment provenance and evaluation reports.

It is deliberately separate from the runtime application (``app/``), the
legacy monitoring engine (``core/``) and the runtime monitoring database.
Nothing here is imported by the runtime, and nothing here writes to the
runtime database.

Typical use::

    from ml.contract import DEFAULT_CLASS_SCHEMA
    from ml.validate import validate_dataset
    from ml.split import SplitPolicy, plan_splits
    from ml.manifest import build_manifest, write_manifest
"""

from __future__ import annotations

__all__ = [
    "catalog",
    "collect",
    "contract",
    "evaluate",
    "experiment",
    "manifest",
    "preview",
    "split",
    "validate",
]

__version__ = "1.0.0"
