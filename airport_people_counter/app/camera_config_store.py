"""Atomic update helper for ``config/camera.yaml``.

One job: change low-level camera keys (``source``, ``rotation``, legacy
geometry points, ``source.shape_w/h``) in an atomic, validate-first way
with a ``.bak`` backup — the same guarantee ``mapping.save_mapping``
gives ``mappings.yaml``. A failed update must leave the running config
untouched (AGENTS.md invariant).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def _scalar_repr(value) -> Optional[str]:
    """YAML text for a scalar update value (``'1'`` → ``'1'``, 90 → 90)."""
    text = yaml.safe_dump(value, default_flow_style=True)
    if text.endswith("...\n"):
        text = text[:-4]
    text = text.strip()
    if "\n" in text or ": " in text or "{" in text or "[" in text:
        return None
    return text


def _patch_scalar_keys(text: str, updates: Dict[str, object]) -> Optional[str]:
    """Patch plain depth-2 scalar keys in place, preserving all comments.

    Returns the new file text, or ``None`` when any update is not a
    simple scalar replacement under a top-level block (list values such
    as legacy ROI points fall back to the full rewrite below). A key
    that does not exist yet is inserted as the first child of its block.
    """
    lines = text.splitlines(keepends=True)
    for dotted_key, value in updates.items():
        parts = dotted_key.split(".")
        if len(parts) != 2:
            return None
        block, key = parts
        if isinstance(value, (list, dict)):
            return None
        new_value = _scalar_repr(value)
        if new_value is None:
            return None
        header_re = re.compile(r"^%s:\s*(?:#.*)?\n?$" % re.escape(block))
        start = None
        for idx, line in enumerate(lines):
            if header_re.match(line):
                start = idx
                break
        if start is None:
            return None
        child_indent = None
        key_line = None
        for idx in range(start + 1, len(lines)):
            line = lines[idx]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 0:
                break  # next top-level block — ``key`` not present
            if child_indent is None:
                child_indent = indent
            if indent != child_indent:
                continue  # deeper nesting under another child
            if stripped.startswith(key + ":"):
                key_line = idx
                break
        if key_line is not None:
            old = lines[key_line]
            trailing = "\n" if old.endswith("\n") else ""
            comment = ""
            comment_pos = old.find("  #")
            if comment_pos != -1:
                comment = old[comment_pos:].rstrip("\n")
            lines[key_line] = (
                f"{' ' * child_indent}{key}: {new_value}{comment}{trailing}"
            )
        else:
            insert_at = None
            for idx in range(start + 1, len(lines)):
                stripped = lines[idx].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                insert_at = idx
                break
            if insert_at is None or child_indent is None:
                return None  # empty block — let the full rewrite handle it
            lines.insert(
                insert_at,
                f"{' ' * child_indent}{key}: {new_value}\n",
            )
    return "".join(lines)


def load_camera_yaml(path) -> Dict:
    """Load the raw camera.yaml document (raises on unreadable files)."""
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the document root")
    return doc


def _set_nested(doc: Dict, dotted_key: str, value) -> None:
    """Set ``a.b.c`` in a nested dict, creating intermediate levels."""
    parts = dotted_key.split(".")
    node = doc
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def update_camera_yaml(path, updates: Dict[str, object], *, backup: bool = True) -> Optional[Path]:
    """Apply ``updates`` (dotted keys) to camera.yaml atomically.

    The file is parsed first; a bad parse raises *before* anything is
    written. The new document is serialised to a temp file next to the
    original and moved into place with ``os.replace``, so a crash or a
    write failure cannot leave a half-written config. Returns the backup
    path when one was kept.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"camera config not found: {path}")

    doc = load_camera_yaml(path)  # validate-first: raise before writing
    for key, value in updates.items():
        _set_nested(doc, key, value)

    backup_path: Optional[Path] = None
    if backup:
        backup_path = path.with_name(path.name + ".bak")
        tmp_backup = path.with_name(path.name + ".bak.tmp")
        import shutil

        shutil.copyfile(path, tmp_backup)
        os.replace(tmp_backup, backup_path)

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            # Scalar-only updates are patched into the original text so
            # the operator's comments survive (rotation/source switches
            # must not rewrite the whole calibrated file). Anything the
            # patcher cannot express falls back to a full rewrite.
            old_text = path.read_text(encoding="utf-8")
            patched = _patch_scalar_keys(old_text, updates)
            if patched is not None and yaml.safe_load(patched) == doc:
                f.write(patched)
            else:
                yaml.safe_dump(doc, f, sort_keys=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return backup_path
