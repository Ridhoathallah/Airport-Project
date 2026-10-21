"""Training experiment metadata and checkpoint provenance.

Every fine-tuning run is described by a *resolved* configuration: the
exact arguments, the dataset version it was trained on, the base
checkpoint, and the environment it ran in. The configuration is hashed
(canonically) so a run can be matched to its stored configuration later,
and the base checkpoint is hashed so the provenance of the weights is
recorded rather than assumed.

Nothing here starts a training run; :mod:`scripts.train_person` does that
using the command this module builds.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .contract import make_experiment_id, sha256_file, sha256_json, utc_now_iso

#: Project root (the directory that contains ``ml/``, ``app/``, ``yolov5/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The vendored YOLOv5 checkout used for training and validation.
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"


@dataclass
class ExperimentConfig:
    """The fully resolved inputs of one training run."""

    data: str
    weights: str
    epochs: int = 50
    imgsz: int = 640
    batch_size: int = 16
    optimizer: str = "SGD"
    device: str = "cpu"
    seed: int = 0
    workers: int = 4
    hyp: str = ""
    project: str = "runs/train"
    name: str = "exp"
    single_cls: bool = True
    exist_ok: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Resolved configuration, suitable for hashing and storage."""
        data = {
            "data": str(self.data),
            "weights": str(self.weights),
            "epochs": int(self.epochs),
            "imgsz": int(self.imgsz),
            "batch_size": int(self.batch_size),
            "optimizer": str(self.optimizer),
            "device": str(self.device),
            "seed": int(self.seed),
            "workers": int(self.workers),
            "hyp": str(self.hyp),
            "project": str(self.project),
            "name": str(self.name),
            "single_cls": bool(self.single_cls),
            "exist_ok": bool(self.exist_ok),
        }
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @property
    def sha256(self) -> str:
        """Canonical hash of the resolved configuration."""
        return sha256_json(self.to_dict())

    def build_command(self) -> list[str]:
        """The exact argv used to invoke the vendored ``yolov5/train.py``."""
        command = [
            sys.executable,
            str(YOLOV5_ROOT / "train.py"),
            "--data", str(self.data),
            "--weights", str(self.weights),
            "--epochs", str(self.epochs),
            "--imgsz", str(self.imgsz),
            "--batch-size", str(self.batch_size),
            "--optimizer", str(self.optimizer),
            "--device", str(self.device),
            "--seed", str(self.seed),
            "--workers", str(self.workers),
            "--project", str(self.project),
            "--name", str(self.name),
        ]
        if self.hyp:
            command += ["--hyp", str(self.hyp)]
        if self.single_cls:
            command.append("--single-cls")
        if self.exist_ok:
            command.append("--exist-ok")
        for key, value in self.extra.items():
            flag = f"--{str(key).replace('_', '-')}"
            if value is True:
                command.append(flag)
            elif value not in (None, False):
                command += [flag, str(value)]
        return command


@dataclass
class TrainingRun:
    """The outcome of a training run (complete, partial or failed)."""

    experiment_id: str
    status: str = "planned"          # planned | running | completed | failed
    started_at_utc: Optional[str] = None
    ended_at_utc: Optional[str] = None
    command: list = field(default_factory=list)
    log_path: Optional[str] = None
    checkpoints: dict = field(default_factory=dict)   # role -> path
    warnings: list = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "command": list(self.command),
            "log_path": self.log_path,
            "checkpoints": dict(self.checkpoints),
            "warnings": list(self.warnings),
            "error": self.error,
        }


def create_experiment_id(config: ExperimentConfig, created_at_utc: Optional[str] = None) -> str:
    """Stable identifier for an experiment (timestamp + config hash)."""
    return make_experiment_id(config.sha256, created_at_utc or utc_now_iso())


def capture_environment() -> dict:
    """Capture the environment a run was executed in.

    Every field is read from the running interpreter; nothing is guessed.
    Fields that cannot be determined are reported as ``None`` rather than
    being filled in with a plausible default.
    """
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.cuda, "version", lambda: None)()
        env["mps_available"] = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
    except Exception as exc:  # torch is optional for the metadata tooling
        env["torch_version"] = None
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torchvision

        env["torchvision_version"] = torchvision.__version__
    except Exception:
        env["torchvision_version"] = None
    try:
        import ultralytics

        env["ultralytics_version"] = ultralytics.__version__
    except Exception:
        env["ultralytics_version"] = None
    env["yolov5_revision"] = yolov5_revision()
    return env


def yolov5_revision() -> Optional[str]:
    """The git revision of the vendored YOLOv5 checkout, if it is a git repo."""
    if not (YOLOV5_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(YOLOV5_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def hash_checkpoint(path) -> Optional[str]:
    """SHA-256 of a checkpoint file, or ``None`` when it does not exist."""
    target = Path(path)
    if not target.is_file():
        return None
    return sha256_file(target)


def describe_checkpoint(path) -> dict:
    """Provenance record for a checkpoint: path, hash, size, mtime."""
    target = Path(path)
    if not target.is_file():
        return {"path": str(target), "exists": False, "sha256": None, "bytes": None}
    stat = target.stat()
    return {
        "path": str(target),
        "exists": True,
        "sha256": sha256_file(target),
        "bytes": stat.st_size,
        "modified_at_utc": utc_now_iso(),
    }


def find_checkpoints(run_dir) -> dict:
    """Locate the ``best.pt`` / ``last.pt`` checkpoints of a finished run."""
    run = Path(run_dir)
    found: dict[str, str] = {}
    for role, filename in (("best", "best.pt"), ("last", "last.pt")):
        candidate = run / "weights" / filename
        if candidate.is_file():
            found[role] = str(candidate)
    return found
