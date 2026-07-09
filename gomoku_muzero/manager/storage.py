"""Run and checkpoint administration for the console.

Listing, per-model information, and deletion. Deletion is guarded three
ways: paths must resolve inside the managed directories, anything the
active training run writes is refused, and the caller must echo the
exact name being deleted (the type-the-name confirmation happens in the
UI; the server re-verifies it).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

MODEL_INFO_OPTION_KEYS = (
    "board_size",
    "win_length",
    "hidden_channels",
    "res_blocks",
    "simulations",
    "games_per_iteration",
    "training_steps",
    "batch_size",
    "learning_rate",
    "value_loss_weight",
    "temperature_moves",
    "replay_capacity",
    "replay_sampling",
    "seed",
)


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(
        p.stat().st_size for p in path.rglob("*") if p.is_file()
    ) / 1e6


def list_storage(
    workdir: str | Path, checkpoint_dir: str | Path
) -> dict[str, Any]:
    """Everything on disk the console manages, with sizes and dates."""
    runs = []
    runs_dir = Path(workdir) / "runs"
    if runs_dir.is_dir():
        for run in sorted(runs_dir.iterdir()):
            if not run.is_dir():
                continue
            entry: dict[str, Any] = {
                "name": run.name,
                "size_mb": round(_size_mb(run), 2),
                "modified": run.stat().st_mtime,
            }
            metrics = run / "metrics.jsonl"
            if metrics.is_file():
                lines = metrics.read_text().splitlines()
                if lines:
                    entry["first_iteration"] = json.loads(lines[0]).get(
                        "step"
                    )
                    entry["last_iteration"] = json.loads(lines[-1]).get(
                        "step"
                    )
            runs.append(entry)

    checkpoints = []
    root = Path(checkpoint_dir)
    if root.is_dir():
        for path in sorted(root.glob("*.pt")):
            checkpoints.append(
                {
                    "name": path.name,
                    "kind": (
                        "training-state"
                        if "state" in path.stem
                        else "model"
                    ),
                    "size_mb": round(_size_mb(path), 2),
                    "modified": path.stat().st_mtime,
                }
            )
    return {"runs": runs, "checkpoints": checkpoints}


def model_info(
    checkpoint_dir: str | Path, name: str, workdir: str | Path = "."
) -> dict[str, Any]:
    """Basic information about a trained model checkpoint.

    Reads the checkpoint's own metadata (board, architecture) and
    cross-references every run whose configuration wrote this file,
    yielding the training options used and the iterations trained.
    Training-state files are summarized without loading their replay
    buffers.
    """
    path = _safe_child(Path(checkpoint_dir), name)
    if not path.is_file():
        raise FileNotFoundError(f"no such checkpoint: {name}")

    info: dict[str, Any] = {
        "name": name,
        "size_mb": round(_size_mb(path), 2),
        "modified": path.stat().st_mtime,
    }
    if "state" in path.stem:
        info["kind"] = "training-state"
        info["note"] = (
            "Resumable training state (model, optimizer, and replay "
            "buffer); use it with --resume. Not inspected in depth to "
            "avoid loading the replay buffer."
        )
        return info

    data = torch.load(path, map_location="cpu", weights_only=True)
    info.update(
        {
            "kind": "model",
            "format_version": data.get("format_version"),
            "board_size": data.get("board_size"),
            "win_length": data.get("win_length"),
            "hidden_channels": data.get("hidden_channels"),
            "num_blocks": data.get("num_blocks"),
            "parameters": sum(
                tensor.numel()
                for tensor in data.get("model_state_dict", {}).values()
            ),
        }
    )

    trained_by = []
    runs_dir = Path(workdir) / "runs"
    if runs_dir.is_dir():
        for config_path in sorted(runs_dir.glob("*/config.json")):
            try:
                config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            written = str(config.get("checkpoint", ""))
            if Path(written).name != name:
                continue
            run_entry: dict[str, Any] = {
                "run": config_path.parent.name,
                "resumed": bool(config.get("resume")),
                "options": {
                    key: config[key]
                    for key in MODEL_INFO_OPTION_KEYS
                    if key in config
                },
            }
            metrics = config_path.parent / "metrics.jsonl"
            if metrics.is_file():
                lines = metrics.read_text().splitlines()
                if lines:
                    run_entry["iterations"] = (
                        json.loads(lines[0]).get("step"),
                        json.loads(lines[-1]).get("step"),
                    )
            trained_by.append(run_entry)
    trained_by.sort(
        key=lambda entry: entry.get("iterations") or (0, 0), reverse=True
    )
    info["trained_by"] = trained_by
    if trained_by and trained_by[0].get("iterations"):
        info["total_iterations"] = trained_by[0]["iterations"][1]
    return info


def delete_item(
    kind: str,
    name: str,
    confirm: str,
    workdir: str | Path,
    checkpoint_dir: str | Path,
    protected: set[str],
) -> dict[str, Any]:
    """Delete a run directory or checkpoint file after strict checks."""
    if confirm != name:
        raise ValueError(
            "confirmation does not match: type the exact name to delete"
        )
    if kind == "run":
        target = _safe_child(Path(workdir) / "runs", name)
        protected_names = {Path(p).name for p in protected if "/" in p}
        if name in protected_names:
            raise ValueError(
                "this run belongs to the active training and cannot "
                "be deleted"
            )
        if not target.is_dir():
            raise FileNotFoundError(f"no such run: {name}")
        shutil.rmtree(target)
    elif kind == "checkpoint":
        target = _safe_child(Path(checkpoint_dir), name)
        for artifact in protected:
            if Path(artifact).name == name:
                raise ValueError(
                    "this file is written by the active training and "
                    "cannot be deleted"
                )
        if not target.is_file():
            raise FileNotFoundError(f"no such checkpoint: {name}")
        target.unlink()
    else:
        raise ValueError("kind must be 'run' or 'checkpoint'")
    return {"deleted": name, "kind": kind}


def _safe_child(root: Path, name: str) -> Path:
    """Resolve ``name`` strictly inside ``root``."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid name")
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError("invalid name")
    return path
