"""Training lifecycle management for the console.

Training always runs as a subprocess of the real CLI
(``python -m gomoku_muzero.train``), so the console is a pure layer on
top of the existing command: a training crash cannot take the console
down, stopping is the same SIGINT a terminal user would send (the
training loop saves checkpoint and state every iteration, so stopping is
loss-free up to the current iteration), and resuming uses ``--resume``
exactly as documented.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from gomoku_muzero.cli.train import build_parser

PROGRESS_METRICS = (
    "loss",
    "policy_kl",
    "value_loss",
    "value_calibration_mae",
    "evaluation_score",
    "heuristic_evaluation_score",
    "games_per_second",
)


def training_options() -> list[dict[str, Any]]:
    """Describe the real training CLI's options for form generation.

    Introspects the actual argparse parser so the console can never
    drift from the command line.
    """
    entries: list[dict[str, Any]] = []
    for action in build_parser()._actions:
        if action.dest == "help":
            continue
        if isinstance(action, argparse._StoreTrueAction):
            kind = "flag"
        elif action.type is int:
            kind = "int"
        elif action.type is float:
            kind = "float"
        else:
            kind = "str"
        entries.append(
            {
                "flag": action.option_strings[0],
                "name": action.dest,
                "kind": kind,
                "default": action.default,
                "choices": list(action.choices) if action.choices else None,
                "help": action.help,
            }
        )
    return entries


def default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


class TrainingManager:
    """Own at most one training subprocess and report its progress."""

    def __init__(self, workdir: str | Path = ".") -> None:
        self.workdir = Path(workdir)
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._run_name: str | None = None
        self._command: list[str] | None = None
        self._log_path: Path | None = None
        self._started_at: str | None = None
        self._finished_status: dict[str, Any] | None = None

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        """Launch ``python -m gomoku_muzero.train`` with validated options."""
        with self._lock:
            if self._alive():
                raise ValueError("a training run is already active")
            known = {entry["name"]: entry for entry in training_options()}
            provided = dict(options)
            run_name = str(
                provided.pop("run_name", "") or default_run_name()
            )
            command = [
                sys.executable,
                "-m",
                "gomoku_muzero.train",
                "--run-name",
                run_name,
            ]
            for name, value in provided.items():
                entry = known.get(name)
                if entry is None:
                    raise ValueError(f"unknown training option: {name}")
                if value is None or value == "":
                    continue
                if entry["kind"] == "flag":
                    if value in (True, "true", "on", 1):
                        command.append(entry["flag"])
                    continue
                command.extend([entry["flag"], str(value)])

            log_dir = self.workdir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / f"{run_name}.log"
            log_file = self._log_path.open("wb")
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.workdir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            finally:
                log_file.close()
            self._run_name = run_name
            self._command = command
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._finished_status = None
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Interrupt the active run; state is saved every iteration."""
        with self._lock:
            if not self._alive():
                raise ValueError("no active training run")
            self._process.send_signal(signal.SIGINT)
        return self.status()

    def status(self) -> dict[str, Any]:
        """Current lifecycle state, progress, and a metrics snapshot."""
        with self._lock:
            if self._process is None:
                return {"state": "idle"}
            returncode = self._process.poll()
            status: dict[str, Any] = {
                "state": "running" if returncode is None else (
                    "finished" if returncode == 0 else "failed"
                ),
                "run_name": self._run_name,
                "pid": self._process.pid,
                "returncode": returncode,
                "started_at": self._started_at,
                "command": self._command,
            }
        status.update(self.run_progress(self._run_name))
        status["log_tail"] = self.log_tail()
        return status

    def run_progress(self, run_name: str | None) -> dict[str, Any]:
        """Progress derived from the run's own metrics files."""
        if not run_name:
            return {}
        run_dir = self.workdir / "runs" / run_name
        progress: dict[str, Any] = {}
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        try:
            config = json.loads(config_path.read_text())
            progress["requested_iterations"] = config.get("iterations")
            progress["resumed_from"] = config.get("resume")
        except (OSError, json.JSONDecodeError):
            return progress
        try:
            lines = metrics_path.read_text().splitlines()
        except OSError:
            lines = []
        if lines:
            first = json.loads(lines[0])
            last = json.loads(lines[-1])
            progress["first_iteration"] = first.get("step")
            progress["iteration"] = last.get("step")
            if isinstance(config.get("iterations"), int):
                progress["target_iteration"] = (
                    first.get("step", 1) + config["iterations"] - 1
                )
            latest = {}
            for row in reversed(lines[-25:]):
                data = json.loads(row)
                for key in PROGRESS_METRICS:
                    if key in data and key not in latest:
                        latest[key] = data[key]
            progress["latest_metrics"] = latest
        return progress

    def log_tail(self, max_lines: int = 12) -> list[str]:
        if self._log_path is None or not self._log_path.exists():
            return []
        try:
            raw = self._log_path.read_bytes()[-16384:]
        except OSError:
            return []
        text = raw.decode("utf-8", errors="replace")
        lines = [
            piece.strip()
            for chunk in text.splitlines()
            for piece in chunk.split("\r")
            if piece.strip()
        ]
        return lines[-max_lines:]

    def active_run_artifacts(self) -> set[str]:
        """Paths a cleanup layer must refuse to touch while running."""
        with self._lock:
            if not self._alive() or self._command is None:
                return set()
            artifacts = {f"runs/{self._run_name}"}
            for flag in ("--checkpoint", "--training-state"):
                if flag in self._command:
                    artifacts.add(
                        self._command[self._command.index(flag) + 1]
                    )
            return artifacts

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None
