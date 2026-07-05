"""Dependency-free run metrics with optional TensorBoard output."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class RunLogger:
    """Persist run configuration and append structured metric events."""

    def __init__(
        self,
        runs_dir: str | Path = "runs",
        run_name: str | None = None,
        tensorboard: bool = False,
    ) -> None:
        name = run_name or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(
                "run_name may contain only letters, numbers, '.', '_', and '-'"
            )
        self.run_dir = Path(runs_dir) / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self._rows: list[dict[str, Any]] = []
        self._csv_fields: list[str] | None = None
        self._tensorboard = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "TensorBoard logging requires the optional 'tensorboard' "
                    "package; install the project with the 'metrics' extra"
                ) from error
            self._tensorboard = SummaryWriter(
                log_dir=str(self.run_dir / "tensorboard")
            )

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Write the immutable run configuration once."""
        path = self.run_dir / "config.json"
        path.write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(
        self,
        event: str,
        step: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """Append one event to JSONL, CSV, and optional TensorBoard."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "step": step,
            **dict(metrics),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows.append(row)
        self._update_csv(row)

        if self._tensorboard is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    self._tensorboard.add_scalar(
                        f"{event}/{name}", value, step
                    )
            self._tensorboard.flush()

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.close()

    def _update_csv(self, row: dict[str, Any]) -> None:
        """Append to the CSV, rewriting only when new columns appear."""
        if self._csv_fields is not None and all(
            key in self._csv_fields for key in row
        ):
            with self.csv_path.open(
                "a", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=self._csv_fields, restval=""
                )
                writer.writerow(row)
            return
        self._rewrite_csv()

    def _rewrite_csv(self) -> None:
        preferred = ["timestamp", "event", "step"]
        extra = sorted(
            {
                key
                for row in self._rows
                for key in row
                if key not in preferred
            }
        )
        self._csv_fields = preferred + extra
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=self._csv_fields, restval=""
            )
            writer.writeheader()
            writer.writerows(self._rows)
        temporary.replace(self.csv_path)
