"""Metric series for the console's charts, read from runs' JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHART_KEYS = (
    "loss",
    "policy_kl",
    "value_loss",
    "value_calibration_mae",
    "evaluation_score",
    "heuristic_evaluation_score",
    "games_per_second",
)
MAX_POINTS_PER_SERIES = 400


def list_runs(workdir: str | Path) -> list[dict[str, Any]]:
    """Runs that have metrics, newest first, with their step ranges."""
    runs_dir = Path(workdir) / "runs"
    if not runs_dir.is_dir():
        return []
    entries = []
    for run in runs_dir.iterdir():
        metrics = run / "metrics.jsonl"
        if not run.is_dir() or not metrics.is_file():
            continue
        lines = metrics.read_text().splitlines()
        if not lines:
            continue
        entries.append(
            {
                "name": run.name,
                "first_iteration": json.loads(lines[0]).get("step"),
                "last_iteration": json.loads(lines[-1]).get("step"),
                "modified": metrics.stat().st_mtime,
            }
        )
    entries.sort(key=lambda entry: entry["modified"], reverse=True)
    return entries


def metric_series(
    workdir: str | Path, run_names: list[str]
) -> dict[str, Any]:
    """Chart-ready ``{key: {run: [[step, value], ...]}}`` for the runs.

    Long runs are downsampled evenly to at most
    ``MAX_POINTS_PER_SERIES`` points per series, always keeping the
    final point.
    """
    runs_dir = Path(workdir) / "runs"
    series: dict[str, dict[str, list[list[float]]]] = {
        key: {} for key in CHART_KEYS
    }
    for name in run_names:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("invalid run name")
        metrics = runs_dir / name / "metrics.jsonl"
        if not metrics.is_file():
            continue
        rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if line.strip()
        ]
        for key in CHART_KEYS:
            points = [
                [row["step"], row[key]]
                for row in rows
                if key in row and isinstance(row.get("step"), int)
            ]
            if points:
                series[key][name] = _downsample(points)
    return {"keys": list(CHART_KEYS), "series": series}


def _downsample(points: list[list[float]]) -> list[list[float]]:
    if len(points) <= MAX_POINTS_PER_SERIES:
        return points
    step = len(points) / MAX_POINTS_PER_SERIES
    sampled = [
        points[int(index * step)]
        for index in range(MAX_POINTS_PER_SERIES)
    ]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    return sampled
