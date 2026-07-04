"""Create static charts from a Gomoku MuZero run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_iteration_metrics(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if row.get("event") == "iteration"]


def create_plots(run_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires Matplotlib; install the project with the "
            "'metrics' extra"
        ) from error

    directory = Path(run_dir)
    rows = load_iteration_metrics(directory)
    if not rows:
        raise ValueError("the run contains no iteration metrics")
    output_dir = directory / "plots"
    output_dir.mkdir(exist_ok=True)
    steps = [row["step"] for row in rows]
    outputs: list[Path] = []

    outputs.extend(
        _plot_series(
            plt,
            steps,
            rows,
            ("loss", "policy_loss", "value_loss", "reward_loss", "policy_kl"),
            "Training losses",
            output_dir / "losses.png",
        )
    )
    outputs.extend(
        _plot_series(
            plt,
            steps,
            rows,
            ("games_per_second", "training_steps_per_second"),
            "Throughput",
            output_dir / "throughput.png",
        )
    )
    outputs.extend(
        _plot_series(
            plt,
            steps,
            rows,
            ("evaluation_score",),
            "Evaluation",
            output_dir / "evaluation.png",
        )
    )
    return outputs


def _plot_series(
    plt,
    steps: list[int],
    rows: list[dict],
    names: Iterable[str],
    title: str,
    output: Path,
) -> list[Path]:
    plotted = False
    figure, axis = plt.subplots()
    for name in names:
        points = [
            (step, row[name])
            for step, row in zip(steps, rows)
            if name in row
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=name,
            )
            plotted = True
    if not plotted:
        plt.close(figure)
        return []
    axis.set_title(title)
    axis.set_xlabel("iteration")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return [output]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    for path in create_plots(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
