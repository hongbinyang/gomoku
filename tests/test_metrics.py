import csv
import json

import pytest

from gomoku_muzero.cli.plot import load_iteration_metrics
from gomoku_muzero.runtime.metrics import RunLogger


def test_run_logger_writes_config_jsonl_and_csv(tmp_path) -> None:
    logger = RunLogger(tmp_path, "test-run")
    logger.write_config({"board_size": 10, "device": "cpu"})
    logger.record("iteration", 1, {"loss": 2.0})
    logger.record("iteration", 2, {"loss": 1.5, "evaluation_score": 0.75})
    logger.close()

    run_dir = tmp_path / "test-run"
    config = json.loads((run_dir / "config.json").read_text())
    jsonl_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    with (run_dir / "metrics.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))

    assert config["board_size"] == 10
    assert [row["loss"] for row in jsonl_rows] == [2.0, 1.5]
    assert csv_rows[1]["evaluation_score"] == "0.75"


def test_run_name_rejects_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_name"):
        RunLogger(tmp_path, "../escape")


def test_existing_run_is_not_overwritten(tmp_path) -> None:
    RunLogger(tmp_path, "same")

    with pytest.raises(FileExistsError):
        RunLogger(tmp_path, "same")


def test_plot_loader_returns_only_iteration_events(tmp_path) -> None:
    logger = RunLogger(tmp_path, "run")
    logger.record("startup", 0, {"value": 1})
    logger.record("iteration", 1, {"loss": 2})
    logger.close()

    rows = load_iteration_metrics(tmp_path / "run")

    assert len(rows) == 1
    assert rows[0]["loss"] == 2
