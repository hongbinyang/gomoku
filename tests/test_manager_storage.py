import json

import pytest
import torch

from gomoku_muzero.manager.metrics import list_runs, metric_series
from gomoku_muzero.manager.storage import (
    delete_item,
    list_storage,
    model_info,
)
from gomoku_muzero.model.checkpoint import save_checkpoint
from gomoku_muzero.model.networks import MuZeroNetwork


@pytest.fixture()
def workspace(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=5, hidden_channels=8, num_blocks=2)
    save_checkpoint(network, checkpoints / "small.pt", win_length=4)
    (checkpoints / "small-state.pt").write_bytes(b"not-a-real-state")

    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "iterations": 3,
                "resume": None,
                "checkpoint": "checkpoints/small.pt",
                "board_size": 5,
                "win_length": 4,
                "simulations": 25,
                "seed": 0,
            }
        )
    )
    rows = [
        {"event": "iteration", "step": 1, "loss": 5.0,
         "value_calibration_mae": 1.0},
        {"event": "iteration", "step": 2, "loss": 4.0,
         "evaluation_score": 0.5},
        {"event": "iteration", "step": 3, "loss": 3.0,
         "policy_kl": 1.5},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    return tmp_path


def test_list_storage(workspace) -> None:
    data = list_storage(workspace, workspace / "checkpoints")

    assert [run["name"] for run in data["runs"]] == ["demo"]
    assert data["runs"][0]["first_iteration"] == 1
    assert data["runs"][0]["last_iteration"] == 3
    kinds = {item["name"]: item["kind"] for item in data["checkpoints"]}
    assert kinds == {
        "small.pt": "model",
        "small-state.pt": "training-state",
    }


def test_model_info_cross_references_runs(workspace) -> None:
    info = model_info(workspace / "checkpoints", "small.pt", workspace)

    assert info["kind"] == "model"
    assert info["board_size"] == 5
    assert info["win_length"] == 4
    assert info["hidden_channels"] == 8
    assert info["num_blocks"] == 2
    assert info["parameters"] > 0
    assert info["total_iterations"] == 3
    trained = info["trained_by"][0]
    assert trained["run"] == "demo"
    assert trained["iterations"] == (1, 3)
    assert trained["options"]["simulations"] == 25


def test_model_info_summarizes_states_without_loading(workspace) -> None:
    info = model_info(
        workspace / "checkpoints", "small-state.pt", workspace
    )

    assert info["kind"] == "training-state"
    assert "note" in info


def test_delete_requires_exact_confirmation(workspace) -> None:
    with pytest.raises(ValueError, match="confirmation"):
        delete_item(
            "run", "demo", "Demo", workspace,
            workspace / "checkpoints", set(),
        )

    result = delete_item(
        "run", "demo", "demo", workspace,
        workspace / "checkpoints", set(),
    )
    assert result == {"deleted": "demo", "kind": "run"}
    assert not (workspace / "runs" / "demo").exists()


def test_delete_refuses_active_artifacts_and_escapes(workspace) -> None:
    protected = {"runs/demo", "checkpoints/small.pt"}

    with pytest.raises(ValueError, match="active training"):
        delete_item(
            "run", "demo", "demo", workspace,
            workspace / "checkpoints", protected,
        )
    with pytest.raises(ValueError, match="active training"):
        delete_item(
            "checkpoint", "small.pt", "small.pt", workspace,
            workspace / "checkpoints", protected,
        )
    with pytest.raises(ValueError, match="invalid name"):
        delete_item(
            "checkpoint", "../evil.pt", "../evil.pt", workspace,
            workspace / "checkpoints", set(),
        )


def test_generate_plots_works_from_a_worker_thread(workspace) -> None:
    """Regression: GUI matplotlib backends crash off the main thread."""
    pytest.importorskip("matplotlib")
    import threading

    from gomoku_muzero.manager.tools import generate_plots

    results: dict = {}

    def worker():
        try:
            results["files"] = generate_plots(workspace, "demo")
        except Exception as error:  # noqa: BLE001
            results["error"] = error

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)

    assert "error" not in results, results.get("error")
    assert "losses.png" in results["files"]
    plots = workspace / "runs" / "demo" / "plots"
    assert (plots / "losses.png").stat().st_size > 0


def test_tensorboard_refuses_occupied_port(tmp_path) -> None:
    import socket

    from gomoku_muzero.manager.tools import TensorBoardManager

    pytest.importorskip("tensorboard")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        manager = TensorBoardManager(tmp_path, port=port)
        with pytest.raises(ValueError, match="already in use"):
            manager.start()
    finally:
        blocker.close()


def test_metric_series_and_run_listing(workspace) -> None:
    runs = list_runs(workspace)
    assert runs[0]["name"] == "demo"
    assert runs[0]["last_iteration"] == 3

    data = metric_series(workspace, ["demo"])
    assert data["series"]["loss"]["demo"] == [
        [1, 5.0], [2, 4.0], [3, 3.0],
    ]
    assert data["series"]["policy_kl"]["demo"] == [[3, 1.5]]
    assert data["series"]["evaluation_score"]["demo"] == [[2, 0.5]]
