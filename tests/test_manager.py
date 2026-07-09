import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import torch

from gomoku_muzero.manager.server import ManagerServer, list_training_states
from gomoku_muzero.manager.training import TrainingManager, training_options
from gomoku_muzero.model.checkpoint import save_checkpoint
from gomoku_muzero.model.networks import MuZeroNetwork


def test_options_mirror_the_real_parser() -> None:
    options = {entry["name"]: entry for entry in training_options()}

    assert options["iterations"]["kind"] == "int"
    assert options["learning_rate"]["kind"] == "float"
    assert options["tensorboard"]["kind"] == "flag"
    assert options["replay_sampling"]["choices"] == ["recent", "uniform"]
    assert options["simulations"]["flag"] == "--simulations"
    # Removed options must not resurface.
    for gone in ("dirichlet_alpha", "unroll_steps", "runs_dir"):
        assert gone not in options


def test_unknown_option_is_rejected(tmp_path) -> None:
    manager = TrainingManager(tmp_path)

    with pytest.raises(ValueError, match="unknown training option"):
        manager.start({"rm_rf": "/"})


def test_run_progress_reads_metrics(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"iterations": 10, "resume": None})
    )
    rows = [
        {"step": 6, "loss": 5.0, "policy_kl": 1.2},
        {"step": 7, "loss": 4.5, "value_calibration_mae": 0.7},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    progress = TrainingManager(tmp_path).run_progress("demo")

    assert progress["iteration"] == 7
    assert progress["first_iteration"] == 6
    assert progress["target_iteration"] == 15  # resumed runs continue
    assert progress["latest_metrics"]["loss"] == 4.5
    assert progress["latest_metrics"]["policy_kl"] == 1.2
    assert progress["latest_metrics"]["value_calibration_mae"] == 0.7


def test_training_lifecycle_end_to_end(tmp_path, monkeypatch) -> None:
    """Start a real (tiny) training subprocess and watch it finish."""
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).resolve().parents[1])
    )
    manager = TrainingManager(tmp_path)
    status = manager.start(
        {
            "run_name": "console-smoke",
            "iterations": 1,
            "simulations": 2,
            "board_size": 4,
            "win_length": 3,
            "hidden_channels": 4,
            "res_blocks": 1,
            "games_per_iteration": 1,
            "training_steps": 1,
            "batch_size": 2,
            "evaluation_interval": 9,
            "self_play_mode": "sync",
            "device": "cpu",
        }
    )
    assert status["state"] == "running"
    assert status["run_name"] == "console-smoke"

    with pytest.raises(ValueError, match="already active"):
        manager.start({"iterations": 1})

    deadline = time.time() + 120
    while manager.status()["state"] == "running" and time.time() < deadline:
        time.sleep(1)
    status = manager.status()

    assert status["state"] == "finished", status.get("log_tail")
    assert status["iteration"] == 1
    assert status["target_iteration"] == 1
    assert status["log_tail"]
    assert (tmp_path / "runs" / "console-smoke" / "metrics.jsonl").exists()
    assert manager.active_run_artifacts() == set()


def test_list_training_states(tmp_path) -> None:
    (tmp_path / "a-state.pt").write_bytes(b"x")
    (tmp_path / "model.pt").write_bytes(b"x")

    assert list_training_states(tmp_path) == ["a-state.pt"]


def test_console_http_and_play_integration(tmp_path) -> None:
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=5, hidden_channels=8, num_blocks=1)
    save_checkpoint(network, tmp_path / "small.pt", win_length=4)
    server = ManagerServer(("127.0.0.1", 0), tmp_path, workdir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get_json(path):
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return json.loads(response.read())

    try:
        with urllib.request.urlopen(base + "/", timeout=30) as response:
            assert "Console" in response.read().decode()
        with urllib.request.urlopen(base + "/play", timeout=30) as response:
            assert "Copy game record" in response.read().decode()

        options = get_json("/manager/api/options")["options"]
        assert any(entry["name"] == "iterations" for entry in options)
        assert get_json("/manager/api/status") == {"state": "idle"}
        # The inherited play API still works untouched.
        assert get_json("/api/checkpoints")["checkpoints"] == ["small.pt"]

        request = urllib.request.Request(
            base + "/api/new",
            data=json.dumps(
                {
                    "checkpoint": "small.pt",
                    "human_color": "black",
                    "num_simulations": 50,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            assert json.loads(response.read())["to_play"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
