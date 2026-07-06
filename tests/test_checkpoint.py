import numpy as np
import pytest
import torch

from gomoku_muzero.model.checkpoint import (
    load_training_state,
    save_training_state,
)
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.training.replay import GameHistory
from gomoku_muzero.training.trainer import MuZeroTrainer


def make_game() -> GameHistory:
    return GameHistory(
        observations=[
            np.zeros((3, 2, 2), dtype=np.float32),
            np.ones((3, 2, 2), dtype=np.float32),
        ],
        actions=[2],
        rewards=[1.0],
        policies=[
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ],
        values=[1.0, 0.0],
        to_play=[1, -1],
        root_values=[0.5, 0.0],
        network_version=3,
    )


def test_training_state_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=2, hidden_channels=4, num_blocks=2)
    trainer = MuZeroTrainer(network, learning_rate=1e-2)
    path = tmp_path / "training-state.pt"

    save_training_state(
        path,
        network,
        trainer.optimizer,
        iteration=7,
        games=[make_game()],
        win_length=2,
    )
    loaded = load_training_state(path)

    assert loaded.iteration == 7
    assert loaded.board_size == 2
    assert loaded.win_length == 2
    assert loaded.hidden_channels == 4
    assert loaded.num_blocks == 2
    assert loaded.network.num_blocks == 2
    for expected, actual in zip(
        network.parameters(), loaded.network.parameters()
    ):
        assert torch.equal(expected, actual)

    assert len(loaded.games) == 1
    game = loaded.games[0]
    game.validate(action_space_size=4)
    assert game.actions == [2]
    assert game.root_values == [0.5, 0.0]
    assert game.network_version == 3
    np.testing.assert_array_equal(
        game.observations[1], np.ones((3, 2, 2), dtype=np.float32)
    )

    # Optimizer state restores into a fresh trainer without error.
    resumed_trainer = MuZeroTrainer(loaded.network, learning_rate=1e-2)
    resumed_trainer.optimizer.load_state_dict(loaded.optimizer_state)


def test_load_training_state_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_training_state(tmp_path / "missing.pt")
