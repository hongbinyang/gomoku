import numpy as np
import pytest
import torch

from gomoku_muzero.model.checkpoint import load_checkpoint, save_checkpoint
from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.cli.play import parse_human_action, render_board


def test_checkpoint_round_trip_preserves_network_and_metadata(tmp_path) -> None:
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=3, hidden_channels=7)
    path = tmp_path / "model.pt"

    save_checkpoint(network, path, win_length=3)
    loaded = load_checkpoint(path)

    assert loaded.board_size == 3
    assert loaded.win_length == 3
    assert loaded.network.hidden_channels == 7
    for expected, actual in zip(
        network.parameters(), loaded.network.parameters()
    ):
        assert torch.equal(expected, actual)
    assert not loaded.network.training


def test_render_board_includes_coordinates_and_stones() -> None:
    env = GomokuEnv(board_size=3, win_length=3)
    env.step(4)

    rendered = render_board(env)

    assert "0 1 2" in rendered
    assert "X" in rendered


def test_parse_human_action_accepts_legal_coordinates() -> None:
    env = GomokuEnv(board_size=3, win_length=3)

    assert parse_human_action("1 2", env) == 5


@pytest.mark.parametrize("text", ["hello", "1", "3 0", "0 -1"])
def test_parse_human_action_rejects_bad_input(text: str) -> None:
    env = GomokuEnv(board_size=3, win_length=3)

    with pytest.raises(ValueError):
        parse_human_action(text, env)


def test_parse_human_action_rejects_occupied_cell() -> None:
    env = GomokuEnv(board_size=3, win_length=3)
    env.step(0)

    with pytest.raises(ValueError, match="occupied"):
        parse_human_action("0 0", env)
