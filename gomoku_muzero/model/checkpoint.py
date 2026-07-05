"""Save and load inference-ready MuZero checkpoints and training state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.training.replay import GameHistory


@dataclass(frozen=True)
class LoadedCheckpoint:
    network: MuZeroNetwork
    board_size: int
    win_length: int


@dataclass(frozen=True)
class LoadedTrainingState:
    """Everything required to continue an interrupted training run."""

    network: MuZeroNetwork
    optimizer_state: dict[str, Any]
    iteration: int
    games: list[GameHistory]
    board_size: int
    win_length: int
    hidden_channels: int


def save_checkpoint(
    network: MuZeroNetwork,
    path: str | Path,
    win_length: int,
) -> Path:
    """Save weights plus the architecture and game metadata needed to play."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in network.state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "board_size": network.board_size,
            "win_length": win_length,
            "hidden_channels": network.hidden_channels,
            "model_state_dict": cpu_state_dict,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Recreate a network and load its trained parameters."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if data.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")

    board_size = int(data["board_size"])
    win_length = int(data["win_length"])
    hidden_channels = int(data["hidden_channels"])
    network = MuZeroNetwork(board_size, hidden_channels)
    network.load_state_dict(data["model_state_dict"])
    network.to(device)
    network.eval()
    return LoadedCheckpoint(network, board_size, win_length)


def save_training_state(
    path: str | Path,
    network: MuZeroNetwork,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    games: list[GameHistory],
    win_length: int,
) -> Path:
    """Save everything needed to resume training after an interruption.

    Replay games are converted to tensors so the file loads with
    ``weights_only=True``. The file is written atomically via a temporary
    sibling so an interrupted save never corrupts the previous state.
    """
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "iteration": iteration,
        "board_size": network.board_size,
        "win_length": win_length,
        "hidden_channels": network.hidden_channels,
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in network.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "games": [_game_to_tensors(game) for game in games],
    }
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(state_path)
    return state_path


def load_training_state(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedTrainingState:
    """Load a saved training state for resumption."""
    state_path = Path(path)
    if not state_path.is_file():
        raise FileNotFoundError(f"training state not found: {state_path}")
    data = torch.load(state_path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("unsupported training state format")

    board_size = int(data["board_size"])
    network = MuZeroNetwork(board_size, int(data["hidden_channels"]))
    network.load_state_dict(data["model_state_dict"])
    network.to(device)
    return LoadedTrainingState(
        network=network,
        optimizer_state=data["optimizer_state_dict"],
        iteration=int(data["iteration"]),
        games=[_game_from_tensors(entry) for entry in data["games"]],
        board_size=board_size,
        win_length=int(data["win_length"]),
        hidden_channels=int(data["hidden_channels"]),
    )


def _game_to_tensors(game: GameHistory) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "observations": torch.from_numpy(np.stack(game.observations)),
        "actions": torch.tensor(game.actions, dtype=torch.int64),
        "rewards": torch.tensor(game.rewards, dtype=torch.float32),
        "policies": torch.from_numpy(np.stack(game.policies)),
        "values": torch.tensor(game.values, dtype=torch.float32),
        "to_play": torch.tensor(game.to_play, dtype=torch.int64),
    }
    if game.root_values is not None:
        entry["root_values"] = torch.tensor(
            game.root_values, dtype=torch.float32
        )
    if game.network_version is not None:
        entry["network_version"] = game.network_version
    return entry


def _game_from_tensors(entry: dict[str, Any]) -> GameHistory:
    root_values = entry.get("root_values")
    return GameHistory(
        observations=list(entry["observations"].numpy()),
        actions=[int(action) for action in entry["actions"]],
        rewards=[float(reward) for reward in entry["rewards"]],
        policies=list(entry["policies"].numpy()),
        values=[float(value) for value in entry["values"]],
        to_play=[int(player) for player in entry["to_play"]],
        root_values=(
            [float(value) for value in root_values]
            if root_values is not None
            else None
        ),
        network_version=entry.get("network_version"),
    )
