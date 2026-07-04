"""Save and load inference-ready MuZero checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from gomoku_muzero.model.networks import MuZeroNetwork


@dataclass(frozen=True)
class LoadedCheckpoint:
    network: MuZeroNetwork
    board_size: int
    win_length: int


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
