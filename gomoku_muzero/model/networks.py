"""Small neural networks for MuZero's learned model."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InitialInferenceOutput(NamedTuple):
    """Outputs produced from a real board observation."""

    hidden_state: Tensor
    policy_logits: Tensor
    value: Tensor


class RecurrentInferenceOutput(NamedTuple):
    """Outputs produced after taking an action in latent space."""

    hidden_state: Tensor
    reward: Tensor
    policy_logits: Tensor
    value: Tensor


class RepresentationNetwork(nn.Module):
    """MuZero's h: encode an observation as a spatial hidden state."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Map ``[B, 3, N, N]`` to ``[B, C, N, N]``."""
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation must have shape [B, 3, N, N]")
        return self.net(observation)


class DynamicsNetwork(nn.Module):
    """MuZero's g: predict the next hidden state and immediate reward."""

    def __init__(
        self,
        board_size: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_space_size = board_size * board_size
        self.transition = nn.Sequential(
            nn.Conv2d(
                hidden_channels + 1,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_space_size, 1),
        )

    def forward(
        self, hidden_state: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Map ``[B,C,N,N]`` and ``[B]`` to next state and reward ``[B,1]``."""
        self._validate_inputs(hidden_state, action)
        action_plane = F.one_hot(
            action.to(torch.long), num_classes=self.action_space_size
        )
        action_plane = action_plane.to(dtype=hidden_state.dtype)
        action_plane = action_plane.view(
            hidden_state.shape[0], 1, self.board_size, self.board_size
        )

        next_hidden_state = self.transition(
            torch.cat((hidden_state, action_plane), dim=1)
        )
        reward = self.reward_head(next_hidden_state)
        return next_hidden_state, reward

    def _validate_inputs(self, hidden_state: Tensor, action: Tensor) -> None:
        expected_spatial = (self.board_size, self.board_size)
        if hidden_state.ndim != 4 or hidden_state.shape[2:] != expected_spatial:
            raise ValueError(
                f"hidden_state must have spatial shape {expected_spatial}"
            )
        if action.ndim != 1 or action.shape[0] != hidden_state.shape[0]:
            raise ValueError("action must have shape [B]")
        if action.is_floating_point():
            raise TypeError("action must have an integer dtype")
        if torch.any(action < 0) or torch.any(action >= self.action_space_size):
            raise ValueError(
                f"actions must be in [0, {self.action_space_size})"
            )


class PredictionNetwork(nn.Module):
    """MuZero's f: predict policy logits and value from a hidden state."""

    def __init__(
        self,
        board_size: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        action_space_size = board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * action_space_size, action_space_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(action_space_size, 1),
            nn.Tanh(),
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Map ``[B,C,N,N]`` to logits ``[B,N*N]`` and value ``[B,1]``."""
        return self.policy_head(hidden_state), self.value_head(hidden_state)


class MuZeroNetwork(nn.Module):
    """Compose h, g, and f behind the two inference operations MuZero uses."""

    def __init__(
        self,
        board_size: int = 10,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if board_size < 1:
            raise ValueError("board_size must be positive")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")

        self.board_size = board_size
        self.hidden_channels = hidden_channels
        self.action_space_size = board_size * board_size
        self.representation = RepresentationNetwork(hidden_channels)
        self.dynamics = DynamicsNetwork(board_size, hidden_channels)
        self.prediction = PredictionNetwork(board_size, hidden_channels)

    def initial_inference(self, observation: Tensor) -> InitialInferenceOutput:
        """Run h then f at the root of a search or training unroll."""
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return InitialInferenceOutput(hidden_state, policy_logits, value)

    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor
    ) -> RecurrentInferenceOutput:
        """Run g then f for one imagined transition."""
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return RecurrentInferenceOutput(
            next_hidden_state, reward, policy_logits, value
        )
