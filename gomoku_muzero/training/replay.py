"""Game storage and fixed-shape K-step samples for MuZero training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Observation = npt.NDArray[np.float32]
Policy = npt.NDArray[np.float32]


@dataclass
class GameHistory:
    """A complete self-play game, containing no neural hidden states.

    For ``T`` moves, state-indexed fields have length ``T + 1`` and
    transition-indexed fields have length ``T``:

    - observations, policies, values, to_play: positions 0 through T
    - actions, rewards: transitions 0 through T-1

    The policy at terminal position T is an all-zero vector because no search
    or action occurs there. Values are scalar training targets from the
    player-to-move perspective at each position. ``root_values`` optionally
    stores the MCTS root value at each searched position (with a trailing
    zero at position T) for value-calibration diagnostics and future
    reanalysis; it is not used as a training target.
    """

    observations: list[Observation]
    actions: list[int]
    rewards: list[float]
    policies: list[Policy]
    values: list[float]
    to_play: list[int]
    root_values: list[float] | None = None
    network_version: int | None = None

    def validate(self, action_space_size: int) -> None:
        """Reject inconsistent games when they enter the replay buffer."""
        num_moves = len(self.actions)
        num_positions = num_moves + 1
        if num_moves == 0:
            raise ValueError("a stored game must contain at least one move")
        if len(self.rewards) != num_moves:
            raise ValueError("rewards must have one entry per action")
        for name, sequence in (
            ("observations", self.observations),
            ("policies", self.policies),
            ("values", self.values),
            ("to_play", self.to_play),
        ):
            if len(sequence) != num_positions:
                raise ValueError(f"{name} must have T + 1 entries")

        observation_shape = self.observations[0].shape
        if len(observation_shape) != 3 or observation_shape[0] != 3:
            raise ValueError("observations must have shape [3, N, N]")
        if any(
            observation.shape != observation_shape
            for observation in self.observations
        ):
            raise ValueError("all observations must have the same shape")
        if any(
            observation.dtype != np.float32
            for observation in self.observations
        ):
            raise ValueError("observations must have dtype float32")

        if any(
            action < 0 or action >= action_space_size
            for action in self.actions
        ):
            raise ValueError("an action is outside the action space")
        if any(policy.shape != (action_space_size,) for policy in self.policies):
            raise ValueError("policies must have shape [action_space_size]")
        if any(policy.dtype != np.float32 for policy in self.policies):
            raise ValueError("policies must have dtype float32")
        if (
            self.root_values is not None
            and len(self.root_values) != num_positions
        ):
            raise ValueError("root_values must have T + 1 entries")
        if any(player not in (-1, 1) for player in self.to_play):
            raise ValueError("to_play entries must be -1 or +1")
        if any(
            self.to_play[index + 1] != -self.to_play[index]
            for index in range(num_moves)
        ):
            raise ValueError("to_play must alternate after every action")

    @property
    def num_moves(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class ReplayBatch:
    """A batch of fixed-shape K-step MuZero training samples."""

    observations: Observation
    actions: npt.NDArray[np.int64]
    target_rewards: npt.NDArray[np.float32]
    target_policies: Policy
    target_values: npt.NDArray[np.float32]
    dynamics_mask: npt.NDArray[np.float32]
    prediction_mask: npt.NDArray[np.float32]
    sampled_game_ages: npt.NDArray[np.int64]


class ReplayBuffer:
    """A bounded FIFO buffer of complete games."""

    def __init__(
        self,
        capacity: int,
        action_space_size: int,
        seed: int | None = None,
        sampling: str = "uniform",
        recency_half_life: float = 100.0,
        augment_symmetries: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if action_space_size < 1:
            raise ValueError("action_space_size must be positive")
        if sampling not in ("uniform", "recent"):
            raise ValueError("sampling must be 'uniform' or 'recent'")
        if recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")
        board_size = int(round(action_space_size**0.5))
        if augment_symmetries and board_size * board_size != action_space_size:
            raise ValueError(
                "augment_symmetries requires a square action space"
            )
        self.capacity = capacity
        self.action_space_size = action_space_size
        self.sampling = sampling
        self.recency_half_life = recency_half_life
        self.augment_symmetries = augment_symmetries
        self._symmetry_maps = (
            _board_symmetry_maps(board_size) if augment_symmetries else None
        )
        self.games: deque[GameHistory] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self._sample_age_sum = 0
        self._sample_age_count = 0
        self._sample_age_max = 0
        self._sampling_probabilities: npt.NDArray[np.float64] | None = None

    def save_game(self, game: GameHistory) -> None:
        """Validate and append a complete game, evicting the oldest if full."""
        game.validate(self.action_space_size)
        self.games.append(game)
        self._refresh_sampling_probabilities()

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int,
    ) -> ReplayBatch:
        """Sample games and positions, padding only beyond terminal states.

        Prediction targets have ``K + 1`` steps: one for initial inference and
        K for recurrent inference. Rewards and actions have K transitions.
        Masks are one for real entries and zero for absorbing padding beyond
        the terminal state. Padded steps carry uniformly random actions and
        zero value/reward targets so the trainer can supervise absorbing
        states the way the MuZero pseudocode does. When symmetry
        augmentation is enabled, each sample is transformed by one of the
        board's eight dihedral symmetries.
        """
        if not self.games:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_unroll_steps < 0:
            raise ValueError("num_unroll_steps must be non-negative")

        samples = [
            self._sample_position(num_unroll_steps)
            for _ in range(batch_size)
        ]
        fields = list(zip(*samples))
        return ReplayBatch(
            observations=np.stack(fields[0]).astype(np.float32),
            actions=np.stack(fields[1]).astype(np.int64),
            target_rewards=np.stack(fields[2]).astype(np.float32),
            target_policies=np.stack(fields[3]).astype(np.float32),
            target_values=np.stack(fields[4]).astype(np.float32),
            dynamics_mask=np.stack(fields[5]).astype(np.float32),
            prediction_mask=np.stack(fields[6]).astype(np.float32),
            sampled_game_ages=np.asarray(fields[7], dtype=np.int64),
        )

    def _sample_position(
        self, num_unroll_steps: int
    ) -> tuple[
        Observation,
        npt.NDArray[np.int64],
        npt.NDArray[np.float32],
        Policy,
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        int,
    ]:
        game, game_age = self._sample_game()
        start = int(self.rng.integers(game.num_moves))
        prediction_steps = num_unroll_steps + 1

        actions = np.zeros(num_unroll_steps, dtype=np.int64)
        rewards = np.zeros(num_unroll_steps, dtype=np.float32)
        policies = np.zeros(
            (prediction_steps, self.action_space_size), dtype=np.float32
        )
        values = np.zeros(prediction_steps, dtype=np.float32)
        dynamics_mask = np.zeros(num_unroll_steps, dtype=np.float32)
        prediction_mask = np.zeros(prediction_steps, dtype=np.float32)

        for step in range(prediction_steps):
            position = start + step
            if position <= game.num_moves:
                policies[step] = game.policies[position]
                values[step] = game.values[position]
                prediction_mask[step] = 1.0
            if step < num_unroll_steps:
                if position < game.num_moves:
                    actions[step] = game.actions[position]
                    rewards[step] = game.rewards[position]
                    dynamics_mask[step] = 1.0
                else:
                    # Absorbing padding: any action should keep the value
                    # and reward at zero, so pad with random actions rather
                    # than biasing the dynamics toward action 0.
                    actions[step] = int(
                        self.rng.integers(self.action_space_size)
                    )

        observation = game.observations[start]
        if self._symmetry_maps is not None:
            observation, actions, policies = self._apply_symmetry(
                observation, actions, policies
            )
        return (
            observation,
            actions,
            rewards,
            policies,
            values,
            dynamics_mask,
            prediction_mask,
            game_age,
        )

    def _apply_symmetry(
        self,
        observation: Observation,
        actions: npt.NDArray[np.int64],
        policies: Policy,
    ) -> tuple[Observation, npt.NDArray[np.int64], Policy]:
        """Apply one random dihedral symmetry consistently to a sample."""
        source, action_map = self._symmetry_maps[
            int(self.rng.integers(len(self._symmetry_maps)))
        ]
        channels = observation.shape[0]
        board_shape = observation.shape[1:]
        observation = (
            observation.reshape(channels, -1)[:, source]
            .reshape(channels, *board_shape)
            .copy()
        )
        actions = action_map[actions]
        policies = policies[:, source].copy()
        return observation, actions, policies

    def sampling_metrics(self, reset: bool = True) -> dict[str, float]:
        """Return observed replay ages since the previous reset."""
        if self._sample_age_count == 0:
            metrics = {
                "replay_sample_age_mean": 0.0,
                "replay_sample_age_max": 0.0,
            }
        else:
            metrics = {
                "replay_sample_age_mean": (
                    self._sample_age_sum / self._sample_age_count
                ),
                "replay_sample_age_max": float(self._sample_age_max),
            }
        if reset:
            self._sample_age_sum = 0
            self._sample_age_count = 0
            self._sample_age_max = 0
        return metrics

    def _sample_game(self) -> tuple[GameHistory, int]:
        num_games = len(self.games)
        if self.sampling == "uniform":
            index = int(self.rng.integers(num_games))
        else:
            index = int(
                self.rng.choice(
                    num_games, p=self._sampling_probabilities
                )
            )
        age = num_games - 1 - index
        self._sample_age_sum += age
        self._sample_age_count += 1
        self._sample_age_max = max(self._sample_age_max, age)
        return self.games[index], age

    def _refresh_sampling_probabilities(self) -> None:
        if self.sampling == "uniform":
            self._sampling_probabilities = None
            return
        # Deque index 0 is oldest; the newest game's age is zero. Recompute
        # only when games enter the buffer, not for every sampled position.
        ages = np.arange(len(self.games) - 1, -1, -1, dtype=np.float64)
        weights = np.power(0.5, ages / self.recency_half_life)
        self._sampling_probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return len(self.games)


def _board_symmetry_maps(
    board_size: int,
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Precompute the eight dihedral symmetries of a square board.

    Each entry is ``(source, action_map)`` where ``source[k]`` is the old
    flattened cell shown at new position ``k`` (gather order, used for
    observations and policy vectors) and ``action_map[a]`` is the new index
    of old action ``a`` (scatter order, used for action indices).
    """
    grid = np.arange(board_size * board_size, dtype=np.int64).reshape(
        board_size, board_size
    )
    maps = []
    for flip in (False, True):
        base = np.fliplr(grid) if flip else grid
        for quarter_turns in range(4):
            source = np.rot90(base, quarter_turns).ravel()
            action_map = np.empty_like(source)
            action_map[source] = np.arange(source.size, dtype=np.int64)
            maps.append((source, action_map))
    return maps
