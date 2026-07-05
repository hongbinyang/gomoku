"""Generate complete Gomoku games with the current network and MCTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS
from gomoku_muzero.training.replay import GameHistory


@dataclass(frozen=True)
class SelfPlayConfig:
    """Controls exploration when selecting actions from root visits."""

    temperature: float = 1.0
    temperature_moves: int = 8
    add_exploration_noise: bool = True


def play_self_play_game(
    env: GomokuEnv,
    mcts: MCTS,
    config: SelfPlayConfig | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> GameHistory:
    """Play one complete game and turn it into replay-ready targets."""
    config = config or SelfPlayConfig()
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if config.temperature_moves < 0:
        raise ValueError("temperature_moves must be non-negative")

    observations = [env.reset()]
    actions: list[int] = []
    rewards: list[float] = []
    policies: list[np.ndarray] = []
    root_values: list[float] = []
    to_play = [env.current_player]

    while not env.terminated:
        move_number = len(actions) + 1
        root = mcts.run(
            observations[-1],
            env.legal_actions(),
            env.current_player,
            add_exploration_noise=config.add_exploration_noise,
            progress_callback=(
                None
                if progress_callback is None
                else lambda completed, total: progress_callback(
                    move_number, completed, total
                )
            ),
        )
        policies.append(mcts.policy_target(root))
        root_values.append(root.value)
        temperature = (
            config.temperature
            if len(actions) < config.temperature_moves
            else 0.0
        )
        action = mcts.select_action(root, temperature)
        observation, reward, _, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        observations.append(observation)
        # The environment flips current_player after every move, including
        # terminal ones, so this matches the observation's to-play plane.
        to_play.append(env.current_player)

    policies.append(
        np.zeros(env.action_space_size, dtype=np.float32)
    )
    root_values.append(0.0)
    outcome_for_black = env.winner
    values = [
        float(outcome_for_black * player)
        for player in to_play[:-1]
    ]
    values.append(0.0)

    game = GameHistory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        policies=policies,
        values=values,
        to_play=to_play,
        root_values=root_values,
    )
    game.validate(env.action_space_size)
    return game
