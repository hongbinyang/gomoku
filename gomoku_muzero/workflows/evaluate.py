"""Evaluate a MuZero agent against a fixed random-action baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS


@dataclass(frozen=True)
class EvaluationResult:
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        """Mean match score using win=1, draw=0.5, loss=0."""
        total = self.wins + self.draws + self.losses
        return (self.wins + 0.5 * self.draws) / total


def evaluate_against_random(
    env: GomokuEnv,
    mcts: MCTS,
    num_games: int = 20,
    seed: int | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> EvaluationResult:
    """Play greedily against random, alternating the agent's color."""
    if num_games < 1:
        raise ValueError("num_games must be positive")
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0

    for game_index in range(num_games):
        env.reset()
        agent_player = env.BLACK if game_index % 2 == 0 else env.WHITE
        while not env.terminated:
            if env.current_player == agent_player:
                root = mcts.run(
                    env.observation(),
                    env.legal_actions(),
                    env.current_player,
                    add_exploration_noise=False,
                    progress_callback=(
                        None
                        if progress_callback is None
                        else lambda completed, total: progress_callback(
                            game_index + 1,
                            num_games,
                            completed,
                            total,
                        )
                    ),
                )
                action = mcts.select_action(root, temperature=0)
            else:
                action = int(rng.choice(env.legal_actions()))
            env.step(action)

        if env.winner == env.EMPTY:
            draws += 1
        elif env.winner == agent_player:
            wins += 1
        else:
            losses += 1

    return EvaluationResult(wins, draws, losses)
