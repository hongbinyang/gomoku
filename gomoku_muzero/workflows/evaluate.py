"""Evaluate a MuZero agent against fixed baseline opponents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS

OpponentFn = Callable[[GomokuEnv, np.random.Generator], int]


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


def random_action(env: GomokuEnv, rng: np.random.Generator) -> int:
    """Play a uniformly random legal move."""
    return int(rng.choice(env.legal_actions()))


def heuristic_action(env: GomokuEnv, rng: np.random.Generator) -> int:
    """Play with one-move tactical awareness.

    Completes an immediate win when available, otherwise blocks the
    opponent's immediate win, otherwise plays randomly. This is the
    minimum opponent that punishes threat-blind play, which random
    evaluation cannot detect.
    """
    own_wins = env.winning_actions(env.current_player)
    if own_wins:
        return int(rng.choice(own_wins))
    opponent_wins = env.winning_actions(-env.current_player)
    if opponent_wins:
        return int(rng.choice(opponent_wins))
    return int(rng.choice(env.legal_actions()))


def evaluate_against_random(
    env: GomokuEnv,
    mcts: MCTS,
    num_games: int = 20,
    seed: int | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> EvaluationResult:
    """Play greedily against random moves, alternating the agent's color."""
    return _evaluate(
        env, mcts, random_action, num_games, seed, progress_callback
    )


def evaluate_against_heuristic(
    env: GomokuEnv,
    mcts: MCTS,
    num_games: int = 20,
    seed: int | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> EvaluationResult:
    """Play greedily against the win-or-block heuristic baseline."""
    return _evaluate(
        env, mcts, heuristic_action, num_games, seed, progress_callback
    )


def _evaluate(
    env: GomokuEnv,
    mcts: MCTS,
    opponent: OpponentFn,
    num_games: int,
    seed: int | None,
    progress_callback: Callable[[int, int, int, int], None] | None,
) -> EvaluationResult:
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
                    env=env,
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
                action = opponent(env, rng)
            env.step(action)

        if env.winner == env.EMPTY:
            draws += 1
        elif env.winner == agent_player:
            wins += 1
        else:
            losses += 1

    return EvaluationResult(wins, draws, losses)
