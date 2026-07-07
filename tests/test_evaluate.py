import numpy as np

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.workflows.evaluate import heuristic_action, random_action


def make_env(actions: list[int]) -> GomokuEnv:
    env = GomokuEnv(board_size=3, win_length=3)
    for action in actions:
        env.step(action)
    return env


def test_heuristic_completes_its_own_win() -> None:
    # Black holds 0, 1 and is to move: it must complete at 2.
    env = make_env([0, 8, 1, 7])

    assert heuristic_action(env, np.random.default_rng(0)) == 2


def test_heuristic_blocks_opponent_win() -> None:
    # Black holds 0, 1; white to move must block at 2.
    env = make_env([0, 8, 1])

    assert heuristic_action(env, np.random.default_rng(0)) == 2


def test_heuristic_prefers_winning_over_blocking() -> None:
    # Both sides threaten: black completes at 2, white completes at 6.
    # Black to move should win, not block.
    env = make_env([0, 8, 1, 7])
    assert env.winning_actions(env.WHITE) == [6]

    assert heuristic_action(env, np.random.default_rng(0)) == 2


def test_fallbacks_stay_legal() -> None:
    env = make_env([4])
    rng = np.random.default_rng(0)

    for _ in range(20):
        assert heuristic_action(env, rng) in env.legal_actions()
        assert random_action(env, rng) in env.legal_actions()
