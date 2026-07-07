import numpy as np
import pytest

from gomoku_muzero.game.env import GomokuEnv


def test_default_game_is_ten_by_ten_five_in_a_row() -> None:
    env = GomokuEnv()

    assert env.board_size == 10
    assert env.win_length == 5
    assert env.action_space_size == 100


def play(env: GomokuEnv, actions: list[int]) -> tuple[float, bool]:
    reward, terminated = 0.0, False
    for action in actions:
        _, reward, terminated, _ = env.step(action)
    return reward, terminated


def test_reset_and_observation_shape() -> None:
    env = GomokuEnv(board_size=5, win_length=5)

    observation = env.reset()

    assert observation.shape == (3, 5, 5)
    assert observation.dtype == np.float32
    assert observation[:2].sum() == 0
    assert observation[2].sum() == 25  # black moves first
    assert env.legal_actions() == list(range(25))


def test_step_places_stone_and_changes_player() -> None:
    env = GomokuEnv(board_size=5, win_length=5)

    observation, reward, terminated, info = env.step(7)

    assert env.board[1, 2] == env.BLACK
    assert reward == 0.0
    assert not terminated
    assert info["acting_player"] == env.BLACK
    assert observation[2].sum() == 0  # white is now to play
    assert 7 not in env.legal_actions()


@pytest.mark.parametrize(
    "black_actions",
    [
        [0, 1, 2, 3, 4],       # horizontal
        [0, 5, 10, 15, 20],    # vertical
        [0, 6, 12, 18, 24],    # main diagonal
        [4, 8, 12, 16, 20],    # anti-diagonal
    ],
)
def test_detects_wins_in_all_directions(black_actions: list[int]) -> None:
    env = GomokuEnv(board_size=5, win_length=5)
    white_fillers = [action for action in [5, 6, 7, 8, 9, 10, 11, 13, 14]
                     if action not in black_actions]
    actions: list[int] = []
    for index, black_action in enumerate(black_actions):
        actions.append(black_action)
        if index < len(black_actions) - 1:
            actions.append(white_fillers[index])

    reward, terminated = play(env, actions)

    assert terminated
    assert reward == 1.0
    assert env.winner == env.BLACK
    assert env.legal_actions() == []


def test_detects_draw() -> None:
    env = GomokuEnv(board_size=3, win_length=3)
    # X O X
    # X O O
    # O X X
    reward, terminated = play(env, [0, 1, 2, 4, 3, 5, 7, 6, 8])

    assert terminated
    assert reward == 0.0
    assert env.winner == env.EMPTY


def test_rejects_illegal_actions_and_steps_after_game() -> None:
    env = GomokuEnv(board_size=5, win_length=5)
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)
    with pytest.raises(ValueError):
        env.step(25)

    play(env, [5, 1, 6, 2, 7, 3, 8, 4])
    with pytest.raises(RuntimeError):
        env.step(9)


def test_terminal_move_still_advances_current_player() -> None:
    """The to-play plane must stay consistent with alternating perspective."""
    env = GomokuEnv(board_size=5, win_length=5)
    play(env, [0, 5, 1, 6, 2, 7, 3, 8, 4])  # black wins on the top row

    assert env.winner == env.BLACK
    assert env.current_player == env.WHITE
    assert env.observation()[2].sum() == 0  # white shown as next to play


def test_winning_actions_finds_completing_cells() -> None:
    env = GomokuEnv(board_size=3, win_length=3)
    play(env, [0, 8, 1])  # black holds 0, 1; white holds 8

    assert env.winning_actions(env.BLACK) == [2]
    assert env.winning_actions(env.WHITE) == []
    # The probe restores the board.
    assert env.board[0, 2] == env.EMPTY


def test_winning_actions_empty_after_termination() -> None:
    env = GomokuEnv(board_size=3, win_length=3)
    play(env, [0, 8, 1, 7, 2])  # black completes the top row

    assert env.winning_actions(env.WHITE) == []


def test_clone_is_independent() -> None:
    env = GomokuEnv(board_size=5, win_length=5)
    env.step(0)

    clone = env.clone()
    clone.step(1)

    assert env.board[0, 1] == env.EMPTY
