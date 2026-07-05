"""Gomoku rules and the self-play-facing environment API."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Board = npt.NDArray[np.int8]
Observation = npt.NDArray[np.float32]


class GomokuEnv:
    """A minimal, deterministic, two-player Gomoku environment.

    Board cells use 0 for empty, +1 for black, and -1 for white.
    Black always moves first. An action is a flattened board index:
    ``action = row * board_size + column``.
    """

    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, board_size: int = 10, win_length: int = 5) -> None:
        if board_size < 1:
            raise ValueError("board_size must be positive")
        if not 1 <= win_length <= board_size:
            raise ValueError("win_length must be between 1 and board_size")

        self.board_size = board_size
        self.win_length = win_length
        self.action_space_size = board_size * board_size
        self.board: Board
        self.current_player: int
        self.winner: int
        self.terminated: bool
        self.move_count: int
        self.reset()

    def reset(self) -> Observation:
        """Start a new game and return its initial observation."""
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.int8
        )
        self.current_player = self.BLACK
        self.winner = self.EMPTY
        self.terminated = False
        self.move_count = 0
        return self.observation()

    def observation(self) -> Observation:
        """Return shape ``(3, board_size, board_size)`` as float32.

        Channel 0 marks black stones, channel 1 marks white stones, and
        channel 2 is all ones for black-to-play and all zeros for white-to-play.
        This retains the complete Markov state without exposing mutable board
        storage to callers.
        """
        black = self.board == self.BLACK
        white = self.board == self.WHITE
        to_play = np.full_like(black, self.current_player == self.BLACK)
        return np.stack((black, white, to_play)).astype(np.float32)

    def legal_actions(self) -> list[int]:
        """Return legal flattened actions in ascending order."""
        if self.terminated:
            return []
        return np.flatnonzero(self.board.ravel() == self.EMPTY).tolist()

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Play one action.

        The returned reward belongs to the player who took ``action``:
        1.0 for a winning move and 0.0 otherwise. Gomoku has no intermediate
        rewards. ``current_player`` advances after every move, including a
        terminal one, so the observation's to-play plane always describes the
        player who would move next. Use ``winner`` for the game result.
        """
        if self.terminated:
            raise RuntimeError("cannot step a terminated game")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not 0 <= int(action) < self.action_space_size:
            raise ValueError(f"action must be in [0, {self.action_space_size})")

        row, column = divmod(int(action), self.board_size)
        if self.board[row, column] != self.EMPTY:
            raise ValueError(f"cell ({row}, {column}) is already occupied")

        acting_player = self.current_player
        self.board[row, column] = acting_player
        self.move_count += 1

        if self._is_winning_move(row, column):
            self.winner = acting_player
            self.terminated = True
            reward = 1.0
        elif self.move_count == self.action_space_size:
            self.terminated = True
            reward = 0.0
        else:
            reward = 0.0
        self.current_player = -self.current_player

        info = {
            "acting_player": acting_player,
            "winner": self.winner,
            "legal_actions": self.legal_actions(),
        }
        return self.observation(), reward, self.terminated, info

    def clone(self) -> GomokuEnv:
        """Return an independent copy, useful for tests and self-play."""
        copy = GomokuEnv(self.board_size, self.win_length)
        copy.board = self.board.copy()
        copy.current_player = self.current_player
        copy.winner = self.winner
        copy.terminated = self.terminated
        copy.move_count = self.move_count
        return copy

    def _is_winning_move(self, row: int, column: int) -> bool:
        player = int(self.board[row, column])
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for row_delta, column_delta in directions:
            line_length = 1
            line_length += self._count_stones(
                row, column, row_delta, column_delta, player
            )
            line_length += self._count_stones(
                row, column, -row_delta, -column_delta, player
            )
            if line_length >= self.win_length:
                return True
        return False

    def _count_stones(
        self,
        row: int,
        column: int,
        row_delta: int,
        column_delta: int,
        player: int,
    ) -> int:
        count = 0
        row += row_delta
        column += column_delta
        while (
            0 <= row < self.board_size
            and 0 <= column < self.board_size
            and self.board[row, column] == player
        ):
            count += 1
            row += row_delta
            column += column_delta
        return count
