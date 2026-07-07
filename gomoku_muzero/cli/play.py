"""Play Gomoku against a trained MuZero checkpoint in the terminal."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from gomoku_muzero.model.checkpoint import load_checkpoint
from gomoku_muzero.runtime.device import resolve_device
from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS, MCTSConfig

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def render_board(env: GomokuEnv) -> str:
    """Render the board with zero-based row and column coordinates."""
    width = len(str(env.board_size - 1))
    header = " " * (width + 1) + " ".join(
        f"{column:>{width}}" for column in range(env.board_size)
    )
    symbols = {
        env.EMPTY: ".",
        env.BLACK: "X",
        env.WHITE: "O",
    }
    rows = [header]
    for row in range(env.board_size):
        cells = " ".join(
            f"{symbols[int(env.board[row, column])]:>{width}}"
            for column in range(env.board_size)
        )
        rows.append(f"{row:>{width}} {cells}")
    return "\n".join(rows)


def parse_human_action(text: str, env: GomokuEnv) -> int:
    """Parse ``row column`` and return a legal flattened action."""
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError("enter two integers: row column")
    try:
        row, column = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("row and column must be integers") from error
    if not (
        0 <= row < env.board_size and 0 <= column < env.board_size
    ):
        raise ValueError(
            f"row and column must be between 0 and {env.board_size - 1}"
        )
    action = row * env.board_size + column
    if action not in env.legal_actions():
        raise ValueError("that cell is already occupied")
    return action


def play_human_game(
    env: GomokuEnv,
    mcts: MCTS,
    human_player: int,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """Play one terminal game and return the winner code."""
    if human_player not in (env.BLACK, env.WHITE):
        raise ValueError("human_player must be BLACK or WHITE")
    env.reset()
    output_fn("You are X (black)." if human_player == env.BLACK
              else "You are O (white).")

    while not env.terminated:
        output_fn(render_board(env))
        if env.current_player == human_player:
            while True:
                text = input_fn("Your move (row column, or q): ").strip()
                if text.lower() in {"q", "quit", "exit"}:
                    output_fn("Game aborted.")
                    return env.EMPTY
                try:
                    action = parse_human_action(text, env)
                    break
                except ValueError as error:
                    output_fn(f"Invalid move: {error}")
        else:
            output_fn("MuZero is thinking...")
            root = mcts.run(
                env.observation(),
                env.legal_actions(),
                env.current_player,
                add_exploration_noise=False,
                env=env,
            )
            action = mcts.select_action(root, temperature=0)
            row, column = divmod(action, env.board_size)
            output_fn(f"MuZero plays: {row} {column}")
        env.step(action)

    output_fn(render_board(env))
    if env.winner == env.EMPTY:
        output_fn("Draw.")
    elif env.winner == human_player:
        output_fn("You win!")
    else:
        output_fn("MuZero wins.")
    return env.winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="trained checkpoint path (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        default="black",
        help="human side; black moves first (default: black)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations for each model move (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="MCTS random seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "tpu"),
        default="auto",
        help="compute backend (default: auto)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device.description}")
    loaded = load_checkpoint(args.checkpoint, device.torch_device)
    env = GomokuEnv(loaded.board_size, loaded.win_length)
    mcts = MCTS(
        loaded.network,
        MCTSConfig(num_simulations=args.simulations),
        seed=args.seed,
    )
    human_player = (
        env.BLACK if args.human_color == "black" else env.WHITE
    )
    play_human_game(env, mcts, human_player)


if __name__ == "__main__":
    main()
