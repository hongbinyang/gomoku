"""A dependency-free web server for playing against trained checkpoints.

The server exposes a small JSON API over Python's standard-library
``http.server`` and serves a single static page that renders a Go-style
board. It is a local, single-game tool: one active game session at a
time, guarded by a lock so concurrent requests cannot corrupt search or
environment state.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.model.checkpoint import load_checkpoint
from gomoku_muzero.search.mcts import MCTS, MCTSConfig

MIN_SIMULATIONS = 50
MAX_SIMULATIONS = 1600
_STATIC_DIR = Path(__file__).parent / "static"


def list_checkpoints(directory: str | Path) -> list[str]:
    """Return playable checkpoint file names in ``directory``.

    Training-state files are excluded by naming convention: they contain
    the replay buffer and are not intended for inference.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.glob("*.pt")
        if "state" not in path.stem
    )


class GameSession:
    """One human-versus-checkpoint game with MuZero search replies."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        human_color: str = "black",
        num_simulations: int = 200,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if human_color not in ("black", "white"):
            raise ValueError("human_color must be 'black' or 'white'")
        if not MIN_SIMULATIONS <= num_simulations <= MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be between {MIN_SIMULATIONS} "
                f"and {MAX_SIMULATIONS}"
            )
        loaded = load_checkpoint(checkpoint_path, device)
        self.checkpoint_name = Path(checkpoint_path).name
        self.env = GomokuEnv(loaded.board_size, loaded.win_length)
        self.mcts = MCTS(
            loaded.network,
            MCTSConfig(num_simulations=num_simulations),
            seed=seed,
        )
        self.human_player = (
            self.env.BLACK if human_color == "black" else self.env.WHITE
        )
        self.last_human_move: int | None = None
        self.last_ai_move: int | None = None
        self.moves: list[dict[str, Any]] = []
        if self.env.current_player != self.human_player:
            self._play_ai_move()

    def play_human(self, row: int, column: int) -> dict[str, Any]:
        """Apply the human move, reply with the AI move, return the state."""
        if self.env.terminated:
            raise ValueError("the game is already over")
        if self.env.current_player != self.human_player:
            raise ValueError("it is not the human player's turn")
        if not (
            0 <= row < self.env.board_size
            and 0 <= column < self.env.board_size
        ):
            raise ValueError("move is outside the board")
        action = row * self.env.board_size + column
        if action not in self.env.legal_actions():
            raise ValueError("that cell is already occupied")

        self._record_move(action, "human")
        self.env.step(action)
        self.last_human_move = action
        if not self.env.terminated:
            self._play_ai_move()
        return self.state()

    def state(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot of the game."""
        return {
            "board": self.env.board.tolist(),
            "board_size": self.env.board_size,
            "win_length": self.env.win_length,
            "to_play": int(self.env.current_player),
            "human_player": int(self.human_player),
            "terminated": self.env.terminated,
            "winner": int(self.env.winner),
            "last_human_move": self.last_human_move,
            "last_ai_move": self.last_ai_move,
            "moves": list(self.moves),
            "checkpoint": self.checkpoint_name,
            "num_simulations": self.mcts.config.num_simulations,
        }

    def _play_ai_move(self) -> None:
        root = self.mcts.run(
            self.env.observation(),
            self.env.legal_actions(),
            self.env.current_player,
            add_exploration_noise=False,
            env=self.env,
        )
        action = self.mcts.select_action(root, temperature=0)
        self._record_move(action, "ai")
        self.env.step(action)
        self.last_ai_move = action

    def _record_move(self, action: int, by: str) -> None:
        row, column = divmod(action, self.env.board_size)
        self.moves.append(
            {
                "number": len(self.moves) + 1,
                "player": int(self.env.current_player),
                "row": row,
                "column": column,
                "by": by,
            }
        )


class _Handler(BaseHTTPRequestHandler):
    """Routes: the static page, checkpoint listing, new game, and moves."""

    server: "GomokuWebServer"

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path in ("/", "/index.html"):
            self._send_page()
        elif self.path == "/api/checkpoints":
            self._send_json(
                200,
                {
                    "checkpoints": list_checkpoints(
                        self.server.checkpoint_dir
                    )
                },
            )
        elif self.path == "/api/state":
            with self.server.lock:
                if self.server.session is None:
                    self._send_json(404, {"error": "no active game"})
                else:
                    self._send_json(200, self.server.session.state())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/new":
                self._handle_new(payload)
            elif self.path == "/api/move":
                self._handle_move(payload)
            else:
                self._send_json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as error:
            self._send_json(400, {"error": str(error) or "bad request"})
        except FileNotFoundError as error:
            self._send_json(400, {"error": str(error)})

    def _handle_new(self, payload: dict[str, Any]) -> None:
        name = str(payload["checkpoint"])
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("invalid checkpoint name")
        with self.server.lock:
            self.server.session = GameSession(
                Path(self.server.checkpoint_dir) / name,
                human_color=str(payload.get("human_color", "black")),
                num_simulations=int(payload.get("num_simulations", 200)),
                device=self.server.device,
            )
            self._send_json(200, self.server.session.state())

    def _handle_move(self, payload: dict[str, Any]) -> None:
        with self.server.lock:
            if self.server.session is None:
                raise ValueError("start a game first")
            state = self.server.session.play_human(
                int(payload["row"]), int(payload["column"])
            )
        self._send_json(200, state)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 1_000_000:
            raise ValueError("missing or oversized request body")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_page(self) -> None:
        body = (_STATIC_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal quiet; errors surface as JSON responses.
        pass


class GomokuWebServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the game session and its lock."""

    def __init__(
        self,
        address: tuple[str, int],
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(address, _Handler)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.session: GameSession | None = None
        self.lock = Lock()


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device = "cpu",
) -> None:
    """Run the web server until interrupted."""
    server = GomokuWebServer((host, port), checkpoint_dir, device)
    print(f"Serving Gomoku at http://{host}:{server.server_address[1]}")
    print(f"Checkpoints from: {Path(checkpoint_dir).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
