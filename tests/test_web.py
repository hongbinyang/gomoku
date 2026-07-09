import json
import threading
import urllib.request

import pytest
import torch

from gomoku_muzero.model.checkpoint import save_checkpoint
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.web.server import (
    GameSession,
    GomokuWebServer,
    list_checkpoints,
)


@pytest.fixture()
def checkpoint_dir(tmp_path):
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=5, hidden_channels=8, num_blocks=1)
    save_checkpoint(network, tmp_path / "small.pt", win_length=4)
    save_checkpoint(network, tmp_path / "small-state.pt", win_length=4)
    return tmp_path


def make_session(checkpoint_dir, **kwargs) -> GameSession:
    defaults = {"human_color": "black", "num_simulations": 50}
    defaults.update(kwargs)
    return GameSession(checkpoint_dir / "small.pt", **defaults)


def test_list_checkpoints_excludes_training_state(checkpoint_dir) -> None:
    assert list_checkpoints(checkpoint_dir) == ["small.pt"]
    assert list_checkpoints(checkpoint_dir / "missing") == []


def test_human_black_moves_first(checkpoint_dir) -> None:
    session = make_session(checkpoint_dir)
    state = session.state()

    assert state["board_size"] == 5
    assert state["to_play"] == state["human_player"] == 1
    assert state["last_ai_move"] is None

    state = session.play_human(2, 2)
    assert state["board"][2][2] == 1
    assert state["last_ai_move"] is not None
    assert state["to_play"] == 1 or state["terminated"]


def test_human_white_gets_ai_opening_move(checkpoint_dir) -> None:
    session = make_session(checkpoint_dir, human_color="white")
    state = session.state()

    assert state["human_player"] == -1
    assert state["last_ai_move"] is not None
    assert sum(cell != 0 for row in state["board"] for cell in row) == 1
    assert state["to_play"] == -1


def test_move_history_supports_full_audit(checkpoint_dir) -> None:
    session = make_session(checkpoint_dir, human_color="white")
    state = session.state()
    assert len(state["moves"]) == 1  # the AI's opening move
    assert state["moves"][0]["by"] == "ai"
    assert state["moves"][0]["player"] == 1

    row, column = next(
        (r, c)
        for r in range(5)
        for c in range(5)
        if state["board"][r][c] == 0
    )
    state = session.play_human(row, column)
    moves = state["moves"]
    assert [m["number"] for m in moves] == list(range(1, len(moves) + 1))
    assert moves[1] == {
        "number": 2, "player": -1, "row": row, "column": column,
        "by": "human",
    }
    # Replaying the history reproduces the board exactly.
    import numpy as np
    board = np.zeros((5, 5), dtype=int)
    for move in moves:
        board[move["row"], move["column"]] = move["player"]
    assert board.tolist() == state["board"]


def test_rejects_illegal_moves(checkpoint_dir) -> None:
    session = make_session(checkpoint_dir)
    session.play_human(2, 2)

    with pytest.raises(ValueError, match="occupied"):
        session.play_human(2, 2)
    with pytest.raises(ValueError, match="outside"):
        session.play_human(9, 9)


def test_session_validates_settings(checkpoint_dir) -> None:
    with pytest.raises(ValueError, match="human_color"):
        make_session(checkpoint_dir, human_color="green")
    with pytest.raises(ValueError, match="num_simulations"):
        make_session(checkpoint_dir, num_simulations=1)


def test_http_round_trip(checkpoint_dir) -> None:
    server = GomokuWebServer(("127.0.0.1", 0), checkpoint_dir)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, json.loads(response.read())

    def post(path, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        with urllib.request.urlopen(base + "/", timeout=30) as response:
            page = response.read().decode()
        assert "Gomoku" in page

        status, listing = get("/api/checkpoints")
        assert status == 200 and listing["checkpoints"] == ["small.pt"]

        status, _ = post("/api/move", {"row": 0, "column": 0})
        assert status == 400  # no game yet

        status, state = post(
            "/api/new",
            {
                "checkpoint": "small.pt",
                "human_color": "black",
                "num_simulations": 50,
            },
        )
        assert status == 200 and state["to_play"] == 1

        status, state = post("/api/move", {"row": 2, "column": 2})
        assert status == 200 and state["board"][2][2] == 1

        status, error = post(
            "/api/new", {"checkpoint": "../evil.pt"}
        )
        assert status == 400 and "invalid" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
