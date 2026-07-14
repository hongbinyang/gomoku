import importlib
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from gomoku_muzero.delivery.packager import VENDORED_FILES, build_bundle
from gomoku_muzero.model.checkpoint import save_checkpoint
from gomoku_muzero.model.networks import MuZeroNetwork


@pytest.fixture()
def small_checkpoint(tmp_path):
    torch.manual_seed(0)
    network = MuZeroNetwork(board_size=5, hidden_channels=8, num_blocks=1)
    path = tmp_path / "small.pt"
    save_checkpoint(network, path, win_length=4)
    return path


def test_bundle_structure_and_isolation(small_checkpoint, tmp_path) -> None:
    bundle, zip_path = build_bundle(small_checkpoint, tmp_path / "dist")

    assert bundle.name == "gomoku-play-small"
    for relative in VENDORED_FILES:
        assert (bundle / "gomoku_play" / relative).exists(), relative
    assert (bundle / "gomoku_play" / "weights" / "small.pt").exists()
    assert (bundle / "gomoku_play" / "play.py").exists()
    assert (bundle / "pyproject.toml").exists()

    readme = (bundle / "README.md").read_text()
    assert "5x5 board" in readme
    assert "pip install" in readme
    assert "gomoku-play" in readme

    # The vendored code must be fully renamed and training-free.
    for source in (bundle / "gomoku_play").rglob("*.py"):
        text = source.read_text()
        assert "gomoku_muzero" not in text, source
    vendored = {p.name for p in (bundle / "gomoku_play").rglob("*.py")}
    for training_only in ("trainer.py", "pipeline.py", "train.py"):
        assert training_only not in vendored

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert f"{bundle.name}/pyproject.toml" in names


def test_bundle_plays_standalone(small_checkpoint, tmp_path) -> None:
    """The vendored code must work without the gomoku_muzero package."""
    bundle, _ = build_bundle(small_checkpoint, tmp_path / "dist")
    sys.path.insert(0, str(bundle))
    try:
        server_module = importlib.import_module("gomoku_play.web.server")
        session = server_module.GameSession(
            bundle / "gomoku_play" / "weights" / "small.pt",
            human_color="black",
            num_simulations=50,
        )
        state = session.play_human(2, 2)
        assert state["board"][2][2] == 1
        assert state["last_ai_move"] is not None
    finally:
        sys.path.remove(str(bundle))
        for name in [n for n in sys.modules if n.startswith("gomoku_play")]:
            del sys.modules[name]


def test_bundle_refuses_overwrite_and_bad_input(
    small_checkpoint, tmp_path
) -> None:
    build_bundle(small_checkpoint, tmp_path / "dist")
    with pytest.raises(FileExistsError):
        build_bundle(small_checkpoint, tmp_path / "dist")
    with pytest.raises(ValueError, match="name"):
        build_bundle(
            small_checkpoint, tmp_path / "dist2", name="../bad"
        )
    with pytest.raises(FileNotFoundError):
        build_bundle(tmp_path / "missing.pt", tmp_path / "dist3")


def test_training_states_cannot_be_packaged(tmp_path) -> None:
    bogus = tmp_path / "state.pt"
    torch.save({"format_version": 2, "games": []}, bogus)
    with pytest.raises(ValueError, match="training state"):
        build_bundle(bogus, tmp_path / "dist")
