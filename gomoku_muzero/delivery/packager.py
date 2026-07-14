"""Build a standalone, pip-installable bundle around a trained model.

The bundle is a self-contained Python distribution named ``gomoku_play``:
the play-path modules of this project (environment, networks, checkpoint
loading, search, device selection, and the web game) are copied in with
the package name rewritten, the trained model ships as package data, and
a generated ``pyproject.toml``/README make installation on any platform
two commands:

    pip install ./gomoku-play-<name>
    gomoku-play

No training code is included, and nothing in the existing project is
modified — packaging reads the installed modules and writes a new
directory (plus a zip for transfer).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch

import gomoku_muzero

# Play path only: everything the web game needs, nothing training needs.
VENDORED_FILES = (
    "game/__init__.py",
    "game/env.py",
    "model/__init__.py",
    "model/networks.py",
    "model/checkpoint.py",
    "training/__init__.py",
    "training/replay.py",  # checkpoint.py imports GameHistory from here
    "search/__init__.py",
    "search/mcts.py",
    "runtime/__init__.py",
    "runtime/device.py",
    "web/__init__.py",
    "web/server.py",
    "web/static/index.html",
)


def build_bundle(
    checkpoint_path: str | Path,
    output_dir: str | Path = "dist",
    name: str | None = None,
) -> tuple[Path, Path]:
    """Create the bundle directory and its zip; returns both paths."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    metadata = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    if metadata.get("format_version") != 2:
        raise ValueError(
            "unsupported checkpoint format; only format 2 model "
            "checkpoints can be packaged"
        )
    if "games" in metadata or "optimizer_state_dict" in metadata:
        raise ValueError(
            "this is a resumable training state, not a model "
            "checkpoint; package the --checkpoint file instead"
        )

    slug = name or checkpoint.stem
    if not re.fullmatch(r"[A-Za-z0-9._-]+", slug):
        raise ValueError(
            "name may contain only letters, numbers, '.', '_', and '-'"
        )
    bundle = Path(output_dir) / f"gomoku-play-{slug}"
    if bundle.exists():
        raise FileExistsError(
            f"{bundle} already exists; remove it or choose --name"
        )

    package_dir = bundle / "gomoku_play"
    source_root = Path(gomoku_muzero.__file__).parent
    for relative in VENDORED_FILES:
        source = source_root / relative
        target = package_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".py":
            text = source.read_text(encoding="utf-8")
            target.write_text(
                text.replace("gomoku_muzero", "gomoku_play"),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source, target)

    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True)
    shutil.copyfile(checkpoint, weights_dir / checkpoint.name)

    board = int(metadata["board_size"])
    win_length = int(metadata["win_length"])
    (package_dir / "__init__.py").write_text(
        '"""Standalone Gomoku player packaged from a trained MuZero '
        'model."""\n\n__version__ = "1.0.0"\n',
        encoding="utf-8",
    )
    (package_dir / "play.py").write_text(
        _PLAY_TEMPLATE, encoding="utf-8"
    )
    (bundle / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(slug=slug.lower()), encoding="utf-8"
    )
    (bundle / "README.md").write_text(
        _README_TEMPLATE.format(
            slug=slug,
            board=board,
            win_length=win_length,
            channels=int(metadata["hidden_channels"]),
            blocks=int(metadata["num_blocks"]),
            model_file=checkpoint.name,
            folder=bundle.name,
        ),
        encoding="utf-8",
    )

    zip_base = shutil.make_archive(
        str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name
    )
    return bundle, Path(zip_base)


_PLAY_TEMPLATE = '''"""Play against the bundled model in your browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from gomoku_play.runtime.device import resolve_device
from gomoku_play.web.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the browser automatically",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device.description}")
    if not args.no_browser:
        threading.Timer(
            1.0,
            webbrowser.open,
            [f"http://{args.host}:{args.port}"],
        ).start()
    serve(
        host=args.host,
        port=args.port,
        checkpoint_dir=Path(__file__).parent / "weights",
        device=device.torch_device,
    )


if __name__ == "__main__":
    main()
'''

_PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gomoku-play-{slug}"
version = "1.0.0"
description = "Standalone Gomoku player packaged from a trained MuZero model"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
]

[project.scripts]
gomoku-play = "gomoku_play.play:main"

[tool.setuptools.packages.find]
include = ["gomoku_play*"]

[tool.setuptools.package-data]
gomoku_play = ["web/static/*.html", "weights/*.pt"]
"""

_README_TEMPLATE = """# Gomoku player: {slug}

Play against a trained MuZero model ({board}x{board} board,
{win_length} in a row, {channels}-channel x {blocks}-block network,
model file `{model_file}`) in your web browser. Works on Windows,
Linux, and macOS.

## Install

1. Install Python 3.10 or newer from https://www.python.org/downloads/
   (on Windows, tick "Add python.exe to PATH" during setup).
2. From the folder containing `{folder}`, run:

   ```
   pip install ./{folder}
   ```

   pip downloads the two dependencies (NumPy and PyTorch) the first
   time. On Linux, PyTorch defaults to a large CUDA build; for a much
   smaller CPU-only install use:

   ```
   pip install ./{folder} --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Play

```
gomoku-play
```

Your browser opens the board automatically (or visit
http://127.0.0.1:8000). Pick your color, press "New game", and click an
intersection to place a stone. The move history panel supports
reviewing any past position and copying the full game record.

Options: `--port N` to use another port, `--no-browser` to skip
opening the browser, `--device cpu|cuda|mps` to pick the compute
backend (`auto` by default).
"""
