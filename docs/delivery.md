# Delivering a trained model

The packaging command wraps a trained model and the browser game into a
small, standalone bundle that installs on Windows, Linux, and macOS with
one `pip` command. The bundle contains only the play path — environment,
networks, search, and the web game — with no training code, and building
it changes nothing in this project.

## Create a bundle

```bash
python -m gomoku_muzero.package \
  --checkpoint checkpoints/go19-v1.pt \
  --output dist
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--checkpoint PATH` | `checkpoints/latest.pt` | Trained model to package |
| `--output PATH` | `dist` | Directory receiving bundle and zip |
| `--name NAME` | checkpoint stem | Bundle name suffix |

This produces `dist/gomoku-play-<name>/` and `dist/gomoku-play-<name>.zip`
(roughly the model size plus ~100 KB of code — about 7 MB for a 19x19
model). Only format 2 model checkpoints can be packaged; resumable
training states are rejected.

Inside the bundle: a `gomoku_play` Python package (the play-path modules
copied with the package name rewritten, so it never conflicts with a
`gomoku_muzero` installation), the model as package data, a
`pyproject.toml` with a `gomoku-play` console command, and a README with
the recipient instructions below.

## Install on a target machine

1. Install Python 3.10+ (on Windows, tick "Add python.exe to PATH").
2. Unzip the bundle and run:

```bash
pip install ./gomoku-play-go19-v1
```

pip downloads NumPy and PyTorch automatically on first install. On
Linux, PyTorch defaults to a large CUDA build; the CPU-only build is
much smaller:

```bash
pip install ./gomoku-play-go19-v1 \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Play on the target machine

```bash
gomoku-play
```

The browser opens the Go-style board automatically (or visit
`http://127.0.0.1:8000`): pick a color, press "New game", click
intersections to play, review the move history, and copy game records —
the same interface as [playing.md](playing.md). Options: `--port`,
`--no-browser`, `--device auto|cpu|cuda|mps`.
