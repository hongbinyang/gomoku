# Gomoku MuZero

## Overview

This project implements a Gomoku-playing agent based on MuZero using Python
and PyTorch. The agent learns a latent dynamics model from self-play and uses
Monte Carlo tree search to plan moves without requiring the learned model to
reconstruct future board observations.

The implementation includes the Gomoku environment, MuZero representation,
dynamics and prediction networks, MCTS, replay, K-step training, self-play,
evaluation, checkpoint persistence, and interactive play against a trained
model.

The default game is 10x10 with five stones required to win.

## Quick start

```bash
cd gomoku
conda activate gomoku
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

Train a small model:

```bash
python -m gomoku_muzero.train \
  --iterations 10 \
  --simulations 25
```

Training saves `checkpoints/latest.pt` (inference weights) and
`checkpoints/training-state.pt` (model, optimizer, and replay buffer).
Resume an interrupted run with
`python -m gomoku_muzero.train --resume checkpoints/training-state.pt`.

Or manage everything — training lifecycle with live progress, and play —
from the browser console:

```bash
python -m gomoku_muzero.manager
```

Play against a trained checkpoint in the browser (Go-style board, model
and color selectable in the page):

```bash
python -m gomoku_muzero.serve
```

Or in the terminal:

```bash
python -m gomoku_muzero.play \
  --checkpoint checkpoints/latest.pt \
  --human-color black
```

Enter moves as zero-based coordinates such as `2 3`. Enter `q` to stop.

## Documentation

- [Architecture](docs/architecture.md): package responsibilities, dependency
  direction, and state conventions.
- [Training](docs/training.md): every training option, examples, progress,
  metrics, and convergence guidance.
- [Management console](docs/manager.md): browser-based training
  lifecycle, progress, and play.
- [Playing](docs/playing.md): checkpoints, play options, and move input.
- [MuZero walkthrough](docs/muzero-walkthrough.md): environment through the
  complete self-play and training loop.
- [Development](docs/development.md): setup, tests, and change checklist.
- [Compute devices](docs/devices.md): CPU, CUDA, Apple MPS, and TPU/XLA.
- [Metrics](docs/metrics.md): JSONL/CSV logs, TensorBoard, and static plots.
- [End-to-end execution](docs/end-to-end.md): setup through training, playing,
  storage inspection, and cleanup.

## Core flow

```text
current network -> MCTS self-play -> replay buffer -> K-step training
                -> periodic evaluation -> saved checkpoint -> human play
```

The networks follow the paper's board-game architecture at reduced scale:
residual towers for representation and dynamics (default 4 blocks, 64
channels) with per-sample min-max hidden-state scaling, and thin prediction
heads. Search follows the paper's PUCT formulation (min-max Q normalization
and the logarithmic exploration schedule), reuses the played subtree
between self-play moves, and is terminal-aware: provably game-ending moves
are pinned to their exact reward using the real rules, so one-move wins
and losses are never missed. Training applies the paper's unroll
stabilizers (hidden-state gradient halving, 1/K step weighting),
supervises absorbing states beyond terminal positions, and augments replay
samples with the board's eight dihedral symmetries by default. Periodic
evaluation plays both a random baseline and a win-or-block heuristic; the
latter is the meaningful strength signal.

The implementation favors a clear core MuZero design and can be scaled by
increasing board size, self-play data, network capacity, training work, and
search simulations.
