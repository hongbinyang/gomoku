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

Training saves `checkpoints/latest.pt`. Play against it:

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

The implementation favors a clear core MuZero design and can be scaled by
increasing board size, self-play data, network capacity, training work, and
search simulations.
