# End-to-end execution

This guide covers one complete lifecycle: environment setup, validation,
training, monitoring, visualization, playing, and cleanup.

Every step below uses the command line; the entire lifecycle is also
available from the browser via the
[management console](manager.md) (`python -m gomoku_muzero.manager`),
which drives these same commands with live progress, charts, model
information, and guarded cleanup.

## 1. Set up the project

From the repository directory:

```bash
conda activate gomoku
python -m pip install -e ".[dev,metrics]"
```

Confirm the installation and available compute backend:

```bash
python -m pytest
python -c "from gomoku_muzero.runtime.device import resolve_device; print(resolve_device('auto').description)"
```

The examples below pass `--device mps` (Apple Silicon) explicitly, which
fails fast if MPS is unavailable; `--device auto` selects the same backend
when MPS is present. On this small network, batch-of-one MCTS inference can
run faster on `cpu` than on `mps` — compare the `games_per_second` metric
between short runs of each before committing to a long experiment.

## 2. Run a quick validation experiment

Use a unique run name:

```bash
python -m gomoku_muzero.train \
  --board-size 5 \
  --win-length 5 \
  --iterations 2 \
  --simulations 5 \
  --games-per-iteration 2 \
  --training-steps 2 \
  --batch-size 8 \
  --device mps \
  --self-play-mode async \
  --run-name smoke-test \
  --checkpoint checkpoints/smoke-test.pt
```

The command should finish quickly and create:

```text
runs/smoke-test/
  config.json
  metrics.jsonl
  metrics.csv

checkpoints/smoke-test.pt        # inference weights
checkpoints/training-state.pt    # resumable model/optimizer/replay state
```

Pass `--training-state` to give the training state a run-specific path.

## 3. Start a longer 10x10 run

```bash
python -m gomoku_muzero.train \
  --board-size 10 \
  --win-length 5 \
  --iterations 500 \
  --simulations 100 \
  --games-per-iteration 10 \
  --training-steps 50 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --value-loss-weight 2.0 \
  --temperature-moves 16 \
  --replay-capacity 2000 \
  --replay-sampling recent \
  --self-play-mode async \
  --device mps \
  --evaluation-interval 10 \
  --evaluation-games 25 \
  --run-name baseline-10x10 \
  --checkpoint checkpoints/baseline-10x10.pt \
  --training-state checkpoints/baseline-10x10-state.pt \
  --tensorboard \
  --seed 0
```

See [training.md](training.md) for what the exploration and value-weight
settings do and which metric confirms they are working, and its
"19x19 configuration" section for the full Go-sized board.

Run names cannot overwrite existing run directories. Choose a new name when
repeating an experiment, such as `baseline-10x10-seed1`.

If the run is interrupted, continue it from the last completed iteration:

```bash
python -m gomoku_muzero.train \
  --resume checkpoints/baseline-10x10-state.pt \
  --iterations 100 \
  --run-name baseline-10x10-resumed \
  --checkpoint checkpoints/baseline-10x10.pt \
  --training-state checkpoints/baseline-10x10-state.pt \
  --tensorboard
```

Remember `--tensorboard` on every resume — it is a per-invocation option,
not part of the saved state.

`--iterations` counts additional iterations; the saved board size and win
length take precedence over command-line values. See
[training.md](training.md) for details.

## 4. Monitor training

The training terminal reports the current self-play, optimization, or
evaluation phase. In a second terminal:

```bash
conda activate gomoku
tensorboard --logdir runs
```

Open the address printed by TensorBoard, normally:

```text
http://localhost:6006
```

Useful signals include:

- policy KL, value loss, and reward loss;
- value calibration error (`value_calibration_mae`) and gradient norm;
- games and optimizer steps per second;
- replay sample age;
- actor queue size and policy lag;
- evaluation scores: `evaluation_score` (vs. random, saturates early) and
  `heuristic_evaluation_score` (vs. win-or-block, the real strength
  signal);
- CUDA or MPS memory use when available.

Stop TensorBoard with `Ctrl-C`; this does not stop training.

## 5. Generate static plots

After at least one iteration:

```bash
python -m gomoku_muzero.plot \
  --run-dir runs/baseline-10x10
```

Generated files appear under:

```text
runs/baseline-10x10/plots/
  losses.png
  throughput.png
  evaluation.png
```

## 6. Play against the checkpoint

Training replaces the specified checkpoint after each completed iteration.
Play as black:

```bash
python -m gomoku_muzero.play \
  --checkpoint checkpoints/baseline-10x10.pt \
  --human-color black \
  --simulations 200 \
  --device mps
```

Play as white by replacing `black` with `white`. Enter moves as zero-based
`row column` coordinates, for example `4 5`.

## 7. Inspect storage

```bash
du -sh runs/baseline-10x10
du -sh checkpoints/baseline-10x10.pt
du -sh checkpoints/baseline-10x10-state.pt
```

Persisted storage consists of configuration, metrics, optional TensorBoard
events, plots, the latest inference checkpoint, and the training state. The
training state includes the replay buffer, so it is the largest artifact;
delete it once a run is finished and will not be resumed.

## 8. Clean up

First confirm the paths you intend to remove:

```bash
ls runs/baseline-10x10
ls checkpoints/baseline-10x10.pt
```

Remove only the selected run data:

```bash
rm -rf runs/baseline-10x10
```

Remove its trained checkpoint and training state separately:

```bash
rm -f checkpoints/baseline-10x10.pt
rm -f checkpoints/baseline-10x10-state.pt
```

For the quick validation artifacts:

```bash
rm -rf runs/smoke-test
rm -f checkpoints/smoke-test.pt checkpoints/training-state.pt
```

Do not remove a checkpoint you still want to play or compare. Both `runs/`
and `checkpoints/` are excluded from Git, so cleanup affects only local
experiment artifacts.
