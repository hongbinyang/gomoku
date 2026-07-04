# End-to-end execution

This guide covers one complete lifecycle: environment setup, validation,
training, monitoring, visualization, playing, and cleanup.

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
  --device auto \
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

checkpoints/smoke-test.pt
```

## 3. Start a longer 10x10 run

```bash
python -m gomoku_muzero.train \
  --board-size 10 \
  --win-length 5 \
  --iterations 200 \
  --simulations 100 \
  --games-per-iteration 10 \
  --training-steps 50 \
  --batch-size 64 \
  --unroll-steps 5 \
  --learning-rate 0.0003 \
  --replay-capacity 2000 \
  --replay-sampling recent \
  --replay-half-life 200 \
  --self-play-mode async \
  --self-play-queue-size 4 \
  --device auto \
  --evaluation-interval 10 \
  --evaluation-games 50 \
  --run-name baseline-10x10 \
  --checkpoint checkpoints/baseline-10x10.pt \
  --tensorboard \
  --seed 0
```

Run names cannot overwrite existing run directories. Choose a new name when
repeating an experiment, such as `baseline-10x10-seed1`.

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
- games and optimizer steps per second;
- replay sample age;
- actor queue size and policy lag;
- evaluation score;
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
  --device auto
```

Play as white by replacing `black` with `white`. Enter moves as zero-based
`row column` coordinates, for example `4 5`.

## 7. Inspect storage

```bash
du -sh runs/baseline-10x10
du -sh checkpoints/baseline-10x10.pt
```

Replay games exist only in memory and disappear when training exits. Persisted
storage consists of configuration, metrics, optional TensorBoard events,
plots, and the latest checkpoint.

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

Remove its trained checkpoint separately:

```bash
rm -f checkpoints/baseline-10x10.pt
```

For the quick validation artifacts:

```bash
rm -rf runs/smoke-test
rm -f checkpoints/smoke-test.pt
```

Do not remove a checkpoint you still want to play or compare. Both `runs/`
and `checkpoints/` are excluded from Git, so cleanup affects only local
experiment artifacts.
