# Metrics and visualization

Every training invocation creates a run directory:

```text
runs/<run-name>/
  config.json
  metrics.jsonl
  metrics.csv
  tensorboard/          # when --tensorboard is enabled
  plots/                # created by the plot command
```

`metrics.jsonl` is the append-only source of truth. `metrics.csv` contains the
same events in a spreadsheet-friendly form. Run directories are ignored by
Git by default.

## Training options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--run-name NAME` | UTC timestamp | Stable directory name for the run |
| `--runs-dir PATH` | `runs` | Parent directory for run data |
| `--tensorboard` | disabled | Also emit TensorBoard event files |

Run names may contain letters, numbers, periods, underscores, and hyphens.
Existing run directories are never overwritten.

Example:

```bash
python -m gomoku_muzero.train \
  --run-name baseline-10x10 \
  --runs-dir runs
```

The configuration records every CLI argument, resolved device, Python
version, and PyTorch version.

## Collected metrics

Training objectives:

- total, policy, value, and reward losses;
- target policy entropy;
- policy KL divergence.

Throughput and timing:

- total iteration time;
- self-play, training, and evaluation time;
- games per second;
- optimizer steps per second;
- moves generated.

Actor and replay state:

- replay-buffer game count;
- mean and maximum sampled-game age;
- actor queue size and total generated games;
- actor and published network versions;
- mean and maximum policy lag.

Evaluation and hardware:

- evaluation wins, draws, losses, and score;
- CUDA or MPS allocated/reserved memory when exposed by PyTorch.

## TensorBoard

Install optional visualization dependencies:

```bash
python -m pip install -e ".[metrics]"
```

Train with TensorBoard output:

```bash
python -m gomoku_muzero.train \
  --run-name baseline-10x10 \
  --tensorboard
```

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

## Static plots

Create PNG charts after or during a run:

```bash
python -m gomoku_muzero.plot \
  --run-dir runs/baseline-10x10
```

The command creates available charts under `plots/`:

- `losses.png`
- `throughput.png`
- `evaluation.png`

Matplotlib is imported only by the plotting command; core training and metrics
persistence have no plotting dependency.
