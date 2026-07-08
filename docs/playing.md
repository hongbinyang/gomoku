# Playing against a trained model

Train and save a checkpoint:

```bash
python -m gomoku_muzero.train \
  --board-size 5 \
  --win-length 5 \
  --iterations 50 \
  --checkpoint checkpoints/my_model.pt
```

Then start an interactive terminal game:

```bash
python -m gomoku_muzero.play [OPTIONS]
```

## Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--checkpoint PATH` | `checkpoints/latest.pt` | Model checkpoint to load |
| `--human-color black` | `black` | Play first as black (`X`) |
| `--human-color white` | | Play second as white (`O`) |
| `--simulations N` | `100` | MCTS simulations for each model move |
| `--seed N` | `0` | MCTS random seed |
| `--device NAME` | `auto` | `auto`, `cpu`, `cuda`, `mps`, or `tpu` |
| `-h`, `--help` | | Print help and exit |

Higher simulation counts generally improve search but make each response
slower. Keep `--simulations` at 100 or above: the search's known-rules
threat refutations (blocking open threes and fours) need roughly that
budget to complete; below it the model may revert to threat-blind play.
Play uses greedy root selection without Dirichlet noise, so identical
weights and positions are deterministic.

## Examples

Play as black:

```bash
python -m gomoku_muzero.play \
  --checkpoint checkpoints/my_model.pt \
  --human-color black \
  --simulations 100
```

Play as white:

```bash
python -m gomoku_muzero.play \
  --checkpoint checkpoints/my_model.pt \
  --human-color white
```

Enter zero-based `row column` coordinates:

```text
Your move (row column, or q): 2 3
```

Enter `q`, `quit`, or `exit` to stop. Board size, win length, network width,
and weights are read from the checkpoint. Playing does not modify the model.

With default paths, the shortest workflow is:

```bash
python -m gomoku_muzero.train
python -m gomoku_muzero.play
```
