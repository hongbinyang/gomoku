# Playing against a trained model

Train and save a checkpoint:

```bash
python -m gomoku_muzero.train \
  --board-size 5 \
  --win-length 5 \
  --iterations 50 \
  --checkpoint checkpoints/my_model.pt
```

## Playing in the browser

Start the local web server and open the printed address:

```bash
python -m gomoku_muzero.serve
```

The same page is available as the Play tab of the
[management console](manager.md) (`python -m gomoku_muzero.manager`).

The page renders a Go-style board (stones on intersections, wooden
background, last-move markers). Pick a model from the dropdown — every
`*.pt` file in the checkpoint directory except training states — choose
your color, set the MCTS simulation count (at least 100 recommended; see
the note below), and press "New game". Click an intersection to place a
stone; MuZero replies automatically, and when you play white it opens the
game.

The panel keeps a numbered move history for auditing: click any entry to
replay the board as it stood after that move (the game pauses while
reviewing; press "Live" or the latest entry to continue), and toggle
"Show move numbers on stones" for a Go-style numbered review of the whole
game.

"Copy game record" puts a plain-text transcript on the clipboard for
sharing — a self-describing header (board, players, model, simulations,
result) followed by one numbered line per move:

```text
Gomoku 10x10, 5 in a row
Black (X): Human
White (O): MuZero (baseline-10x10.pt)
MCTS simulations: 200
Result: Black wins on move 23

Moves as (row, column), zero-based from the top-left:
  1. X (4, 4)  Human
  2. O (5, 5)  MuZero
  ...
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--host ADDR` | `127.0.0.1` | Bind address; local-only by default |
| `--port N` | `8000` | Port to listen on |
| `--checkpoint-dir PATH` | `checkpoints` | Directory scanned for models |
| `--device NAME` | `auto` | `auto`, `cpu`, `cuda`, `mps`, or `tpu` |

The server is a dependency-free single-game tool for local use: it holds
one active game at a time, and starting a new game replaces the previous
one.

## Playing in the terminal

Start an interactive terminal game:

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
