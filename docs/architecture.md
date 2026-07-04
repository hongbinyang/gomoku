# Architecture

The package is organized by responsibility:

```text
gomoku_muzero/
  game/
    env.py              # board rules and environment API
  model/
    networks.py         # representation h, dynamics g, prediction f
    checkpoint.py       # model persistence and architecture metadata
  search/
    mcts.py             # nodes, PUCT, expansion, backup, root policy
  training/
    replay.py           # complete games and K-step samples
    trainer.py          # unrolled losses and optimizer updates
    pipeline.py         # outer self-play/train/evaluate loop
    async_pipeline.py   # actor-learner orchestration
  workflows/
    self_play.py        # MCTS-driven game generation
    self_play_actor.py  # background actor and weight snapshots
    evaluate.py         # matches against fixed baselines
  cli/
    train.py            # training command implementation
    play.py             # interactive game implementation
  runtime/
    device.py           # CPU, CUDA, MPS, and TPU/XLA selection
    metrics.py          # JSONL/CSV and optional TensorBoard logging
  train.py              # compatibility CLI entry point
  play.py               # compatibility CLI entry point
```

## Where new code belongs

| Concern | Package | Examples |
| --- | --- | --- |
| Gomoku rules and observations | `game` | legality, terminal detection, board encoding |
| Learned latent model | `model` | new heads, architecture changes, checkpoints |
| Planning | `search` | PUCT changes, node statistics, root selection |
| Learning mechanics | `training` | targets, replay sampling, losses, schedules |
| Multi-component use cases | `workflows` | self-play, evaluation, tournaments |
| User-facing commands | `cli` | arguments, terminal output, interactive input |
| Runtime services | `runtime` | devices, accelerator synchronization, metrics |

The intended dependency direction is:

```text
game + model -> search
game + search + training.replay -> workflows
model + training.replay -> training.trainer
workflows + training.trainer -> training.pipeline
all feature packages -> cli
```

Lower-level modules do not import CLI code. The `training` and `workflows`
package initializers remain lightweight to avoid circular imports. Stable
public classes are also importable from the root `gomoku_muzero` package.

## State and perspective conventions

- Board: `(N, N)` signed `int8`; empty `0`, black `+1`, white `-1`.
- Observation: `(3, N, N)` `float32`; black stones, white stones, and a
  black-to-play plane.
- Action: integer in `[0, N*N)`, mapped with `row * N + column`.
- Predicted value: scalar from the player-to-move's perspective.
- Transition reward: scalar from the player who selected the action.
- Terminal outcome: black's result (`+1`, `0`, or `-1`); value targets are
  signed for the player at each position.

These conventions make alternating-player backup explicit:

```text
parent_value = child_reward - discount * child_value
```

See [MuZero walkthrough](muzero-walkthrough.md) for the end-to-end data flow.
