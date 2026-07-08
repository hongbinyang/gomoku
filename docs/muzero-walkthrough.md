# MuZero implementation walkthrough

This guide maps each implementation checkpoint to the core MuZero algorithm.

## 1. Gomoku environment

`GomokuEnv` provides board representation, legal actions, win/draw detection,
and the self-play API. Run:

```bash
python -m pytest tests/test_env.py
```

## 2. Learned model

`MuZeroNetwork` contains the paper's three functions:

- Representation `h`: `[B, 3, N, N] -> [B, C, N, N]`
- Dynamics `g`: hidden state plus action `[B]` to next hidden state and reward
  `[B, 1]`
- Prediction `f`: hidden state to policy logits `[B, N*N]` and value `[B, 1]`

Following the paper's board-game architecture at reduced scale, `h` is a
residual tower (default 4 blocks, 64 channels, GroupNorm), `g` is a
half-depth residual tower over the hidden state concatenated with a one-hot
action plane, and `f` is a pair of thin convolutional heads. Hidden states
are min-max scaled to `[0, 1]` per sample after both `h` and `g` (Appendix
G), bounding the latent space during deep unrolls. Policy outputs are raw
logits; legal masking and softmax belong to MCTS. Value uses `tanh` because
Gomoku outcomes lie in `[-1, 1]`.

```bash
python -m pytest tests/test_networks.py
```

## 3. MCTS

At the real root, MCTS calls initial inference (`h` then `f`). Each simulation
uses PUCT, calls recurrent inference (`g` then `f`) for one newly visited
latent transition, expands legal children, and backs up the leaf value.
Following the paper, visited children's Q values are normalized to `[0, 1]`
with the range observed during the search (`MinMaxStats`), and the
exploration coefficient uses the logarithmic `pb_c_init`/`pb_c_base`
schedule. Leaves whose expansion produced no children cache their network
value, so revisits never repeat inference.

A node stores:

- policy prior assigned by its parent;
- hidden state `[1, C, N, N]`, computed lazily;
- incoming reward for its parent player;
- the cached network value from its expansion;
- mean value from its own player-to-move perspective;
- action-to-child mapping.

The policy training target is normalized root visit counts, not raw network
policy. Optional Dirichlet noise diversifies self-play roots.

Self-play reuses the played action's subtree as the next move's root: its
statistics are kept, its hidden state is re-grounded on the real
observation through `h`, and `num_simulations` becomes a target for the
root's total visit count, so only the missing simulations run.

When the caller provides the real environment, the search applies a
known-rules layer that never trusts the learned model where the rules can
answer directly. Each simulation's action path is replayed on a clone,
and then: moves that provably end the game are pinned to their exact
reward (win 1, draw 0) with value zero and no children; non-terminal
in-tree rewards are exactly zero; a player to move with an immediate
completion is a proven win (+1) and a player facing two opponent
completion cells is a proven loss (-1), both without expansion; a player
facing exactly one opponent completion cell expands only the forced
block; and a fixed fraction of prior mass (``threat_prior_fraction``) is
redistributed over direct-threat cells for either side, so tactical lines
are searched even when the learned policy underrates them — the search
remains free to override the bias. Together these make one-move outcomes
certain and open-three/open-four traps refutable within roughly 100
simulations at any training level. Like in-tree legality tracking, this
is a deliberate known-rules extension of pure MuZero, which would rely on
the dynamics network alone.

```bash
python -m pytest tests/test_mcts.py
```

## 4. Replay and K-step samples

A `GameHistory` with `T` moves stores:

- `T+1` observations, policies, values, and `to_play` entries;
- `T` actions and rewards;
- no hidden states.

For batch size `B` and unroll length `K`:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| observations | `[B, 3, N, N]` | Real starting states |
| actions | `[B, K]` | Dynamics actions |
| target_rewards | `[B, K]` | Transition rewards |
| target_policies | `[B, K+1, N*N]` | MCTS visit distributions |
| target_values | `[B, K+1]` | Player-to-move outcomes |
| dynamics_mask | `[B, K]` | Real transitions versus absorbing padding |
| prediction_mask | `[B, K+1]` | Real positions versus absorbing padding |

Samples that cross the end of a game are padded with absorbing steps: random
actions with zero value and reward targets, which the trainer supervises so
the model cannot hallucinate value beyond terminal states (the masks remain
available to distinguish real data in diagnostics). Each sample is also
transformed by a random dihedral board symmetry unless augmentation is
disabled.

The bounded buffer evicts its oldest game and samples uniformly by default;
exponential recency weighting is available so newly generated
policy-improvement data is sampled more often.

```bash
python -m pytest tests/test_replay.py
```

## 5. K-step training

Training recomputes hidden states using the current network:

```text
s0 = h(observation)
(policy0, value0) = f(s0)
(s1, reward1) = g(s0, action0)
(policy1, value1) = f(s1)
...
```

The objectives are policy cross-entropy, value MSE, and reward MSE. Two
stabilizers from the paper are applied: the hidden state's gradient is
halved at every application of `g`, and each unrolled step's loss is scaled
by `1/K` while the initial prediction keeps weight one. Backpropagation
flows through `h`, every application of `g`, and every application of `f`.

```bash
python -m pytest tests/test_trainer.py
```

## 6. Complete actor-learner loop

```text
learner publishes weights -> background MCTS self-play actor
                         actor -> game queue -> replay
                         replay -> K-step learner updates
                         learner -> periodic evaluation
```

Self-play signs each non-terminal value target for the player to move and uses
zero for the terminal value. The actor uses a frozen weight snapshot for an
entire game while the learner can update the live network independently.

```bash
python -m pytest \
  tests/test_self_play.py \
  tests/test_pipeline.py \
  tests/test_async_self_play.py
```
