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

Policy outputs are raw logits; legal masking and softmax belong to MCTS. Value
uses `tanh` because Gomoku outcomes lie in `[-1, 1]`.

```bash
python -m pytest tests/test_networks.py
```

## 3. MCTS

At the real root, MCTS calls initial inference (`h` then `f`). Each simulation
uses PUCT, calls recurrent inference (`g` then `f`) for one newly visited
latent transition, expands legal children, and backs up the leaf value.

A node stores:

- policy prior assigned by its parent;
- hidden state `[1, C, N, N]`, computed lazily;
- incoming reward for its parent player;
- mean value from its own player-to-move perspective;
- action-to-child mapping.

The policy training target is normalized root visit counts, not raw network
policy. Optional Dirichlet noise diversifies self-play roots.

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
| dynamics_mask | `[B, K]` | Real transitions versus padding |
| prediction_mask | `[B, K+1]` | Real positions versus padding |

Terminal-adjacent samples are padded and masked.

The bounded buffer evicts its oldest game. By default it applies exponential
recency weighting so newly generated policy-improvement data is sampled more
often without completely discarding older experience. Uniform sampling remains
available for controlled comparisons.

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

The objectives are policy cross-entropy, value MSE, and reward MSE.
Backpropagation flows through `h`, every application of `g`, and every
application of `f`.

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
