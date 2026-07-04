from dataclasses import replace

import numpy as np
import pytest
import torch

from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.training.replay import GameHistory, ReplayBatch, ReplayBuffer
from gomoku_muzero.training.trainer import MuZeroTrainer


def make_batch(num_unroll_steps: int = 2) -> ReplayBatch:
    action_space_size = 4
    game = GameHistory(
        observations=[
            np.zeros((3, 2, 2), dtype=np.float32),
            np.ones((3, 2, 2), dtype=np.float32),
            np.full((3, 2, 2), 2, dtype=np.float32),
        ],
        actions=[0, 1],
        rewards=[0.0, 1.0],
        policies=[
            np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
            np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ],
        values=[1.0, -1.0, 0.0],
        to_play=[1, -1, 1],
    )
    buffer = ReplayBuffer(1, action_space_size, seed=0)
    buffer.save_game(game)
    return buffer.sample_batch(2, num_unroll_steps)


def test_compute_loss_returns_finite_scalars() -> None:
    trainer = MuZeroTrainer(MuZeroNetwork(board_size=2, hidden_channels=8))

    losses = trainer.compute_loss(make_batch())

    for loss in (
        losses.total,
        losses.policy,
        losses.value,
        losses.reward,
        losses.policy_target_entropy,
        losses.policy_kl,
    ):
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_unrolled_loss_sends_gradients_to_h_g_and_f() -> None:
    network = MuZeroNetwork(board_size=2, hidden_channels=8)
    trainer = MuZeroTrainer(network)

    trainer.compute_loss(make_batch()).total.backward()

    for module in (
        network.representation,
        network.dynamics,
        network.prediction,
    ):
        assert any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters()
        )


def test_train_step_updates_parameters_and_reports_metrics() -> None:
    network = MuZeroNetwork(board_size=2, hidden_channels=8)
    trainer = MuZeroTrainer(network, learning_rate=1e-2)
    before = [parameter.detach().clone() for parameter in network.parameters()]

    metrics = trainer.train_step(make_batch())

    assert set(metrics) == {
        "loss",
        "policy_loss",
        "value_loss",
        "reward_loss",
        "policy_target_entropy",
        "policy_kl",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, network.parameters())
    )


def test_policy_loss_decomposes_into_entropy_and_kl() -> None:
    trainer = MuZeroTrainer(MuZeroNetwork(board_size=2, hidden_channels=8))

    losses = trainer.compute_loss(make_batch())

    assert losses.policy.item() == pytest.approx(
        losses.policy_target_entropy.item() + losses.policy_kl.item(),
        abs=1e-6,
    )


def test_padded_targets_do_not_change_loss() -> None:
    torch.manual_seed(0)
    trainer = MuZeroTrainer(MuZeroNetwork(board_size=2, hidden_channels=8))
    batch = make_batch(num_unroll_steps=3)

    changed_rewards = batch.target_rewards.copy()
    changed_values = batch.target_values.copy()
    changed_policies = batch.target_policies.copy()
    changed_rewards[batch.dynamics_mask == 0] = 99
    changed_values[batch.prediction_mask == 0] = 99
    changed_policies[batch.prediction_mask == 0] = 99
    changed = replace(
        batch,
        target_rewards=changed_rewards,
        target_values=changed_values,
        target_policies=changed_policies,
    )

    original_loss = trainer.compute_loss(batch).total
    changed_loss = trainer.compute_loss(changed).total

    assert changed_loss.item() == pytest.approx(original_loss.item())


def test_zero_step_unroll_has_no_reward_loss() -> None:
    trainer = MuZeroTrainer(MuZeroNetwork(board_size=2, hidden_channels=8))

    losses = trainer.compute_loss(make_batch(num_unroll_steps=0))

    assert losses.reward.item() == 0.0
