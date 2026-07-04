"""K-step MuZero training through representation, dynamics, and prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.runtime.device import optimizer_step
from gomoku_muzero.training.replay import ReplayBatch


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of MuZero's three supervised objectives."""

    policy: float = 1.0
    value: float = 1.0
    reward: float = 1.0


@dataclass(frozen=True)
class MuZeroLosses:
    """Scalar differentiable losses from one replay batch."""

    total: Tensor
    policy: Tensor
    value: Tensor
    reward: Tensor
    policy_target_entropy: Tensor
    policy_kl: Tensor


class MuZeroTrainer:
    """Optimize a MuZero network from fixed-shape replay samples."""

    def __init__(
        self,
        network: MuZeroNetwork,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: LossWeights | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.network = network
        self.loss_weights = loss_weights or LossWeights()
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def compute_loss(self, batch: ReplayBatch) -> MuZeroLosses:
        """Unroll K model steps and compute masked MuZero losses.

        Prediction index 0 comes from ``h(observation)`` followed by ``f``.
        Prediction indices 1 through K come from repeated ``g`` then ``f``.
        Reward index k supervises the transition from position k to k+1.
        """
        device = next(self.network.parameters()).device
        observations = torch.as_tensor(
            batch.observations, dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            batch.actions, dtype=torch.long, device=device
        )
        target_rewards = torch.as_tensor(
            batch.target_rewards, dtype=torch.float32, device=device
        )
        target_policies = torch.as_tensor(
            batch.target_policies, dtype=torch.float32, device=device
        )
        target_values = torch.as_tensor(
            batch.target_values, dtype=torch.float32, device=device
        )
        dynamics_mask = torch.as_tensor(
            batch.dynamics_mask, dtype=torch.float32, device=device
        )
        prediction_mask = torch.as_tensor(
            batch.prediction_mask, dtype=torch.float32, device=device
        )
        self._validate_shapes(
            observations,
            actions,
            target_rewards,
            target_policies,
            target_values,
            dynamics_mask,
            prediction_mask,
        )

        initial = self.network.initial_inference(observations)
        hidden_state = initial.hidden_state
        policy_logits = [initial.policy_logits]
        predicted_values = [initial.value.squeeze(-1)]
        predicted_rewards: list[Tensor] = []

        for step in range(actions.shape[1]):
            recurrent = self.network.recurrent_inference(
                hidden_state, actions[:, step]
            )
            hidden_state = recurrent.hidden_state
            predicted_rewards.append(recurrent.reward.squeeze(-1))
            policy_logits.append(recurrent.policy_logits)
            predicted_values.append(recurrent.value.squeeze(-1))

        policy_logits_tensor = torch.stack(policy_logits, dim=1)
        predicted_values_tensor = torch.stack(predicted_values, dim=1)

        per_position_policy_loss = -(
            target_policies
            * F.log_softmax(policy_logits_tensor, dim=-1)
        ).sum(dim=-1)
        # Terminal positions have a zero policy because no action is selected.
        policy_mask = prediction_mask * (
            target_policies.sum(dim=-1) > 0
        ).to(torch.float32)
        policy_loss = self._masked_mean(
            per_position_policy_loss, policy_mask
        )
        per_position_target_entropy = -(
            target_policies
            * target_policies.clamp_min(1e-8).log()
        ).sum(dim=-1)
        policy_target_entropy = self._masked_mean(
            per_position_target_entropy, policy_mask
        )
        # CE(target, prediction) = H(target) + KL(target || prediction).
        policy_kl = (policy_loss - policy_target_entropy).clamp_min(0.0)

        per_position_value_loss = F.mse_loss(
            predicted_values_tensor, target_values, reduction="none"
        )
        value_loss = self._masked_mean(
            per_position_value_loss, prediction_mask
        )

        if predicted_rewards:
            predicted_rewards_tensor = torch.stack(
                predicted_rewards, dim=1
            )
            per_transition_reward_loss = F.mse_loss(
                predicted_rewards_tensor,
                target_rewards,
                reduction="none",
            )
            reward_loss = self._masked_mean(
                per_transition_reward_loss, dynamics_mask
            )
        else:
            reward_loss = observations.new_zeros(())

        total = (
            self.loss_weights.policy * policy_loss
            + self.loss_weights.value * value_loss
            + self.loss_weights.reward * reward_loss
        )
        return MuZeroLosses(
            total,
            policy_loss,
            value_loss,
            reward_loss,
            policy_target_entropy,
            policy_kl,
        )

    def train_step(self, batch: ReplayBatch) -> dict[str, float]:
        """Apply one optimizer update and return detached scalar metrics."""
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_loss(batch)
        losses.total.backward()
        device = next(self.network.parameters()).device
        optimizer_step(self.optimizer, device)
        return {
            "loss": float(losses.total.detach()),
            "policy_loss": float(losses.policy.detach()),
            "value_loss": float(losses.value.detach()),
            "reward_loss": float(losses.reward.detach()),
            "policy_target_entropy": float(
                losses.policy_target_entropy.detach()
            ),
            "policy_kl": float(losses.policy_kl.detach()),
        }

    @staticmethod
    def _masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _validate_shapes(
        self,
        observations: Tensor,
        actions: Tensor,
        target_rewards: Tensor,
        target_policies: Tensor,
        target_values: Tensor,
        dynamics_mask: Tensor,
        prediction_mask: Tensor,
    ) -> None:
        batch_size, num_unroll_steps = actions.shape
        prediction_shape = (batch_size, num_unroll_steps + 1)
        if observations.shape != (
            batch_size,
            3,
            self.network.board_size,
            self.network.board_size,
        ):
            raise ValueError("observations have an incompatible shape")
        if target_rewards.shape != actions.shape:
            raise ValueError("target_rewards must have shape [B, K]")
        if dynamics_mask.shape != actions.shape:
            raise ValueError("dynamics_mask must have shape [B, K]")
        if target_values.shape != prediction_shape:
            raise ValueError("target_values must have shape [B, K+1]")
        if prediction_mask.shape != prediction_shape:
            raise ValueError("prediction_mask must have shape [B, K+1]")
        if target_policies.shape != (
            *prediction_shape,
            self.network.action_space_size,
        ):
            raise ValueError(
                "target_policies must have shape [B, K+1, action_space_size]"
            )
