"""K-step MuZero training through representation, dynamics, and prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.runtime.device import optimizer_step
from gomoku_muzero.training.replay import ReplayBatch


# Updates are clipped to this global norm. Chosen from observed healthy
# training (norms of 1-8); the logged grad_norm metric reports the
# pre-clip value, so growth beyond this bound stays visible.
MAX_GRAD_NORM = 10.0


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of MuZero's three supervised objectives."""

    policy: float = 1.0
    value: float = 1.0
    reward: float = 1.0


@dataclass(frozen=True)
class MuZeroLosses:
    """Scalar differentiable losses from one replay batch.

    ``policy``, ``value``, and ``reward`` are the optimized quantities with
    the paper's per-unroll-step 1/K weighting. ``policy_ce``,
    ``policy_target_entropy``, and ``policy_kl`` are unweighted diagnostics
    averaged over real (searched) positions only, satisfying
    ``policy_ce = policy_target_entropy + policy_kl``.
    """

    total: Tensor
    policy: Tensor
    value: Tensor
    reward: Tensor
    policy_ce: Tensor
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
        """Unroll K model steps and compute the MuZero training loss.

        Prediction index 0 comes from ``h(observation)`` followed by ``f``.
        Prediction indices 1 through K come from repeated ``g`` then ``f``.
        Reward index k supervises the transition from position k to k+1.

        Two stabilizers from the paper are applied: the hidden state's
        gradient is halved at every dynamics application, and each unrolled
        step's loss is scaled by 1/K (the initial prediction keeps weight 1).
        Value and reward are supervised at every step, including absorbing
        padding beyond terminal states where the targets are zero. The
        policy loss vanishes naturally on positions with all-zero targets.
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

        num_unroll_steps = actions.shape[1]
        initial = self.network.initial_inference(observations)
        hidden_state = initial.hidden_state
        policy_logits = [initial.policy_logits]
        predicted_values = [initial.value.squeeze(-1)]
        predicted_rewards: list[Tensor] = []

        for step in range(num_unroll_steps):
            recurrent = self.network.recurrent_inference(
                hidden_state, actions[:, step]
            )
            # Halve the gradient flowing back through the dynamics chain
            # (Appendix G of the MuZero paper) to stabilize deep unrolls.
            hidden_state = (
                0.5 * recurrent.hidden_state
                + 0.5 * recurrent.hidden_state.detach()
            )
            predicted_rewards.append(recurrent.reward.squeeze(-1))
            policy_logits.append(recurrent.policy_logits)
            predicted_values.append(recurrent.value.squeeze(-1))

        policy_logits_tensor = torch.stack(policy_logits, dim=1)
        predicted_values_tensor = torch.stack(predicted_values, dim=1)

        # Per-step loss weights: the initial prediction keeps weight one and
        # every unrolled step is scaled by 1/K, following the paper.
        unroll_weight = (
            1.0 / num_unroll_steps if num_unroll_steps > 0 else 1.0
        )
        prediction_weights = observations.new_full(
            (1, num_unroll_steps + 1), unroll_weight
        )
        prediction_weights[0, 0] = 1.0

        per_position_policy_loss = -(
            target_policies
            * F.log_softmax(policy_logits_tensor, dim=-1)
        ).sum(dim=-1)
        # Zero policy targets (terminal and absorbing positions) contribute
        # exactly zero, so no extra mask is needed for the optimized loss.
        policy_loss = (
            (per_position_policy_loss * prediction_weights).sum(dim=1).mean()
        )

        per_position_value_loss = F.mse_loss(
            predicted_values_tensor, target_values, reduction="none"
        )
        # Absorbing padding beyond terminal states is supervised toward its
        # zero target so search cannot exploit hallucinated values there.
        value_loss = (
            (per_position_value_loss * prediction_weights).sum(dim=1).mean()
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
            reward_loss = (
                per_transition_reward_loss.sum(dim=1).mean() * unroll_weight
            )
        else:
            reward_loss = observations.new_zeros(())

        # Diagnostics over real searched positions only, unweighted.
        policy_mask = prediction_mask * (
            target_policies.sum(dim=-1) > 0
        ).to(torch.float32)
        policy_ce = self._masked_mean(per_position_policy_loss, policy_mask)
        per_position_target_entropy = -(
            target_policies
            * target_policies.clamp_min(1e-8).log()
        ).sum(dim=-1)
        policy_target_entropy = self._masked_mean(
            per_position_target_entropy, policy_mask
        )
        # CE(target, prediction) = H(target) + KL(target || prediction).
        policy_kl = (policy_ce - policy_target_entropy).clamp_min(0.0)

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
            policy_ce,
            policy_target_entropy,
            policy_kl,
        )

    def train_step(self, batch: ReplayBatch) -> dict[str, float]:
        """Apply one optimizer update and return detached scalar metrics."""
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_loss(batch)
        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=MAX_GRAD_NORM
        )
        device = next(self.network.parameters()).device
        optimizer_step(self.optimizer, device)
        return {
            "loss": float(losses.total.detach()),
            "policy_loss": float(losses.policy.detach()),
            "value_loss": float(losses.value.detach()),
            "reward_loss": float(losses.reward.detach()),
            "policy_ce": float(losses.policy_ce.detach()),
            "policy_target_entropy": float(
                losses.policy_target_entropy.detach()
            ),
            "policy_kl": float(losses.policy_kl.detach()),
            "grad_norm": float(grad_norm),
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
