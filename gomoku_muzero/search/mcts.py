"""Monte Carlo tree search over MuZero's learned hidden states."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from gomoku_muzero.model.networks import MuZeroNetwork


@dataclass
class MCTSConfig:
    """The small set of search hyperparameters used in this implementation."""

    num_simulations: int = 50
    pb_c: float = 1.5
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


@dataclass
class Node:
    """One latent search state.

    ``prior`` is the probability assigned by the parent policy. ``reward`` is
    the predicted reward for the parent player after entering this node.
    ``value_sum`` stores values from this node's player-to-move perspective.
    """

    prior: float
    to_play: int
    reward: float = 0.0
    hidden_state: Tensor | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        """Mean backed-up value, from ``to_play``'s perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Run PUCT search using only MuZero network inference."""

    def __init__(
        self,
        network: MuZeroNetwork,
        config: MCTSConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.network = network
        self.config = config or MCTSConfig()
        if self.config.num_simulations < 1:
            raise ValueError("num_simulations must be positive")
        if self.config.pb_c < 0:
            raise ValueError("pb_c must be non-negative")
        if not 0 <= self.config.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        observation: np.ndarray | Tensor,
        legal_actions: Sequence[int],
        to_play: int,
        add_exploration_noise: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Node:
        """Build and return a search tree rooted at a real observation.

        The input observation has shape ``[3,N,N]`` (or batched
        ``[1,3,N,N]``). Search inference always uses batch size one.
        """
        legal_actions = self._validate_legal_actions(legal_actions)
        if to_play not in (-1, 1):
            raise ValueError("to_play must be -1 or +1")

        observation_tensor = self._observation_tensor(observation)
        with torch.inference_mode():
            initial = self.network.initial_inference(observation_tensor)

        root = Node(
            prior=1.0,
            to_play=to_play,
            hidden_state=initial.hidden_state.detach(),
        )
        self._expand(root, legal_actions, initial.policy_logits[0])
        if add_exploration_noise:
            self.add_exploration_noise(root)

        report_every = max(
            1, (self.config.num_simulations + 19) // 20
        )
        for simulation in range(1, self.config.num_simulations + 1):
            node = root
            search_path = [node]
            action_path: list[int] = []

            while node.children:
                action, node = self.select_child(search_path[-1])
                action_path.append(action)
                search_path.append(node)
                if node.hidden_state is None:
                    break

            if len(search_path) == 1:
                leaf_value = root.value
            else:
                parent = search_path[-2]
                action = action_path[-1]
                action_tensor = torch.tensor(
                    [action],
                    dtype=torch.long,
                    device=parent.hidden_state.device,
                )
                with torch.inference_mode():
                    recurrent = self.network.recurrent_inference(
                        parent.hidden_state, action_tensor
                    )

                node.hidden_state = recurrent.hidden_state.detach()
                node.reward = float(recurrent.reward.item())
                remaining_actions = [
                    candidate
                    for candidate in parent.children
                    if candidate != action
                ]
                self._expand(
                    node, remaining_actions, recurrent.policy_logits[0]
                )
                leaf_value = float(recurrent.value.item())

            self.backup(search_path, leaf_value)
            if progress_callback is not None and (
                simulation % report_every == 0
                or simulation == self.config.num_simulations
            ):
                progress_callback(
                    simulation, self.config.num_simulations
                )

        return root

    def select_child(self, parent: Node) -> tuple[int, Node]:
        """Choose the child with maximal Q + PUCT exploration bonus."""
        if not parent.children:
            raise ValueError("cannot select from a node without children")

        def score(item: tuple[int, Node]) -> tuple[float, int]:
            action, child = item
            q_value = child.reward - self.config.discount * child.value
            exploration = (
                self.config.pb_c
                * child.prior
                * sqrt(parent.visit_count + 1)
                / (child.visit_count + 1)
            )
            # Prefer the lower action index when scores are exactly equal.
            return q_value + exploration, -action

        return max(parent.children.items(), key=score)

    def backup(self, search_path: Sequence[Node], leaf_value: float) -> None:
        """Back up a leaf value through an alternating-player path."""
        value = leaf_value
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward - self.config.discount * value

    def policy_target(self, root: Node) -> np.ndarray:
        """Return the normalized root visit counts with shape ``[N*N]``."""
        target = np.zeros(
            self.network.action_space_size, dtype=np.float32
        )
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )
        if total_visits == 0:
            return target
        for action, child in root.children.items():
            target[action] = child.visit_count / total_visits
        return target

    def select_action(self, root: Node, temperature: float = 1.0) -> int:
        """Select a root action from visit counts.

        Temperature zero is greedy. Positive temperatures sample from
        ``visit_count ** (1 / temperature)``.
        """
        if not root.children:
            raise ValueError("cannot select an action from an empty root")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        actions = np.array(sorted(root.children), dtype=np.int64)
        visits = np.array(
            [root.children[int(action)].visit_count for action in actions],
            dtype=np.float64,
        )
        if temperature == 0:
            return int(actions[np.argmax(visits)])

        weights = visits ** (1.0 / temperature)
        if weights.sum() == 0:
            weights = np.array(
                [root.children[int(action)].prior for action in actions]
            )
        probabilities = weights / weights.sum()
        return int(self.rng.choice(actions, p=probabilities))

    def add_exploration_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into root priors for diverse self-play."""
        if not root.children:
            return
        fraction = self.config.root_exploration_fraction
        if not 0 <= fraction <= 1:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if self.config.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")

        noise = self.rng.dirichlet(
            [self.config.dirichlet_alpha] * len(root.children)
        )
        for child, sample in zip(root.children.values(), noise):
            child.prior = (
                (1 - fraction) * child.prior + fraction * float(sample)
            )

    def _expand(
        self, node: Node, legal_actions: Sequence[int], policy_logits: Tensor
    ) -> None:
        node.expanded = True
        if not legal_actions:
            return
        action_tensor = torch.tensor(
            legal_actions, dtype=torch.long, device=policy_logits.device
        )
        priors = torch.softmax(policy_logits[action_tensor], dim=0)
        for action, prior in zip(legal_actions, priors.tolist()):
            node.children[action] = Node(
                prior=prior,
                to_play=-node.to_play,
            )

    def _validate_legal_actions(
        self, legal_actions: Sequence[int]
    ) -> list[int]:
        actions = [int(action) for action in legal_actions]
        if len(actions) != len(set(actions)):
            raise ValueError("legal_actions must not contain duplicates")
        if any(
            action < 0 or action >= self.network.action_space_size
            for action in actions
        ):
            raise ValueError("legal action is outside the action space")
        return sorted(actions)

    def _observation_tensor(self, observation: np.ndarray | Tensor) -> Tensor:
        device = next(self.network.parameters()).device
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        )
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        expected = (1, 3, self.network.board_size, self.network.board_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"observation must have shape {expected[1:]}")
        return tensor
