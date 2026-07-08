"""Monte Carlo tree search over MuZero's learned hidden states."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.model.networks import MuZeroNetwork


@dataclass
class MCTSConfig:
    """The small set of search hyperparameters used in this implementation."""

    num_simulations: int = 50
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    # Fraction of prior mass redistributed uniformly over tactically hot
    # cells (either side's direct-threat moves) when the real environment
    # is available. Guarantees search attention on threats the learned
    # policy underrates; 0 disables the bias.
    threat_prior_fraction: float = 0.25


class MinMaxStats:
    """Track the observed Q-value range so PUCT can normalize to [0, 1].

    MuZero normalizes Q values with the minimum and maximum seen inside the
    current search tree, keeping the exploitation term on a stable scale
    relative to the prior-driven exploration term.
    """

    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class Node:
    """One latent search state.

    ``prior`` is the probability assigned by the parent policy. ``reward`` is
    the predicted reward for the parent player after entering this node.
    ``value_sum`` stores values from this node's player-to-move perspective.
    ``predicted_value`` caches the network value produced when the node was
    expanded so revisits never repeat inference. ``terminal`` marks nodes
    whose incoming move provably ended the real game; their reward is exact
    (1.0 win, 0.0 draw), their value is zero, and they are never expanded.
    ``proven_value`` marks nodes decided by static threat analysis (the
    player to move either has an immediate win, +1, or faces an unstoppable
    double threat, -1); they are also never expanded.
    """

    prior: float
    to_play: int
    reward: float = 0.0
    hidden_state: Tensor | None = None
    predicted_value: float = 0.0
    terminal: bool = False
    proven_value: float | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)

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
        if self.config.pb_c_init < 0:
            raise ValueError("pb_c_init must be non-negative")
        if self.config.pb_c_base <= 0:
            raise ValueError("pb_c_base must be positive")
        if not 0 <= self.config.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= self.config.threat_prior_fraction < 1:
            raise ValueError("threat_prior_fraction must be in [0, 1)")
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        observation: np.ndarray | Tensor,
        legal_actions: Sequence[int],
        to_play: int,
        add_exploration_noise: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        reuse_root: Node | None = None,
        env: GomokuEnv | None = None,
    ) -> Node:
        """Build and return a search tree rooted at a real observation.

        The input observation has shape ``[3,N,N]`` (or batched
        ``[1,3,N,N]``). Search inference always uses batch size one.

        ``reuse_root`` may pass the subtree that a previous search built
        under the action that was actually played. Its statistics are kept
        and its hidden state is re-grounded on the real observation via
        the representation function. ``num_simulations`` then acts as a
        target for the root's total visit count: only the missing
        simulations are run (always at least one).

        ``env`` optionally provides the real environment matching the root
        observation. When present, the search replays each simulation's
        action path on a clone and applies known-rules reasoning instead of
        trusting the learned model: terminal moves are pinned to their
        exact reward, in-tree rewards of non-terminal moves are exactly
        zero, and one ply of static threat analysis proves immediate wins
        (+1), unstoppable double threats (-1), and restricts expansion to
        the forced block when the opponent threatens exactly one cell.
        Like in-tree legality tracking, this is a deliberate known-rules
        extension of MuZero. The caller's environment is never mutated.
        """
        legal_actions = self._validate_legal_actions(legal_actions)
        if to_play not in (-1, 1):
            raise ValueError("to_play must be -1 or +1")

        self.network.eval()
        observation_tensor = self._observation_tensor(observation)
        with torch.inference_mode():
            initial = self.network.initial_inference(observation_tensor)

        root: Node | None = None
        if (
            reuse_root is not None
            and reuse_root.children
            and reuse_root.to_play == to_play
            and set(reuse_root.children) == set(legal_actions)
        ):
            root = reuse_root
            root.reward = 0.0
            root.hidden_state = initial.hidden_state.detach()
            root.predicted_value = float(initial.value.item())
        if root is None:
            root = Node(
                prior=1.0,
                to_play=to_play,
                hidden_state=initial.hidden_state.detach(),
                predicted_value=float(initial.value.item()),
            )
            self._expand(
                root,
                legal_actions,
                initial.policy_logits[0],
                boost_actions=(
                    self._tactical_cells(env) if env is not None else ()
                ),
            )
        if add_exploration_noise:
            self.add_exploration_noise(root)

        num_simulations = max(
            1, self.config.num_simulations - root.visit_count
        )
        min_max_stats = MinMaxStats()
        report_every = max(1, (num_simulations + 19) // 20)
        for simulation in range(1, num_simulations + 1):
            node = root
            search_path = [node]
            action_path: list[int] = []

            while node.children:
                action, node = self.select_child(
                    search_path[-1], min_max_stats
                )
                action_path.append(action)
                search_path.append(node)
                if node.hidden_state is None:
                    break

            if node.terminal:
                # Provably finished: exact reward is pinned, value is zero.
                leaf_value = 0.0
            elif node.proven_value is not None:
                leaf_value = node.proven_value
            elif node.hidden_state is not None:
                # Root without children, or a revisited leaf whose expansion
                # produced no children: reuse the cached network value
                # instead of repeating inference.
                leaf_value = node.predicted_value
            else:
                leaf_value = self._expand_leaf(
                    node, search_path[-2], action_path, env
                )

            self.backup(search_path, leaf_value, min_max_stats)
            if progress_callback is not None and (
                simulation % report_every == 0
                or simulation == num_simulations
            ):
                progress_callback(simulation, num_simulations)

        return root

    def select_child(
        self,
        parent: Node,
        min_max_stats: MinMaxStats | None = None,
    ) -> tuple[int, Node]:
        """Choose the child with maximal normalized Q + PUCT exploration.

        Follows the paper's PUCT formula: ``pb_c`` grows logarithmically
        with the parent visit count, and visited children's Q values are
        normalized to [0, 1] using the range observed during this search.
        Unvisited children score zero on the exploitation term.
        """
        if not parent.children:
            raise ValueError("cannot select from a node without children")
        stats = min_max_stats or MinMaxStats()
        pb_c = (
            log(
                (parent.visit_count + self.config.pb_c_base + 1)
                / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )

        def score(item: tuple[int, Node]) -> tuple[float, int]:
            action, child = item
            if child.visit_count > 0:
                q_value = stats.normalize(
                    child.reward - self.config.discount * child.value
                )
            else:
                q_value = 0.0
            exploration = (
                pb_c
                * child.prior
                * sqrt(parent.visit_count)
                / (child.visit_count + 1)
            )
            # Prefer the lower action index when scores are exactly equal.
            return q_value + exploration, -action

        return max(parent.children.items(), key=score)

    def backup(
        self,
        search_path: Sequence[Node],
        leaf_value: float,
        min_max_stats: MinMaxStats | None = None,
    ) -> None:
        """Back up a leaf value through an alternating-player path."""
        value = leaf_value
        for index, node in enumerate(reversed(search_path)):
            node.value_sum += value
            node.visit_count += 1
            if min_max_stats is not None and index < len(search_path) - 1:
                # Record the Q value this node now presents to its parent.
                min_max_stats.update(
                    node.reward - self.config.discount * node.value
                )
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

        # A proven immediate win can never be worse than any alternative,
        # so it overrides both visit counts and sampling temperature. This
        # removes the "won position but wanders" behavior that discount=1
        # otherwise permits.
        winning_actions = [
            action
            for action, child in root.children.items()
            if child.terminal and child.reward == 1.0
        ]
        if winning_actions:
            return min(winning_actions)

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

    def _expand_leaf(
        self,
        node: Node,
        parent: Node,
        action_path: Sequence[int],
        env: GomokuEnv | None,
    ) -> float:
        """Evaluate a freshly reached leaf and expand it when appropriate.

        With a real environment, known-rules reasoning runs first: exact
        terminal rewards, static win/loss proofs, and forced-block
        pruning. The network is consulted only for positions the rules
        cannot decide.
        """
        action = action_path[-1]
        simulation = self._replay(env, action_path) if env is not None else None
        if simulation is not None and simulation.terminated:
            node.terminal = True
            node.reward = (
                1.0 if simulation.winner != simulation.EMPTY else 0.0
            )
            node.predicted_value = 0.0
            return 0.0

        forced_block: int | None = None
        if simulation is not None:
            # One ply of static threat analysis with the real rules.
            node.reward = 0.0  # non-terminal moves never carry reward
            if simulation.winning_actions(simulation.current_player):
                node.proven_value = 1.0
                node.predicted_value = 1.0
                return 1.0
            opponent_wins = simulation.winning_actions(
                -simulation.current_player
            )
            if len(opponent_wins) >= 2:
                node.proven_value = -1.0
                node.predicted_value = -1.0
                return -1.0
            if len(opponent_wins) == 1:
                # Every other move loses to the opponent's completion.
                forced_block = opponent_wins[0]

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
        node.predicted_value = float(recurrent.value.item())
        boost_actions: Sequence[int] = ()
        if simulation is None:
            node.reward = float(recurrent.reward.item())
            expand_actions: Sequence[int] = [
                candidate
                for candidate in parent.children
                if candidate != action
            ]
        elif forced_block is not None:
            expand_actions = [forced_block]
        else:
            expand_actions = simulation.legal_actions()
            boost_actions = self._tactical_cells(simulation)
        self._expand(
            node,
            expand_actions,
            recurrent.policy_logits[0],
            boost_actions=boost_actions,
        )
        return node.predicted_value

    def _tactical_cells(self, env: GomokuEnv) -> list[int]:
        """Cells that win immediately or create a direct threat, either side.

        Immediate completions are the most urgent tier — without boosting
        them, the search may never visit its own winning move among many
        legal actions. Threat-makers are attacking continuations for the
        player to move and squares to deny for the opponent. All deserve
        search attention even when the learned policy underrates them.
        """
        if self.config.threat_prior_fraction == 0:
            return []
        player = env.current_player
        cells = set(env.winning_actions(player))
        cells.update(env.winning_actions(-player))
        cells.update(env.threat_actions(player))
        cells.update(env.threat_actions(-player))
        return sorted(cells)

    @staticmethod
    def _replay(
        env: GomokuEnv, action_path: Sequence[int]
    ) -> GomokuEnv | None:
        """Replay a simulation's actions on a clone of the real game.

        Returns the resulting environment, or ``None`` for a stale path
        that terminates before its last action — possible only if a caller
        mixes searches with and without ``env`` on one reused tree — in
        which case the caller falls back to the learned model.
        """
        simulation = env.clone()
        for index, action in enumerate(action_path):
            _, _, terminated, _ = simulation.step(action)
            if terminated and index < len(action_path) - 1:
                return None
        return simulation

    def _expand(
        self,
        node: Node,
        legal_actions: Sequence[int],
        policy_logits: Tensor,
        boost_actions: Sequence[int] = (),
    ) -> None:
        if not legal_actions:
            return
        action_tensor = torch.tensor(
            legal_actions, dtype=torch.long, device=policy_logits.device
        )
        priors = torch.softmax(policy_logits[action_tensor], dim=0).tolist()
        boost = set(boost_actions) & set(legal_actions)
        if boost:
            # Redistribute a fixed fraction of prior mass uniformly over
            # tactically hot cells so threats are searched even when the
            # learned policy assigns them negligible probability.
            fraction = self.config.threat_prior_fraction
            share = fraction / len(boost)
            priors = [
                (1 - fraction) * prior
                + (share if action in boost else 0.0)
                for action, prior in zip(legal_actions, priors)
            ]
        for action, prior in zip(legal_actions, priors):
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
