import numpy as np
import pytest
import torch
from torch import Tensor, nn

from gomoku_muzero.search.mcts import MCTS, MCTSConfig, Node
from gomoku_muzero.model.networks import (
    InitialInferenceOutput,
    RecurrentInferenceOutput,
)


class FakeNetwork(nn.Module):
    """Deterministic network that makes MCTS behavior easy to inspect."""

    board_size = 2
    action_space_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def initial_inference(self, observation: Tensor) -> InitialInferenceOutput:
        batch_size = observation.shape[0]
        hidden = torch.zeros(batch_size, 1, 2, 2)
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        value = torch.zeros(batch_size, 1)
        return InitialInferenceOutput(hidden, logits, value)

    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor
    ) -> RecurrentInferenceOutput:
        batch_size = action.shape[0]
        next_hidden = hidden_state + 1
        reward = (action == 3).float().view(batch_size, 1)
        logits = torch.zeros(batch_size, 4)
        value = torch.full((batch_size, 1), 0.25)
        return RecurrentInferenceOutput(
            next_hidden, reward, logits, value
        )


def make_search(num_simulations: int = 8) -> MCTS:
    return MCTS(
        FakeNetwork(),  # type: ignore[arg-type]
        MCTSConfig(num_simulations=num_simulations),
        seed=7,
    )


def test_search_expands_only_legal_actions_and_counts_simulations() -> None:
    root = make_search().run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[1, 3],
        to_play=1,
    )

    assert set(root.children) == {1, 3}
    assert root.to_play == 1
    assert all(child.to_play == -1 for child in root.children.values())
    assert root.visit_count == 8
    assert sum(child.visit_count for child in root.children.values()) == 8
    assert root.children[3].visit_count > root.children[1].visit_count


def test_expansion_masks_policy_before_softmax() -> None:
    root = make_search(num_simulations=1).run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[0, 2],
        to_play=1,
    )

    expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=0).numpy()
    actual = np.array([root.children[0].prior, root.children[2].prior])
    np.testing.assert_allclose(actual, expected)


def test_backup_changes_perspective_and_includes_reward() -> None:
    search = make_search()
    root = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=-1, reward=1.0)

    search.backup([root, child], leaf_value=0.25)

    assert child.value == pytest.approx(0.25)
    assert root.value == pytest.approx(0.75)  # 1.0 - 0.25


def test_policy_target_is_normalized_visit_count_distribution() -> None:
    root = make_search().run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[1, 3],
        to_play=1,
    )

    target = make_search().policy_target(root)

    assert target.shape == (4,)
    assert target.sum() == pytest.approx(1.0)
    assert target[0] == 0
    assert target[2] == 0
    assert target[1] == pytest.approx(
        root.children[1].visit_count / 8
    )


def test_zero_temperature_selects_most_visited_action() -> None:
    root = make_search().run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[1, 3],
        to_play=1,
    )

    assert make_search().select_action(root, temperature=0) == 3


def test_childless_leaves_are_not_reexpanded() -> None:
    """A latent leaf with no legal continuations is inferred exactly once."""

    class CountingNetwork(FakeNetwork):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent_calls = 0

        def recurrent_inference(self, hidden_state, action):
            self.recurrent_calls += 1
            return super().recurrent_inference(hidden_state, action)

    network = CountingNetwork()
    search = MCTS(
        network,  # type: ignore[arg-type]
        MCTSConfig(num_simulations=6),
        seed=7,
    )

    # A single legal action: after one expansion the only child has no
    # remaining actions, so later simulations revisit a childless leaf.
    search.run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[2],
        to_play=1,
    )

    assert network.recurrent_calls == 1


def test_search_reports_simulation_progress() -> None:
    updates: list[tuple[int, int]] = []

    make_search(num_simulations=8).run(
        np.zeros((3, 2, 2), dtype=np.float32),
        legal_actions=[1, 3],
        to_play=1,
        progress_callback=lambda completed, total: updates.append(
            (completed, total)
        ),
    )

    assert updates[-1] == (8, 8)
