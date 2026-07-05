import numpy as np
import pytest

from gomoku_muzero.training.replay import GameHistory, ReplayBuffer


def make_game(num_moves: int = 3, action_space_size: int = 4) -> GameHistory:
    observations = [
        np.full((3, 2, 2), position, dtype=np.float32)
        for position in range(num_moves + 1)
    ]
    policies = [
        np.eye(action_space_size, dtype=np.float32)[position]
        if position < num_moves
        else np.zeros(action_space_size, dtype=np.float32)
        for position in range(num_moves + 1)
    ]
    return GameHistory(
        observations=observations,
        actions=list(range(num_moves)),
        rewards=[0.0] * (num_moves - 1) + [1.0],
        policies=policies,
        values=[1.0, -1.0, 1.0, 0.0][: num_moves + 1],
        to_play=[1 if position % 2 == 0 else -1
                 for position in range(num_moves + 1)],
    )


def test_game_validation_checks_time_dimensions() -> None:
    game = make_game()
    game.rewards.pop()

    with pytest.raises(ValueError, match="rewards"):
        game.validate(action_space_size=4)


def test_buffer_is_bounded_fifo() -> None:
    buffer = ReplayBuffer(capacity=2, action_space_size=4)
    first = make_game(1)
    second = make_game(2)
    third = make_game(3)

    buffer.save_game(first)
    buffer.save_game(second)
    buffer.save_game(third)

    assert len(buffer) == 2
    assert list(buffer.games) == [second, third]


def test_sample_batch_shapes_and_dtypes() -> None:
    buffer = ReplayBuffer(capacity=2, action_space_size=4, seed=0)
    buffer.save_game(make_game())

    batch = buffer.sample_batch(batch_size=5, num_unroll_steps=2)

    assert batch.observations.shape == (5, 3, 2, 2)
    assert batch.actions.shape == (5, 2)
    assert batch.actions.dtype == np.int64
    assert batch.target_rewards.shape == (5, 2)
    assert batch.target_policies.shape == (5, 3, 4)
    assert batch.target_values.shape == (5, 3)
    assert batch.dynamics_mask.shape == (5, 2)
    assert batch.prediction_mask.shape == (5, 3)
    assert batch.sampled_game_ages.shape == (5,)
    assert batch.observations.dtype == np.float32
    assert batch.target_policies.dtype == np.float32


def test_sample_near_terminal_is_padded_and_masked() -> None:
    buffer = ReplayBuffer(capacity=1, action_space_size=4, seed=0)
    buffer.save_game(make_game(num_moves=1))

    batch = buffer.sample_batch(batch_size=1, num_unroll_steps=3)

    # The real transition keeps its action; absorbing padding carries
    # random in-range actions with zero reward and value targets.
    assert batch.actions[0][0] == 0
    assert all(0 <= action < 4 for action in batch.actions[0])
    np.testing.assert_array_equal(batch.target_rewards[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(batch.dynamics_mask[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        batch.prediction_mask[0], [1.0, 1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(batch.target_values[0], [1.0, -1.0, 0, 0])


def test_replay_stores_no_hidden_states() -> None:
    game = make_game()

    assert "hidden" not in game.__dataclass_fields__


def test_empty_buffer_cannot_be_sampled() -> None:
    buffer = ReplayBuffer(capacity=1, action_space_size=4)

    with pytest.raises(ValueError, match="empty"):
        buffer.sample_batch(batch_size=1, num_unroll_steps=1)


def test_recent_sampling_favors_newer_games() -> None:
    buffer = ReplayBuffer(
        capacity=4,
        action_space_size=4,
        seed=0,
        sampling="recent",
        recency_half_life=0.5,
    )
    for marker in range(4):
        game = make_game(num_moves=1)
        game.observations[0].fill(marker)
        buffer.save_game(game)

    batch = buffer.sample_batch(batch_size=2_000, num_unroll_steps=0)
    markers = batch.observations[:, 0, 0, 0]

    assert np.count_nonzero(markers == 3) > np.count_nonzero(markers == 0)
    assert batch.sampled_game_ages.mean() < 1.0


def test_symmetry_augmentation_keeps_sample_consistent() -> None:
    """Observation, policy, and action must transform together."""
    buffer = ReplayBuffer(
        capacity=1,
        action_space_size=4,
        seed=3,
        augment_symmetries=True,
    )
    # One move at cell 1; plane 0 marks that cell, policy is one-hot there.
    observations = [
        np.zeros((3, 2, 2), dtype=np.float32),
        np.zeros((3, 2, 2), dtype=np.float32),
    ]
    observations[0][0, 0, 1] = 1.0
    policies = [
        np.eye(4, dtype=np.float32)[1],
        np.zeros(4, dtype=np.float32),
    ]
    buffer.save_game(
        GameHistory(
            observations=observations,
            actions=[1],
            rewards=[1.0],
            policies=policies,
            values=[1.0, 0.0],
            to_play=[1, -1],
        )
    )

    batch = buffer.sample_batch(batch_size=64, num_unroll_steps=0)

    for sample in range(64):
        marked_cell = int(batch.observations[sample][0].ravel().argmax())
        policy_cell = int(batch.target_policies[sample][0].argmax())
        assert marked_cell == policy_cell
    # Across many samples every corner should appear at least once.
    assert len(
        {int(row[0].ravel().argmax()) for row in batch.observations}
    ) == 4


def test_augmented_actions_follow_the_same_symmetry() -> None:
    buffer = ReplayBuffer(
        capacity=1,
        action_space_size=4,
        seed=5,
        augment_symmetries=True,
    )
    observations = [
        np.zeros((3, 2, 2), dtype=np.float32),
        np.zeros((3, 2, 2), dtype=np.float32),
    ]
    observations[0][0, 0, 1] = 1.0
    buffer.save_game(
        GameHistory(
            observations=observations,
            actions=[1],
            rewards=[1.0],
            policies=[
                np.eye(4, dtype=np.float32)[1],
                np.zeros(4, dtype=np.float32),
            ],
            values=[1.0, 0.0],
            to_play=[1, -1],
        )
    )

    batch = buffer.sample_batch(batch_size=64, num_unroll_steps=1)

    for sample in range(64):
        marked_cell = int(batch.observations[sample][0].ravel().argmax())
        assert int(batch.actions[sample][0]) == marked_cell


def test_root_values_length_is_validated() -> None:
    game = make_game()
    game.root_values = [0.0]

    with pytest.raises(ValueError, match="root_values"):
        game.validate(action_space_size=4)


def test_uniform_sampling_remains_available() -> None:
    buffer = ReplayBuffer(
        capacity=4,
        action_space_size=4,
        seed=0,
        sampling="uniform",
    )
    for _ in range(4):
        buffer.save_game(make_game(num_moves=1))

    batch = buffer.sample_batch(batch_size=100, num_unroll_steps=0)

    assert set(batch.sampled_game_ages) == {0, 1, 2, 3}
