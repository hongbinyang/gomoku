import torch

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.search.mcts import MCTS, MCTSConfig
from gomoku_muzero.training.async_pipeline import AsyncMuZeroPipeline
from gomoku_muzero.training.pipeline import LearningConfig
from gomoku_muzero.training.replay import ReplayBuffer
from gomoku_muzero.training.trainer import MuZeroTrainer
from gomoku_muzero.workflows.self_play import SelfPlayConfig
from gomoku_muzero.workflows.self_play_actor import (
    PublishedWeights,
    SelfPlayActor,
)


def test_published_weights_are_versioned_snapshots() -> None:
    network = MuZeroNetwork(board_size=2, hidden_channels=4)
    published = PublishedWeights(network)
    version, snapshot = published.newer_than(-1)  # type: ignore[misc]
    first_value = next(iter(snapshot.values())).clone()

    with torch.no_grad():
        next(network.parameters()).add_(1)
    new_version = published.publish(network)

    assert new_version == version + 1
    assert torch.equal(next(iter(snapshot.values())), first_value)
    assert published.newer_than(new_version) is None


def test_actor_generates_replay_ready_game() -> None:
    network = MuZeroNetwork(board_size=2, hidden_channels=4)
    actor = SelfPlayActor(
        board_size=2,
        win_length=2,
        hidden_channels=4,
        device=torch.device("cpu"),
        weights=PublishedWeights(network),
        mcts_config=MCTSConfig(num_simulations=1),
        self_play_config=SelfPlayConfig(
            temperature=0,
            temperature_moves=0,
            add_exploration_noise=False,
        ),
        queue_size=1,
        seed=0,
    )

    actor.start()
    try:
        game = actor.get_game()
    finally:
        actor.stop()

    game.validate(action_space_size=4)
    assert game.network_version == 0
    assert actor.games_generated >= 1


def test_async_pipeline_overlaps_actor_and_learner() -> None:
    env = GomokuEnv(board_size=2, win_length=2)
    network = MuZeroNetwork(board_size=2, hidden_channels=4)
    mcts = MCTS(network, MCTSConfig(num_simulations=1), seed=0)
    replay = ReplayBuffer(4, env.action_space_size, seed=0)
    pipeline = AsyncMuZeroPipeline(
        env,
        mcts,
        replay,
        MuZeroTrainer(network),
        LearningConfig(
            games_per_iteration=1,
            training_steps_per_iteration=1,
            batch_size=1,
            num_unroll_steps=1,
            evaluation_interval=2,
            evaluation_games=1,
        ),
        SelfPlayConfig(
            temperature=0,
            temperature_moves=0,
            add_exploration_noise=False,
        ),
        self_play_queue_size=1,
        actor_seed=1,
    )

    pipeline.start()
    try:
        result = pipeline.run_iteration(1)
    finally:
        pipeline.stop()

    assert len(replay) == 1
    assert result.latest_metrics is not None
    assert pipeline.published_weights.newer_than(0) is not None
