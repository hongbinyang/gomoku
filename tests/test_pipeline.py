from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS, MCTSConfig
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.training.pipeline import LearningConfig, MuZeroPipeline
from gomoku_muzero.training.replay import ReplayBuffer
from gomoku_muzero.workflows.self_play import SelfPlayConfig
from gomoku_muzero.training.trainer import MuZeroTrainer


def test_one_end_to_end_learning_iteration() -> None:
    env = GomokuEnv(board_size=2, win_length=2)
    network = MuZeroNetwork(board_size=2, hidden_channels=4)
    mcts = MCTS(network, MCTSConfig(num_simulations=1), seed=0)
    replay = ReplayBuffer(4, env.action_space_size, seed=0)
    trainer = MuZeroTrainer(network)
    pipeline = MuZeroPipeline(
        env,
        mcts,
        replay,
        trainer,
        LearningConfig(
            games_per_iteration=1,
            training_steps_per_iteration=1,
            batch_size=2,
            num_unroll_steps=2,
            evaluation_interval=1,
            evaluation_games=2,
        ),
        SelfPlayConfig(
            temperature=0,
            temperature_moves=0,
            add_exploration_noise=False,
        ),
    )

    result = pipeline.run_iteration(iteration=1)

    assert len(replay) == 1
    assert result.latest_metrics is not None
    assert result.operational_metrics["iteration_seconds"] >= 0
    assert result.operational_metrics["moves_generated"] >= 1
    assert "value_calibration_mae" in result.operational_metrics
    assert result.evaluation is not None
    assert (
        result.evaluation.wins
        + result.evaluation.draws
        + result.evaluation.losses
        == 2
    )
