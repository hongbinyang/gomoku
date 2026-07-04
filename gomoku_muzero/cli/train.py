"""Run a small end-to-end MuZero experiment with ``python -m``."""

from __future__ import annotations

import argparse
import platform

import numpy as np
import torch

from gomoku_muzero.model.checkpoint import save_checkpoint
from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS, MCTSConfig
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.runtime.device import (
    device_memory_metrics,
    resolve_device,
)
from gomoku_muzero.runtime.metrics import RunLogger
from gomoku_muzero.training.async_pipeline import AsyncMuZeroPipeline
from gomoku_muzero.training.pipeline import LearningConfig, MuZeroPipeline
from gomoku_muzero.training.replay import ReplayBuffer
from gomoku_muzero.training.trainer import MuZeroTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="self-play/training iterations to run (default: 10)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=25,
        help="MCTS simulations per selected move (default: 25)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=10,
        help="board width and height (default: 10)",
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=5,
        help="consecutive stones needed to win (default: 5)",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=2,
        help="new self-play games per iteration (default: 2)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10,
        help="optimizer updates per iteration (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="replay samples per optimizer update (default: 32)",
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=5,
        help="recurrent dynamics steps per sample (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=500,
        help="maximum complete games retained (default: 500)",
    )
    parser.add_argument(
        "--replay-sampling",
        choices=("recent", "uniform"),
        default="recent",
        help="game sampling distribution (default: recent)",
    )
    parser.add_argument(
        "--replay-half-life",
        type=float,
        default=100.0,
        help="recency sampling half-life in games (default: 100)",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=5,
        help="iterations between evaluations (default: 5)",
    )
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=20,
        help="games against random per evaluation (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducible runs (default: 0)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="where to save latest model weights (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "tpu"),
        default="auto",
        help="compute backend (default: auto)",
    )
    parser.add_argument(
        "--self-play-mode",
        choices=("async", "sync"),
        default="async",
        help="overlap self-play with training or run sequentially (default: async)",
    )
    parser.add_argument(
        "--self-play-queue-size",
        type=int,
        default=4,
        help="maximum completed games waiting for learner (default: 4)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="metrics run name (default: UTC timestamp)",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="root directory for metrics runs (default: runs)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="also write TensorBoard event files",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.backend == "tpu" and args.self_play_mode == "async":
        parser.error(
            "TPU/XLA currently requires --self-play-mode sync; "
            "process-based TPU actors are not implemented yet"
        )
    env = GomokuEnv(
        board_size=args.board_size,
        win_length=args.win_length,
    )
    network = MuZeroNetwork(
        board_size=args.board_size,
        hidden_channels=32,
    ).to(device.torch_device)
    mcts = MCTS(
        network,
        MCTSConfig(num_simulations=args.simulations),
        seed=args.seed,
    )
    replay = ReplayBuffer(
        capacity=args.replay_capacity,
        action_space_size=env.action_space_size,
        seed=args.seed,
        sampling=args.replay_sampling,
        recency_half_life=args.replay_half_life,
    )
    trainer = MuZeroTrainer(
        network,
        learning_rate=args.learning_rate,
    )
    pipeline_class = (
        AsyncMuZeroPipeline
        if args.self_play_mode == "async"
        else MuZeroPipeline
    )
    pipeline_kwargs = {}
    if args.self_play_mode == "async":
        pipeline_kwargs = {
            "self_play_queue_size": args.self_play_queue_size,
            "actor_seed": args.seed + 10_000,
        }
    pipeline = pipeline_class(
        env,
        mcts,
        replay,
        trainer,
        LearningConfig(
            games_per_iteration=args.games_per_iteration,
            training_steps_per_iteration=args.training_steps,
            batch_size=args.batch_size,
            num_unroll_steps=args.unroll_steps,
            evaluation_interval=args.evaluation_interval,
            evaluation_games=args.evaluation_games,
        ),
        **pipeline_kwargs,
    )
    logger = RunLogger(
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        tensorboard=args.tensorboard,
    )
    logger.write_config(
        {
            **vars(args),
            "resolved_device": device.description,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
    )

    progress_width = 140
    print(
        f"device={device.description} "
        f"self_play={args.self_play_mode}"
    )

    def show_progress(message: str) -> None:
        print(
            f"\r{message:<{progress_width}}",
            end="",
            flush=True,
        )

    if isinstance(pipeline, AsyncMuZeroPipeline):
        pipeline.start()
    try:
        for iteration in range(1, args.iterations + 1):
            result = pipeline.run_iteration(
                iteration,
                progress_callback=show_progress,
            )
            message = (
                f"iteration={result.iteration} "
                f"games={result.games_generated} "
                f"loss={result.latest_metrics['loss']:.3f} "
                f"policy={result.latest_metrics['policy_loss']:.3f} "
                f"entropy={result.latest_metrics['policy_target_entropy']:.3f} "
                f"kl={result.latest_metrics['policy_kl']:.3f} "
                f"value={result.latest_metrics['value_loss']:.3f} "
                f"reward={result.latest_metrics['reward_loss']:.3f}"
            )
            if result.evaluation is not None:
                evaluation = result.evaluation
                message += (
                    f" eval={evaluation.wins}W/"
                    f"{evaluation.draws}D/{evaluation.losses}L"
                )
            metrics = {
                **(result.latest_metrics or {}),
                **result.operational_metrics,
                **device_memory_metrics(device.torch_device),
            }
            if result.evaluation is not None:
                metrics.update(
                    {
                        "evaluation_wins": result.evaluation.wins,
                        "evaluation_draws": result.evaluation.draws,
                        "evaluation_losses": result.evaluation.losses,
                        "evaluation_score": result.evaluation.score,
                    }
                )
            logger.record("iteration", result.iteration, metrics)
            save_checkpoint(network, args.checkpoint, env.win_length)
            print(f"\r{message:<{progress_width}}")
    finally:
        if isinstance(pipeline, AsyncMuZeroPipeline):
            pipeline.stop()
        logger.close()
    print(f"checkpoint={args.checkpoint}")
    print(f"run_dir={logger.run_dir}")


if __name__ == "__main__":
    main()
