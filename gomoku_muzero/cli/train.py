"""Run a small end-to-end MuZero experiment with ``python -m``."""

from __future__ import annotations

import argparse
import platform

import numpy as np
import torch

from gomoku_muzero.model.checkpoint import (
    load_training_state,
    save_checkpoint,
    save_training_state,
)
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
from gomoku_muzero.training.trainer import LossWeights, MuZeroTrainer
from gomoku_muzero.workflows.self_play import SelfPlayConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser.

    Exposed separately so the management console can introspect the
    available options without executing a training run.
    """
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
        "--hidden-channels",
        type=int,
        default=64,
        help="channels in the residual towers (default: 64)",
    )
    parser.add_argument(
        "--res-blocks",
        type=int,
        default=4,
        help=(
            "residual blocks in the representation tower; dynamics uses "
            "half (default: 4)"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="relative weight of the value objective (default: 1.0)",
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=8,
        help=(
            "opening moves sampled at temperature 1 before greedy play "
            "(default: 8)"
        ),
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
        default="uniform",
        help="game sampling distribution (default: uniform)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="disable dihedral symmetry augmentation of replay samples",
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
        "--training-state",
        default="checkpoints/training-state.pt",
        help=(
            "where to save resumable training state "
            "(default: checkpoints/training-state.pt)"
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="resume from a saved training state file",
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
        "--run-name",
        default=None,
        help="metrics run name (default: UTC timestamp)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="also write TensorBoard event files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.backend == "tpu" and args.self_play_mode == "async":
        parser.error(
            "TPU/XLA currently requires --self-play-mode sync; "
            "process-based TPU actors are not implemented yet"
        )

    resumed_state = None
    start_iteration = 1
    if args.resume is not None:
        resumed_state = load_training_state(
            args.resume, device.torch_device
        )
        start_iteration = resumed_state.iteration + 1
        if (
            resumed_state.board_size != args.board_size
            or resumed_state.win_length != args.win_length
        ):
            print(
                "resume: using saved game configuration "
                f"{resumed_state.board_size}x{resumed_state.board_size} "
                f"win={resumed_state.win_length}"
            )
        args.board_size = resumed_state.board_size
        args.win_length = resumed_state.win_length

    env = GomokuEnv(
        board_size=args.board_size,
        win_length=args.win_length,
    )
    if resumed_state is not None:
        network = resumed_state.network
    else:
        network = MuZeroNetwork(
            board_size=args.board_size,
            hidden_channels=args.hidden_channels,
            num_blocks=args.res_blocks,
        ).to(device.torch_device)
    # The paper scales root Dirichlet noise as roughly 10/branching-factor
    # (chess 0.3, Go 0.03); derive it from the board instead of exposing a
    # knob that silently goes stale when the board size changes.
    dirichlet_alpha = 10.0 / env.action_space_size
    mcts = MCTS(
        network,
        MCTSConfig(
            num_simulations=args.simulations,
            dirichlet_alpha=dirichlet_alpha,
        ),
        seed=args.seed,
    )
    replay = ReplayBuffer(
        capacity=args.replay_capacity,
        action_space_size=env.action_space_size,
        seed=args.seed,
        sampling=args.replay_sampling,
        recency_half_life=200.0,
        augment_symmetries=not args.no_augment,
    )
    trainer = MuZeroTrainer(
        network,
        learning_rate=args.learning_rate,
        loss_weights=LossWeights(value=args.value_loss_weight),
    )
    if resumed_state is not None:
        trainer.optimizer.load_state_dict(resumed_state.optimizer_state)
        for game in resumed_state.games:
            replay.save_game(game)
        print(
            f"resumed from {args.resume}: iteration {resumed_state.iteration}"
            f", {len(resumed_state.games)} replay games"
        )
    pipeline_class = (
        AsyncMuZeroPipeline
        if args.self_play_mode == "async"
        else MuZeroPipeline
    )
    pipeline_kwargs = {}
    if args.self_play_mode == "async":
        pipeline_kwargs = {"actor_seed": args.seed + 10_000}
    pipeline = pipeline_class(
        env,
        mcts,
        replay,
        trainer,
        LearningConfig(
            games_per_iteration=args.games_per_iteration,
            training_steps_per_iteration=args.training_steps,
            batch_size=args.batch_size,
            num_unroll_steps=5,
            evaluation_interval=args.evaluation_interval,
            evaluation_games=args.evaluation_games,
        ),
        SelfPlayConfig(temperature_moves=args.temperature_moves),
        **pipeline_kwargs,
    )
    logger = RunLogger(
        runs_dir="runs",
        run_name=args.run_name,
        tensorboard=args.tensorboard,
    )
    logger.write_config(
        {
            **vars(args),
            "dirichlet_alpha": dirichlet_alpha,
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
        last_iteration = start_iteration + args.iterations - 1
        for iteration in range(start_iteration, last_iteration + 1):
            result = pipeline.run_iteration(
                iteration,
                progress_callback=show_progress,
            )
            message = (
                f"iteration={result.iteration} "
                f"games={result.games_generated}"
            )
            if result.latest_metrics is not None:
                metrics_line = result.latest_metrics
                message += (
                    f" loss={metrics_line['loss']:.3f} "
                    f"policy={metrics_line['policy_loss']:.3f} "
                    f"entropy={metrics_line['policy_target_entropy']:.3f} "
                    f"kl={metrics_line['policy_kl']:.3f} "
                    f"value={metrics_line['value_loss']:.3f} "
                    f"reward={metrics_line['reward_loss']:.3f}"
                )
            if result.evaluation is not None:
                evaluation = result.evaluation
                message += (
                    f" eval={evaluation.wins}W/"
                    f"{evaluation.draws}D/{evaluation.losses}L"
                )
            if result.heuristic_evaluation is not None:
                heuristic = result.heuristic_evaluation
                message += (
                    f" heval={heuristic.wins}W/"
                    f"{heuristic.draws}D/{heuristic.losses}L"
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
            if result.heuristic_evaluation is not None:
                metrics.update(
                    {
                        "heuristic_evaluation_wins": (
                            result.heuristic_evaluation.wins
                        ),
                        "heuristic_evaluation_draws": (
                            result.heuristic_evaluation.draws
                        ),
                        "heuristic_evaluation_losses": (
                            result.heuristic_evaluation.losses
                        ),
                        "heuristic_evaluation_score": (
                            result.heuristic_evaluation.score
                        ),
                    }
                )
            logger.record("iteration", result.iteration, metrics)
            save_checkpoint(network, args.checkpoint, env.win_length)
            save_training_state(
                args.training_state,
                network,
                trainer.optimizer,
                iteration,
                list(replay.games),
                env.win_length,
            )
            print(f"\r{message:<{progress_width}}")
    finally:
        if isinstance(pipeline, AsyncMuZeroPipeline):
            pipeline.stop()
        logger.close()
    print(f"checkpoint={args.checkpoint}")
    print(f"training_state={args.training_state}")
    print(f"run_dir={logger.run_dir}")


if __name__ == "__main__":
    main()
