"""The outer self-play, replay, training, and evaluation loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from time import perf_counter

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.workflows.evaluate import (
    EvaluationResult,
    evaluate_against_heuristic,
    evaluate_against_random,
)
from gomoku_muzero.search.mcts import MCTS
from gomoku_muzero.training.replay import GameHistory, ReplayBuffer
from gomoku_muzero.workflows.self_play import SelfPlayConfig, play_self_play_game
from gomoku_muzero.training.trainer import MuZeroTrainer


@dataclass(frozen=True)
class LearningConfig:
    games_per_iteration: int = 2
    training_steps_per_iteration: int = 10
    batch_size: int = 32
    num_unroll_steps: int = 5
    evaluation_interval: int = 5
    evaluation_games: int = 20


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    games_generated: int
    latest_metrics: dict[str, float] | None
    evaluation: EvaluationResult | None
    heuristic_evaluation: EvaluationResult | None
    operational_metrics: dict[str, float]


class MuZeroPipeline:
    """A deliberately synchronous and readable MuZero learning loop."""

    def __init__(
        self,
        env: GomokuEnv,
        mcts: MCTS,
        replay_buffer: ReplayBuffer,
        trainer: MuZeroTrainer,
        learning_config: LearningConfig | None = None,
        self_play_config: SelfPlayConfig | None = None,
    ) -> None:
        self.env = env
        self.mcts = mcts
        self.replay_buffer = replay_buffer
        self.trainer = trainer
        self.config = learning_config or LearningConfig()
        self.self_play_config = self_play_config or SelfPlayConfig()
        self._calibration_errors: list[float] = []
        if self.config.games_per_iteration < 1:
            raise ValueError("games_per_iteration must be positive")
        if self.config.training_steps_per_iteration < 0:
            raise ValueError(
                "training_steps_per_iteration must be non-negative"
            )
        if self.config.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")

    def run_iteration(
        self,
        iteration: int,
        progress_callback: Callable[[str], None] | None = None,
    ) -> IterationResult:
        """Generate games, train, and optionally evaluate once."""
        report = progress_callback or (lambda _: None)
        self._calibration_errors = []
        iteration_started = perf_counter()
        phase_started = perf_counter()
        games_generated, moves_generated = self._generate_games(
            iteration, report
        )
        self_play_seconds = perf_counter() - phase_started

        phase_started = perf_counter()
        averaged_metrics = self._train(iteration, report)
        training_seconds = perf_counter() - phase_started

        phase_started = perf_counter()
        evaluation, heuristic_evaluation = self._evaluate(iteration, report)
        evaluation_seconds = perf_counter() - phase_started
        iteration_seconds = perf_counter() - iteration_started
        operational_metrics = {
            "iteration_seconds": iteration_seconds,
            "self_play_seconds": self_play_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "games_per_second": (
                games_generated / self_play_seconds
                if self_play_seconds > 0
                else 0.0
            ),
            "training_steps_per_second": (
                self.config.training_steps_per_iteration / training_seconds
                if training_seconds > 0
                else 0.0
            ),
            "moves_generated": float(moves_generated),
            "replay_games": float(len(self.replay_buffer)),
        }
        operational_metrics.update(self._extra_operational_metrics())
        return IterationResult(
            iteration,
            games_generated,
            averaged_metrics,
            evaluation,
            heuristic_evaluation,
            operational_metrics,
        )

    def _generate_games(
        self,
        iteration: int,
        report: Callable[[str], None],
    ) -> tuple[int, int]:
        """Synchronously generate this iteration's replay games."""
        moves_generated = 0
        for game_index in range(1, self.config.games_per_iteration + 1):
            game = play_self_play_game(
                self.env,
                self.mcts,
                self.self_play_config,
                progress_callback=lambda move, completed, total: report(
                    f"iteration {iteration} | self-play game "
                    f"{game_index}/{self.config.games_per_iteration} | "
                    f"move {move} | MCTS {completed}/{total}"
                ),
            )
            self.replay_buffer.save_game(game)
            self._note_game(game)
            moves_generated += game.num_moves
        return self.config.games_per_iteration, moves_generated

    def _note_game(self, game: GameHistory) -> None:
        """Track how well search root values predicted final outcomes."""
        if game.root_values is None or game.num_moves == 0:
            return
        error = fmean(
            abs(game.root_values[index] - game.values[index])
            for index in range(game.num_moves)
        )
        self._calibration_errors.append(error)

    def _extra_operational_metrics(self) -> dict[str, float]:
        metrics = self.replay_buffer.sampling_metrics()
        if self._calibration_errors:
            metrics["value_calibration_mae"] = fmean(
                self._calibration_errors
            )
        return metrics

    def _train(
        self,
        iteration: int,
        report: Callable[[str], None],
    ) -> dict[str, float] | None:
        """Apply and average this iteration's optimizer updates."""
        step_metrics: list[dict[str, float]] = []
        for step in range(1, self.config.training_steps_per_iteration + 1):
            report(
                f"iteration {iteration} | training step "
                f"{step}/{self.config.training_steps_per_iteration}"
            )
            batch = self.replay_buffer.sample_batch(
                self.config.batch_size,
                self.config.num_unroll_steps,
            )
            step_metrics.append(self.trainer.train_step(batch))

        averaged_metrics = None
        if step_metrics:
            averaged_metrics = {
                name: fmean(metrics[name] for metrics in step_metrics)
                for name in step_metrics[0]
            }
        return averaged_metrics

    def _evaluate(
        self,
        iteration: int,
        report: Callable[[str], None],
    ) -> tuple[EvaluationResult | None, EvaluationResult | None]:
        """Run periodic evaluation against both fixed baselines.

        Random is a saturating smoke test; the win-or-block heuristic is
        the informative opponent because it punishes threat-blind play.
        """
        if iteration % self.config.evaluation_interval != 0:
            return None, None
        evaluation = evaluate_against_random(
            self.env,
            self.mcts,
            num_games=self.config.evaluation_games,
            seed=iteration,
            progress_callback=(
                lambda game, total_games, completed, total: report(
                    f"iteration {iteration} | evaluation (random) game "
                    f"{game}/{total_games} | MCTS {completed}/{total}"
                )
            ),
        )
        heuristic_evaluation = evaluate_against_heuristic(
            self.env,
            self.mcts,
            num_games=self.config.evaluation_games,
            seed=iteration + 5_000,
            progress_callback=(
                lambda game, total_games, completed, total: report(
                    f"iteration {iteration} | evaluation (heuristic) game "
                    f"{game}/{total_games} | MCTS {completed}/{total}"
                )
            ),
        )
        return evaluation, heuristic_evaluation

    def run(
        self,
        num_iterations: int,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[IterationResult]:
        """Run a requested number of complete learning iterations."""
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        return [
            self.run_iteration(iteration, progress_callback)
            for iteration in range(1, num_iterations + 1)
        ]
