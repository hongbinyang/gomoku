"""Actor-learner pipeline that overlaps self-play and optimization."""

from __future__ import annotations

from collections.abc import Callable

from gomoku_muzero.training.pipeline import MuZeroPipeline
from gomoku_muzero.workflows.self_play_actor import (
    PublishedWeights,
    SelfPlayActor,
)


class AsyncMuZeroPipeline(MuZeroPipeline):
    """Consume games from an independent background self-play actor."""

    def __init__(
        self,
        *args: object,
        self_play_queue_size: int = 4,
        actor_seed: int = 10_000,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        network = self.trainer.network
        self.published_weights = PublishedWeights(network)
        device = next(network.parameters()).device
        self.actor = SelfPlayActor(
            board_size=self.env.board_size,
            win_length=self.env.win_length,
            hidden_channels=network.hidden_channels,
            device=device,
            weights=self.published_weights,
            mcts_config=self.mcts.config,
            self_play_config=self.self_play_config,
            queue_size=self_play_queue_size,
            seed=actor_seed,
        )
        self._policy_lag_mean = 0.0
        self._policy_lag_max = 0.0

    def start(self) -> None:
        self.actor.start()

    def stop(self) -> None:
        self.actor.stop()

    def _generate_games(
        self,
        iteration: int,
        report: Callable[[str], None],
    ) -> tuple[int, int]:
        moves_generated = 0
        policy_lags: list[int] = []
        for game_index in range(1, self.config.games_per_iteration + 1):
            report(
                f"iteration {iteration} | waiting for self-play game "
                f"{game_index}/{self.config.games_per_iteration} | "
                f"queue {self.actor.queue_size}"
            )
            game = self.actor.get_game()
            self.replay_buffer.save_game(game)
            self._note_game(game)
            moves_generated += game.num_moves
            if game.network_version is not None:
                policy_lags.append(
                    self.published_weights.version - game.network_version
                )
        if policy_lags:
            self._policy_lag_mean = sum(policy_lags) / len(policy_lags)
            self._policy_lag_max = float(max(policy_lags))
        return self.config.games_per_iteration, moves_generated

    def _extra_operational_metrics(self) -> dict[str, float]:
        return {
            **super()._extra_operational_metrics(),
            "actor_queue_size": float(self.actor.queue_size),
            "actor_games_generated": float(self.actor.games_generated),
            "actor_network_version": float(self.actor.network_version),
            "published_network_version": float(
                self.published_weights.version
            ),
            "policy_lag_mean": self._policy_lag_mean,
            "policy_lag_max": self._policy_lag_max,
            "games_per_second": self.actor.games_per_second,
        }

    def _train(
        self,
        iteration: int,
        report: Callable[[str], None],
    ) -> dict[str, float] | None:
        metrics = super()._train(iteration, report)
        self.published_weights.publish(self.trainer.network)
        return metrics

    def run(
        self,
        num_iterations: int,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.start()
        try:
            return super().run(num_iterations, progress_callback)
        finally:
            self.stop()
