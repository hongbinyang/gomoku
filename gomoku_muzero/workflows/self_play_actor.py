"""Background self-play actor with frozen, versioned network snapshots."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter

import torch

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.search.mcts import MCTS, MCTSConfig
from gomoku_muzero.training.replay import GameHistory
from gomoku_muzero.workflows.self_play import (
    SelfPlayConfig,
    play_self_play_game,
)


class PublishedWeights:
    """Thread-safe immutable CPU snapshots published by the learner."""

    def __init__(self, network: MuZeroNetwork) -> None:
        self._lock = Lock()
        self._version = -1
        self._state_dict: dict[str, torch.Tensor] = {}
        self.publish(network)

    def publish(self, network: MuZeroNetwork) -> int:
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in network.state_dict().items()
        }
        with self._lock:
            self._version += 1
            self._state_dict = state_dict
            return self._version

    def newer_than(
        self, version: int
    ) -> tuple[int, dict[str, torch.Tensor]] | None:
        with self._lock:
            if self._version <= version:
                return None
            # The published dictionary and tensors are never mutated after
            # publication, so sharing this reference with the actor is safe.
            return self._version, self._state_dict

    @property
    def version(self) -> int:
        with self._lock:
            return self._version


class SelfPlayActor:
    """Generate games independently and place them on a bounded queue."""

    def __init__(
        self,
        board_size: int,
        win_length: int,
        hidden_channels: int,
        device: torch.device,
        weights: PublishedWeights,
        mcts_config: MCTSConfig,
        self_play_config: SelfPlayConfig,
        queue_size: int = 4,
        seed: int = 0,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.board_size = board_size
        self.win_length = win_length
        self.hidden_channels = hidden_channels
        self.device = device
        self.weights = weights
        self.mcts_config = mcts_config
        self.self_play_config = self_play_config
        self.seed = seed
        self.games: Queue[GameHistory] = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None
        self._games_generated = 0
        self._self_play_seconds = 0.0
        self._version = -1

    @property
    def games_generated(self) -> int:
        return self._games_generated

    @property
    def queue_size(self) -> int:
        return self.games.qsize()

    @property
    def network_version(self) -> int:
        return self._version

    @property
    def games_per_second(self) -> float:
        if self._self_play_seconds == 0:
            return 0.0
        return self._games_generated / self._self_play_seconds

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = Thread(
            target=self._run,
            name="muzero-self-play",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def get_game(self, timeout: float = 0.25) -> GameHistory:
        """Wait for a game while surfacing actor failures promptly."""
        while True:
            self._raise_if_failed()
            try:
                return self.games.get(timeout=timeout)
            except Empty:
                if self._thread is None or not self._thread.is_alive():
                    self._raise_if_failed()
                    raise RuntimeError("self-play actor stopped unexpectedly")

    def _run(self) -> None:
        try:
            env = GomokuEnv(self.board_size, self.win_length)
            network = MuZeroNetwork(
                self.board_size, self.hidden_channels
            ).to(self.device)
            mcts = MCTS(network, self.mcts_config, seed=self.seed)

            while not self._stop.is_set():
                update = self.weights.newer_than(self._version)
                if update is not None:
                    self._version, state_dict = update
                    network.load_state_dict(state_dict)
                    network.eval()

                started = perf_counter()
                game = play_self_play_game(
                    env, mcts, self.self_play_config
                )
                elapsed = perf_counter() - started
                game.network_version = self._version
                while not self._stop.is_set():
                    try:
                        self.games.put(game, timeout=0.25)
                        self._games_generated += 1
                        self._self_play_seconds += elapsed
                        break
                    except Full:
                        continue
        except BaseException as error:
            self._error = error
            self._stop.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("self-play actor failed") from self._error
