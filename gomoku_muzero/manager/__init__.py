"""Management console: training lifecycle, play, and run administration."""

from gomoku_muzero.manager.server import ManagerServer, serve
from gomoku_muzero.manager.training import TrainingManager, training_options

__all__ = [
    "ManagerServer",
    "TrainingManager",
    "serve",
    "training_options",
]
