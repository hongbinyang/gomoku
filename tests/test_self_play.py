import numpy as np

from gomoku_muzero.game.env import GomokuEnv
from gomoku_muzero.search.mcts import MCTS, MCTSConfig
from gomoku_muzero.model.networks import MuZeroNetwork
from gomoku_muzero.workflows.self_play import SelfPlayConfig, play_self_play_game


def test_self_play_produces_valid_complete_game() -> None:
    env = GomokuEnv(board_size=2, win_length=2)
    network = MuZeroNetwork(board_size=2, hidden_channels=4)
    mcts = MCTS(network, MCTSConfig(num_simulations=2), seed=0)

    game = play_self_play_game(
        env,
        mcts,
        SelfPlayConfig(
            temperature=0,
            temperature_moves=0,
            add_exploration_noise=False,
        ),
    )

    assert 3 <= game.num_moves <= 4
    assert len(game.observations) == game.num_moves + 1
    assert len(set(game.actions)) == game.num_moves
    assert np.all(game.policies[-1] == 0)
    assert game.values[-1] == 0
    assert game.rewards[-1] in (0.0, 1.0)
    assert game.root_values is not None
    assert len(game.root_values) == game.num_moves + 1
    assert game.root_values[-1] == 0.0
    assert game.to_play[-1] == env.current_player
    for index in range(game.num_moves):
        assert game.values[index] == env.winner * game.to_play[index]
