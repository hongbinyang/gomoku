# Development guide

## Setup

```bash
cd gomoku
conda activate gomoku
python -m pip install -e ".[dev]"
```

The editable installation makes source changes immediately available.

## Tests

Run everything:

```bash
python -m pytest
```

Run components independently:

```bash
python -m pytest tests/test_env.py
python -m pytest tests/test_networks.py
python -m pytest tests/test_mcts.py
python -m pytest tests/test_replay.py
python -m pytest tests/test_trainer.py
python -m pytest tests/test_self_play.py
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_play.py
python -m pytest tests/test_checkpoint.py
python -m pytest tests/test_device.py
python -m pytest tests/test_async_self_play.py
python -m pytest tests/test_metrics.py
```

Check both command interfaces:

```bash
python -m gomoku_muzero.train --help
python -m gomoku_muzero.play --help
python -m gomoku_muzero.serve --help
python -m gomoku_muzero.manager --help
python -m gomoku_muzero.plot --help
```

## Change checklist

1. Put the implementation in the responsibility package described in
   [architecture.md](architecture.md).
2. Keep tensor shapes and player perspectives explicit.
3. Add focused component tests before integration tests.
4. Run the complete suite.
5. Update the relevant guide when behavior or CLI options change.
