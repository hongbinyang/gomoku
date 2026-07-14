"""Run the management console for training, playing, and run admin."""

from __future__ import annotations

import argparse

from gomoku_muzero.manager.server import serve
from gomoku_muzero.runtime.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1, local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory scanned for models (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps", "tpu"),
        default="auto",
        help="compute backend for interactive play (default: auto)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device.description}")
    try:
        serve(
            host=args.host,
            port=args.port,
            checkpoint_dir=args.checkpoint_dir,
            device=device.torch_device,
        )
    except OSError as error:
        raise SystemExit(
            f"cannot listen on {args.host}:{args.port} ({error}); "
            "is another server running? Choose a different --port."
        ) from error


if __name__ == "__main__":
    main()
