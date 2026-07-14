"""Package a trained model into a standalone, installable play bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_muzero.delivery.packager import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="trained model to package (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--output",
        default="dist",
        help="directory receiving the bundle and zip (default: dist)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="bundle name suffix (default: the checkpoint file's stem)",
    )
    args = parser.parse_args()

    bundle, zip_path = build_bundle(
        args.checkpoint, args.output, args.name
    )
    size_mb = zip_path.stat().st_size / 1e6
    print(f"bundle={bundle}")
    print(f"zip={zip_path} ({size_mb:.1f} MB)")
    print("Send the zip to the target machine, then:")
    print(f"  unzip {zip_path.name}")
    print(f"  pip install ./{bundle.name}")
    print("  gomoku-play")
    print(f"Full instructions are in {bundle / 'README.md'}.")


if __name__ == "__main__":
    main()
