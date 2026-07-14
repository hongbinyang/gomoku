"""Optional external tools: TensorBoard process and static plots.

Both integrate the existing workflows without new required
dependencies: TensorBoard and Matplotlib come from the project's
``metrics`` extra, and the console degrades to a clear message when
they are not installed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Any


class TensorBoardManager:
    """Start and stop one TensorBoard subprocess over ``runs/``."""

    def __init__(self, workdir: str | Path = ".", port: int = 6006) -> None:
        self.workdir = Path(workdir)
        self.port = port
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def available(self) -> bool:
        return importlib.util.find_spec("tensorboard") is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = (
                self._process is not None
                and self._process.poll() is None
            )
            return {
                "available": self.available(),
                "running": running,
                "url": f"http://127.0.0.1:{self.port}" if running else None,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if not self.available():
                raise ValueError(
                    "TensorBoard is not installed; install the project's "
                    "'metrics' extra: pip install -e '.[metrics]'"
                )
            if self._process is not None and self._process.poll() is None:
                return self.status()
            import socket

            with socket.socket() as probe:
                probe.settimeout(0.3)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    raise ValueError(
                        f"port {self.port} is already in use — likely a "
                        "TensorBoard started manually. Stop that one, or "
                        f"open http://127.0.0.1:{self.port} directly."
                    )
            log_dir = self.workdir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "tensorboard.log"
            with log_path.open("wb") as log_file:
                self._process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        str(self.workdir / "runs"),
                        "--port",
                        str(self.port),
                        "--host",
                        "127.0.0.1",
                    ],
                    cwd=self.workdir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            # TensorBoard that fails at startup dies within a few
            # seconds; surface that instead of silently showing
            # "stopped" again. Return as soon as it starts serving.
            import time

            deadline = time.time() + 8.0
            while time.time() < deadline:
                if self._process.poll() is not None:
                    tail = ""
                    try:
                        tail = log_path.read_text(errors="replace")[-400:]
                    except OSError:
                        pass
                    raise ValueError(
                        "TensorBoard exited immediately (is another "
                        f"TensorBoard already using port {self.port}?). "
                        f"Log tail: {tail.strip()}"
                    )
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                        break
                time.sleep(0.25)
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
        return self.status()


def generate_plots(workdir: str | Path, run_name: str) -> list[str]:
    """Create the standard static charts for one run; returns file names."""
    if importlib.util.find_spec("matplotlib") is None:
        raise ValueError(
            "Matplotlib is not installed; install the project's "
            "'metrics' extra: pip install -e '.[metrics]'"
        )
    if "/" in run_name or "\\" in run_name or run_name.startswith("."):
        raise ValueError("invalid run name")
    # Force the headless backend before pyplot is imported: this runs in
    # a server worker thread, and GUI backends (macOS in particular)
    # crash the whole process outside the main thread. We only write
    # PNG files, so Agg is always sufficient.
    import matplotlib

    matplotlib.use("Agg", force=True)
    from gomoku_muzero.cli.plot import create_plots

    run_dir = Path(workdir) / "runs" / run_name
    return [path.name for path in create_plots(run_dir)]
