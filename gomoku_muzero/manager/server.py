"""The management console server: a layer over the existing modules.

Routes under ``/manager/api/*`` control training; the console page is
served at ``/`` and the unchanged game page at ``/play`` — the play
handler, session, and API endpoints are inherited untouched from
``gomoku_muzero.web``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gomoku_muzero.manager.training import TrainingManager, training_options
from gomoku_muzero.web.server import GomokuWebServer, _Handler

_STATIC_DIR = Path(__file__).parent / "static"


def list_training_states(directory: str | Path) -> list[str]:
    """Resumable training-state files, newest first."""
    root = Path(directory)
    if not root.is_dir():
        return []
    states = [p for p in root.glob("*.pt") if "state" in p.stem]
    states.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in states]


class _ManagerHandler(_Handler):
    server: "ManagerServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_console()
        elif self.path == "/play":
            self._send_page()  # the unchanged game page
        elif self.path == "/manager/api/options":
            self._send_json(200, {"options": training_options()})
        elif self.path == "/manager/api/status":
            self._send_json(200, self.server.training.status())
        elif self.path == "/manager/api/states":
            self._send_json(
                200,
                {
                    "states": list_training_states(
                        self.server.checkpoint_dir
                    )
                },
            )
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/manager/api/"):
            super().do_POST()
            return
        try:
            if self.path == "/manager/api/start":
                payload = self._read_json()
                self._send_json(
                    200,
                    self.server.training.start(
                        payload.get("options", {})
                    ),
                )
            elif self.path == "/manager/api/stop":
                self._send_json(200, self.server.training.stop())
            else:
                self._send_json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as error:
            self._send_json(400, {"error": str(error) or "bad request"})

    def _send_console(self) -> None:
        body = (_STATIC_DIR / "console.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ManagerServer(GomokuWebServer):
    """The play server plus a training manager, under one console."""

    def __init__(
        self,
        address: tuple[str, int],
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
        workdir: str | Path = ".",
    ) -> None:
        super().__init__(
            address, checkpoint_dir, device, handler=_ManagerHandler
        )
        self.training = TrainingManager(workdir)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device = "cpu",
) -> None:
    """Run the management console until interrupted."""
    server = ManagerServer((host, port), checkpoint_dir, device)
    print(
        "Management console at "
        f"http://{host}:{server.server_address[1]}"
    )
    print(f"Checkpoints from: {Path(checkpoint_dir).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
