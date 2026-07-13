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

from gomoku_muzero.manager.metrics import list_runs, metric_series
from gomoku_muzero.manager.storage import (
    delete_item,
    list_storage,
    model_info,
)
from gomoku_muzero.manager.tools import TensorBoardManager, generate_plots
from gomoku_muzero.manager.training import (
    TrainingManager,
    resume_config,
    training_options,
)
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
        try:
            self._route_get()
        except (ValueError, FileNotFoundError) as error:
            self._send_json(400, {"error": str(error)})

    def _route_get(self) -> None:
        path, _, query = self.path.partition("?")
        params = dict(
            part.split("=", 1)
            for part in query.split("&")
            if "=" in part
        )
        if path in ("/", "/index.html"):
            self._send_console()
        elif path == "/play":
            self._send_page()  # the unchanged game page
        elif path == "/manager/api/options":
            self._send_json(200, {"options": training_options()})
        elif path == "/manager/api/status":
            self._send_json(200, self.server.training.status())
        elif path == "/manager/api/states":
            self._send_json(
                200,
                {
                    "states": list_training_states(
                        self.server.checkpoint_dir
                    )
                },
            )
        elif path == "/manager/api/resume-config":
            self._send_json(
                200,
                resume_config(
                    params.get("state", ""), self.server.workdir
                ),
            )
        elif path == "/manager/api/runs":
            self._send_json(
                200, {"runs": list_runs(self.server.workdir)}
            )
        elif path == "/manager/api/metrics":
            names = [
                name
                for name in params.get("runs", "").split(",")
                if name
            ]
            self._send_json(
                200, metric_series(self.server.workdir, names)
            )
        elif path == "/manager/api/storage":
            self._send_json(
                200,
                list_storage(
                    self.server.workdir, self.server.checkpoint_dir
                ),
            )
        elif path == "/manager/api/model-info":
            self._send_json(
                200,
                model_info(
                    self.server.checkpoint_dir,
                    params.get("name", ""),
                    self.server.workdir,
                ),
            )
        elif path == "/manager/api/tensorboard":
            self._send_json(200, self.server.tensorboard.status())
        elif path.startswith("/manager/plots/"):
            self._send_plot(path)
        else:
            super().do_GET()

    def _send_plot(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) != 5 or not parts[4].endswith(".png"):
            self._send_json(404, {"error": "not found"})
            return
        run_name, file_name = parts[3], parts[4]
        for piece in (run_name, file_name):
            if "/" in piece or "\\" in piece or piece.startswith("."):
                self._send_json(400, {"error": "invalid path"})
                return
        target = (
            Path(self.server.workdir)
            / "runs"
            / run_name
            / "plots"
            / file_name
        )
        if not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            elif self.path == "/manager/api/tensorboard":
                payload = self._read_json()
                action = payload.get("action")
                if action == "start":
                    self._send_json(
                        200, self.server.tensorboard.start()
                    )
                elif action == "stop":
                    self._send_json(200, self.server.tensorboard.stop())
                else:
                    raise ValueError("action must be 'start' or 'stop'")
            elif self.path == "/manager/api/plots":
                payload = self._read_json()
                files = generate_plots(
                    self.server.workdir, str(payload.get("run", ""))
                )
                self._send_json(
                    200, {"run": payload.get("run"), "files": files}
                )
            elif self.path == "/manager/api/delete":
                payload = self._read_json()
                self._send_json(
                    200,
                    delete_item(
                        str(payload.get("kind", "")),
                        str(payload.get("name", "")),
                        str(payload.get("confirm", "")),
                        self.server.workdir,
                        self.server.checkpoint_dir,
                        self.server.training.active_run_artifacts(),
                    ),
                )
            else:
                self._send_json(404, {"error": "not found"})
        except FileNotFoundError as error:
            self._send_json(404, {"error": str(error)})
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
        self.workdir = Path(workdir)
        self.training = TrainingManager(workdir)
        self.tensorboard = TensorBoardManager(workdir)


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
