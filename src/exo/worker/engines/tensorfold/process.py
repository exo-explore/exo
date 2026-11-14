import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import httpx

from exo.worker.engines.tensorfold.capabilities import Health, LoadedModel, ModelList
from exo.worker.engines.tensorfold.config import (
    NativeSettings,
    validate_single_rank_environment,
)


class NativeProcess:
    def __init__(self, settings: NativeSettings):
        self.settings = settings
        self._directory = tempfile.TemporaryDirectory(prefix="exo-tf-")
        self.socket_path = Path(self._directory.name) / "engine.sock"
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._diagnostics: deque[bytes] = deque(maxlen=32)
        self.health: Health | None = None

    def start(
        self, model_path: Path, model_id: str, context_length: int, parallel: int | None
    ) -> LoadedModel:
        command = [
            str(self.settings.binary),
            "serve",
            str(model_path),
            "--name",
            model_id,
            "--unix-socket",
            str(self.socket_path),
        ]
        if parallel is not None:
            command.extend(["--parallel", str(parallel)])
        if context_length > 0:
            command.extend(["--context", str(context_length)])
        guardian = Path(__file__).with_name("guardian.py")
        try:
            validate_single_rank_environment(os.environ)
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    str(guardian),
                    str(self.settings.shutdown_timeout),
                    str(self.socket_path),
                    *command,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._reader = threading.Thread(target=self._read_diagnostics, daemon=True)
            self._reader.start()
            return self._wait_ready(model_id)
        except BaseException:
            self.close()
            raise

    def _read_diagnostics(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        with process.stderr:
            while chunk := process.stderr.read(1024):
                self._diagnostics.append(chunk)

    def check_alive(self) -> None:
        process = self._process
        if process is None:
            raise RuntimeError("TensorFold has not been started")
        if (code := process.poll()) is not None:
            detail = (
                b"".join(self._diagnostics)[-4096:]
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise RuntimeError(f"TensorFold exited with {code}: {detail}")

    def _wait_ready(self, model_id: str) -> LoadedModel:
        deadline = time.monotonic() + self.settings.startup_timeout
        transport = httpx.HTTPTransport(uds=str(self.socket_path))
        with httpx.Client(
            transport=transport,
            base_url="http://localhost",
            timeout=1.0,
            trust_env=False,
        ) as client:
            while time.monotonic() < deadline:
                self.check_alive()
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        health = Health.model_validate_json(response.content)
                        if health.status == "ok" and not health.warming:
                            models = client.get("/v1/models")
                            models.raise_for_status()
                            loaded = ModelList.model_validate_json(models.content)
                            matches = [
                                model for model in loaded.data if model.id == model_id
                            ]
                            if (
                                len(matches) != 1
                                or not matches[0].capabilities.text_generation
                            ):
                                raise ValueError(
                                    "TensorFold did not load the requested text model"
                                )
                            if health.model != model_id or health.backend not in {
                                "metal",
                                "cuda",
                            }:
                                raise ValueError(
                                    "TensorFold readiness identity does not match its launch"
                                )
                            self.health = health
                            return matches[0]
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            "TensorFold did not become ready before its startup deadline"
        )

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=self.settings.shutdown_timeout + 3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=self.settings.shutdown_timeout + 3.0)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        self._directory.cleanup()
