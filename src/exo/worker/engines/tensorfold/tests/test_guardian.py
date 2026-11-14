import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import psutil
import pytest

_CHILD = """import json, os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]))
if sys.argv[2] == 'exit':
    os._exit(7)
while True:
    time.sleep(0.1)
"""


def alive(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("native_action", ["stay", "exit"])
def test_guardian_reaps_owned_group_on_lifeline_close_or_native_exit(
    tmp_path: Path, native_action: str
) -> None:
    child_script = tmp_path / "owned_child.py"
    child_script.write_text(_CHILD)
    record = tmp_path / "owned-processes.json"
    socket_directory = tmp_path / "private-socket"
    socket_directory.mkdir()
    socket_path = socket_directory / "engine.sock"
    socket_path.touch()
    guardian_script = Path(__file__).resolve().parents[1] / "guardian.py"
    owned: list[psutil.Process] = []
    guardian = subprocess.Popen(
        [
            sys.executable,
            str(guardian_script),
            "0.1",
            str(socket_path),
            sys.executable,
            str(child_script),
            str(record),
            native_action,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert record.exists(), "The owned native test process never started"
        for identifier in cast(list[int], json.loads(record.read_text())):
            # The guardian may already have reaped the exited group leader.
            with contextlib.suppress(psutil.NoSuchProcess):
                owned.append(psutil.Process(identifier))
        assert guardian.stdin is not None
        guardian.stdin.close()
        guardian.wait(timeout=5)
        deadline = time.monotonic() + 3
        while any(alive(process) for process in owned) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(alive(process) for process in owned), (
            "A process owned by the native runtime outlived its guardian"
        )
        assert not socket_path.exists()
        assert not socket_directory.exists()
    finally:
        if guardian.poll() is None:
            guardian.kill()
            guardian.wait(timeout=5)
        for process in owned:
            if alive(process):
                process.kill()
        if guardian.stderr is not None:
            guardian.stderr.close()
