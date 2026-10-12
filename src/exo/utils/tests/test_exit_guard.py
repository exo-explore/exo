import subprocess
import sys
from pathlib import Path

_HANGS_AT_EXIT = """
import threading
from pathlib import Path

from exo.utils.exit_guard import exit_if_shutdown_hangs

# A non-daemon thread that never finishes: the interpreter waits for it at exit.
threading.Thread(target=threading.Event().wait).start()
if {guarded}:
    exit_if_shutdown_hangs(Path({dump!r}), grace_seconds=1)
"""


def _run(tmp_path: Path, guarded: bool, timeout: float) -> int | None:
    script = _HANGS_AT_EXIT.format(guarded=guarded, dump=str(tmp_path / "exo.log"))
    try:
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        return None


def test_a_process_that_would_hang_at_exit_exits_and_logs_why(tmp_path: Path):
    # Without the guard the process never exits...
    assert _run(tmp_path, guarded=False, timeout=3) is None
    # ...with it, it exits after the grace period and records every thread's stack.
    assert _run(tmp_path, guarded=True, timeout=30) == 1
    dump = (tmp_path / "exo.log").read_text()
    assert "most recent call first" in dump
    assert "threading.py" in dump
