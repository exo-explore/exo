import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType


def signal_group(process_id: int, number: int) -> bool:
    try:
        os.killpg(process_id, number)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS refuses to signal a group that holds only zombies; nothing in it can run.
        return False
    return True


def stop_group(child: subprocess.Popen[bytes], grace: float) -> None:
    if not signal_group(child.pid, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        child.poll()
        if not signal_group(child.pid, 0):
            return
        time.sleep(0.02)
    signal_group(child.pid, signal.SIGKILL)


def main(arguments: list[str]) -> int:
    if len(arguments) < 3:
        return 2
    grace = float(arguments[0])
    socket_path = Path(arguments[1])
    stopped = False

    def stop(_number: int, _frame: FrameType | None) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = subprocess.Popen(
        arguments[2:],
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
    )
    try:
        while not stopped and child.poll() is None:
            readable, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
            if readable and not os.read(sys.stdin.fileno(), 1):
                break
        stop_group(child, grace)
        return child.wait()
    finally:
        signal_group(child.pid, signal.SIGKILL)
        child.wait()
        socket_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            socket_path.parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
