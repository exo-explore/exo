"""Rust's networking threads never wait for Python's GIL while Python waits for them.

Each test runs a real NetworkingHandle in a child process with a time limit: a deadlock shows
as a child that doesn't finish, which is killed and fails the test.
"""

import subprocess
import sys
import textwrap

import pytest

# The child couldn't start networking here (no free port, no network)
NO_NETWORKING = 77

_START = textwrap.dedent(
    """
    import asyncio, ctypes, logging, os, random, sys, threading, time
    import exo_rs

    # Rust's records reach Python's logging, at the levels exo uses
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    def handle(name):
        port = 20000 + random.randrange(30000)
        zid = os.urandom(16).hex().lstrip("0")
        try:
            return exo_rs.NetworkingHandle.new(zid, f"gil-{os.getpid()}-{name}", port, port + 1)
        except Exception as error:
            print(f"no networking: {error}", flush=True)
            sys.exit(77)
    """
)

# Reading the membership in a loop while the networking starts: the membership's thread logs
# this node's own token at once, and used to do so holding the lock that members() waits for
# with the GIL held, while the log call waited for the GIL.
_READ_WHILE_STARTING = _START + textwrap.dedent(
    """
    async def one(i):
        h = handle(i)
        stop = threading.Event()
        def read():
            while not stop.is_set():
                h.members()
        threading.Thread(target=read, daemon=True).start()
        async def drain():
            while True:
                await h.recv()
        task = asyncio.create_task(drain())
        await asyncio.sleep(1.0)
        stop.set()
        task.cancel()
        print(f"started {i}", flush=True)

    async def main():
        for i in range(3):
            await one(i)

    asyncio.run(main())
    print("done", flush=True)
    """
)

# Python holds the GIL for 7 s (a long call into C) while the networking starts and logs.
# The membership's timer must keep time meanwhile: had its thread waited for the GIL to log,
# its next tick would be 7 s late and this node would count itself as stalled (Resuming).
_HOLD_THE_GIL = _START + textwrap.dedent(
    """
    async def main():
        h = handle("hold")
        first = asyncio.ensure_future(h.recv())
        # libc's sleep through PyDLL keeps the GIL for its whole length
        ctypes.PyDLL(None).sleep(7)
        await first
        await asyncio.sleep(1.5)
        own = h.members().members[0]
        print(f"own state {own.state!r}", flush=True)

    asyncio.run(main())
    """
)


def _run(script: str, timeout: float) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        output = expired.stdout or ""
        raise AssertionError(
            f"the process hung (killed after {timeout:.0f} s); it printed: {output!r}"
        ) from None
    if result.returncode == NO_NETWORKING:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_reading_the_membership_while_rust_logs_never_hangs() -> None:
    output = _run(_READ_WHILE_STARTING, timeout=30)
    assert output.splitlines()[-1] == "done"


def test_rust_keeps_time_while_python_holds_the_gil() -> None:
    output = _run(_HOLD_THE_GIL, timeout=30)
    assert "own state MemberState.Alive" in output, output
