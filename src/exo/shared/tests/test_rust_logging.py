"""Rust's log records reach Python's logging through a queue: a Rust thread never waits for the
GIL to log. What a queue risks is covered here: records dropped while Python was busy are
counted (and it says when), records still queued at exit get out, and a panic reaches the log
file at once. Each test runs exo_rs in a child process."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# The child couldn't start networking here (no free port, no network)
NO_NETWORKING = 77

# Every NetworkingHandle.forget() of a valid id makes the membership log one record
_RECORDS = textwrap.dedent(
    """
    import asyncio, ctypes, logging, os, random, sys
    import exo_rs

    def handle():
        port = 20000 + random.randrange(30000)
        zid = os.urandom(16).hex().lstrip("0")
        try:
            return exo_rs.NetworkingHandle.new(zid, f"log-{os.getpid()}", port, port + 1)
        except Exception as error:
            print(f"no networking: {error}", flush=True)
            sys.exit(77)

    async def log_records(count):
        h = handle()
        first = asyncio.ensure_future(h.recv())
        await asyncio.sleep(1.0)
        for i in range(count):
            h.forget(f"{i + 1:x}")
        # Python holds the GIL (a long call into C) while the records are made
        ctypes.PyDLL(None).sleep(3)
        return h
    """
)


def _run(script: str, *args: str, timeout: float = 60) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == NO_NETWORKING:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.splitlines()


def test_records_dropped_while_python_was_busy_are_counted_after_the_rest() -> None:
    script = _RECORDS + textwrap.dedent(
        """
        got = []
        class Collect(logging.Handler):
            def emit(self, record):
                got.append(record.getMessage())
        logging.basicConfig(handlers=[Collect()], level=logging.INFO)

        async def main():
            await log_records(20_000)
            # A full queue can take longer than one flush's bound to pass on
            while not exo_rs.flush_logs():
                pass
        asyncio.run(main())
        removed = [int(m.split()[-1], 16) for m in got if m.startswith("the coordinator removed")]
        notices = [m for m in got if "were dropped" in m]
        print(len(removed))
        print(removed == sorted(removed))
        print(notices[-1] if notices else "no notice")
        print(got[-1])
        """
    )
    delivered, in_order, notice, last = _run(script)[-4:]
    dropped = int(notice.split()[0])
    # Every record is delivered, in order, or counted
    assert int(delivered) + dropped == 20_000
    assert dropped > 0, "the queue never filled"
    assert in_order == "True"
    # The notice says when, and comes after the records queued before the drops (all older)
    assert " were dropped between " in notice
    assert last == notice


def test_records_still_queued_at_exit_reach_the_log(tmp_path: Path) -> None:
    log = tmp_path / "rust.log"
    script = _RECORDS + textwrap.dedent(
        """
        logging.basicConfig(filename=sys.argv[1], level=logging.INFO)

        async def main():
            await log_records(3_000)
        asyncio.run(main())
        # Python exits now, with records still queued
        """
    )
    _run(script, str(log))
    written = log.read_text().count("the coordinator removed")
    assert written == 3_000


def test_a_rust_panic_is_written_into_the_log_file_at_once(tmp_path: Path) -> None:
    log = tmp_path / "exo.log"
    script = textwrap.dedent(
        """
        import os, sys
        import exo_rs
        exo_rs.write_panics_to(sys.argv[1])
        try:
            # A port of 0 isn't supported yet: a panic in Rust
            exo_rs.NetworkingHandle.new(os.urandom(16).hex().lstrip("0"), "panic", 0, 1)
        except BaseException as error:
            print(type(error).__name__)
        """
    )
    assert _run(script, str(log)) == ["PanicException"]
    written = log.read_text()
    assert "| CRITICAL | exo_rs:panic ] Rust panicked" in written
    assert "cannot listen on port 0 yet" in written
