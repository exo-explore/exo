from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

import pytest

import exo.utils.pidfile as pidfile
from exo.utils.pidfile import acquire_exo_pidfile

_CHILD_ACQUIRE_PIDFILE_SCRIPT: Final = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from unittest.mock import patch

    import exo.utils.pidfile as pidfile
    from exo.utils.pidfile import PidfileLockError, acquire_exo_pidfile

    with patch.object(pidfile, "EXO_PID_FILE", Path(sys.argv[1])):
        try:
            handle = acquire_exo_pidfile()
        except PidfileLockError as exception:
            print(str(exception))
            raise SystemExit(73) from exception

        del handle
    """
)


def _use_pidfile_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(pidfile, "EXO_PID_FILE", path)


def _run_child_acquire_pidfile(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ACQUIRE_PIDFILE_SCRIPT, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_acquire_exo_pidfile_writes_current_pid_and_removes_on_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "exo.pid"
    _use_pidfile_path(monkeypatch, path)

    handle = acquire_exo_pidfile()
    assert path.read_text() == str(os.getpid())

    del handle
    gc.collect()

    assert not path.exists()


def test_acquire_exo_pidfile_rejects_second_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "exo.pid"
    _use_pidfile_path(monkeypatch, path)

    handle = acquire_exo_pidfile()
    try:
        blocked_child = _run_child_acquire_pidfile(path)
        assert blocked_child.returncode == 73
        assert "Failed to acquire EXO pidfile" in blocked_child.stdout
    finally:
        del handle
        gc.collect()

    unblocked_child = _run_child_acquire_pidfile(path)
    assert unblocked_child.returncode == 0
    assert unblocked_child.stdout == ""
