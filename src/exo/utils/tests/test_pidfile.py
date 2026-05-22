from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

from exo_pyo3_bindings.exo_pyo3_bindings import Pidfile

_CHILD_ACQUIRE_PIDFILE_SCRIPT: Final = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from exo_pyo3_bindings.exo_pyo3_bindings import Pidfile, PidfileError

    path = Path(sys.argv[1])
    try:
        handle = Pidfile(path, 0o0600)
        handle.write()
    except (OSError, PidfileError) as exception:
        print(f"Failed to acquire EXO pidfile at {path}: {exception}")
        raise SystemExit(73) from exception

    del handle
    """
)


def _run_child_acquire_pidfile(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ACQUIRE_PIDFILE_SCRIPT, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_acquire_exo_pidfile_writes_current_pid_and_removes_on_drop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exo.pid"

    handle = Pidfile(path, 0o0600)
    handle.write()
    assert path.read_text() == str(os.getpid())

    del handle
    gc.collect()

    assert not path.exists()


def test_acquire_exo_pidfile_rejects_second_process(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exo.pid"

    handle = Pidfile(path, 0o0600)
    handle.write()
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
