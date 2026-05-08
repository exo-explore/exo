from __future__ import annotations

from typing import Final

from exo_pyo3_bindings import Pidfile, PidfileError

from exo.shared.constants import EXO_PID_FILE

_PIDFILE_MODE: Final = 0o600


class PidfileLockError(RuntimeError):
    pass


def acquire_exo_pidfile() -> Pidfile:
    path = EXO_PID_FILE
    try:
        pidfile = Pidfile(path, _PIDFILE_MODE)
        pidfile.write()
    except (OSError, PidfileError) as exception:
        raise PidfileLockError(
            f"Failed to acquire EXO pidfile at {path}: {exception}"
        ) from exception

    return pidfile
