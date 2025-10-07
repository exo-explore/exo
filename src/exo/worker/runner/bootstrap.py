import asyncio
import faulthandler
import os
import sys
from multiprocessing.connection import Connection


def _redirect_stderr_to_file(path: str) -> None:
    # Replace fd 2 (stderr) with a file descriptor pointing to `path`
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.close(fd)
    # Rebind sys.stderr so Python's own writes go to the new fd as well (line-buffered)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)


def entrypoint(raw_conn: Connection, err_path: str) -> None:
    """
    Minimal entrypoint for the spawned child process.

    It redirects fd=2 (stderr) to a pipe provided by the parent, *then* imports
    the heavy runner module so that any C/C++ or MLX logs/crashes land in that pipe.
    """
    _redirect_stderr_to_file(err_path)
    faulthandler.enable(file=sys.stderr, all_threads=True)

    # Import the heavy runner only after stderr is redirected
    from exo.worker.runner.runner import main

    asyncio.run(main(raw_conn))
