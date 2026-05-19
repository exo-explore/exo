import os
import sys

_STDIN_FD = 0
_STDOUT_FD = 1
_STDERR_FD = 2


def detach_stdio_to_devnull() -> None:
    """Redirect process stdio file descriptors to /dev/null."""

    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()

    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    stdout_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # dup2 closes the target fd first, but leaves the source fd open.
        os.dup2(stdin_fd, _STDIN_FD)
        os.dup2(stdout_fd, _STDOUT_FD)
        os.dup2(stderr_fd, _STDERR_FD)
    finally:
        for fd in (stdin_fd, stdout_fd, stderr_fd):
            if fd not in (_STDIN_FD, _STDOUT_FD, _STDERR_FD):
                os.close(fd)
