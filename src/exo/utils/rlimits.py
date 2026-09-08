import sys


def raise_nofile_limit(min_soft: int) -> None:
    """Raise the RLIMIT_NOFILE soft limit when the platform supports it.

    Windows does not expose POSIX resource limits; the process already inherits
    a large handle table, so this is a no-op there.
    """
    if sys.platform == "win32":
        return

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, min_soft), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
