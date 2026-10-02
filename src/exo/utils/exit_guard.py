import faulthandler
from pathlib import Path

EXIT_GRACE_SECONDS = 30.0


def exit_if_shutdown_hangs(
    dump_to: Path, grace_seconds: float = EXIT_GRACE_SECONDS
) -> None:
    """Make sure a process that has finished shutting down actually exits.

    After exo stops, the interpreter still joins every non-daemon thread and runs exit handlers.
    If one of those never returns, the process stays alive without doing anything, and whatever
    would restart a node that has stopped (launchd, the app) never gets the chance. If this
    process is still running `grace_seconds` from now, every thread's stack is appended to
    `dump_to`, so it's clear what held it up, and the process exits with status 1.

    faulthandler does the waiting on a thread of its own that doesn't need the GIL, so this
    works however far the interpreter has got in shutting down.
    """
    dump_to.parent.mkdir(parents=True, exist_ok=True)
    # faulthandler keeps a reference to the file, and writes to it only if the timer fires.
    dump_file = dump_to.open("a")  # noqa: SIM115
    faulthandler.dump_traceback_later(grace_seconds, exit=True, file=dump_file)
