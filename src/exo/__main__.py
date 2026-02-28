from __future__ import annotations

import sys
from collections.abc import Sequence
from multiprocessing import freeze_support
from typing import Final

from exo.main import main

INLINE_CODE_FLAG: Final[str] = "-c"


def _maybe_run_inline_code(argv: Sequence[str]) -> bool:
    """
    Reproduce the bare minimum of Python's `-c` flag so multiprocessing
    helper processes (for example the resource tracker) can execute.
    """

    try:
        flag_index = argv.index(INLINE_CODE_FLAG)
    except ValueError:
        return False

    code_index = flag_index + 1
    if code_index >= len(argv):
        return False

    inline_code = argv[code_index]
    sys.argv = ["-c", *argv[code_index + 1 :]]
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(inline_code, namespace, namespace)
    return True


def _is_frozen() -> bool:
    """Check if running in a frozen environment (e.g., PyInstaller)."""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


def _maybe_run_runner(argv: Sequence[str]) -> bool:
    """
    Handle the --exo-runner flag for spawning runner subprocesses in frozen builds.
    """
    if "--exo-runner" not in argv:
        return False

    try:
        flag_index = argv.index("--exo-runner")
        args_index = flag_index + 1
        if args_index >= len(argv):
            return False

        # Set up the runner environment
        import os

        os.environ["EXO_FROZEN"] = "1"

        # Import and run the bootstrap entrypoint
        from exo.worker.runner.bootstrap import entrypoint

        # Inject the JSON args into sys.argv as expected by entrypoint
        sys.argv = [argv[0], argv[args_index]]
        entrypoint()
        return True
    except Exception as e:
        print(f"Failed to start runner: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    if _maybe_run_inline_code(sys.argv):
        sys.exit(0)

    # Check if we should run as a runner subprocess (frozen builds)
    if _maybe_run_runner(sys.argv):
        sys.exit(0)

    freeze_support()

    # Store frozen status for subprocess spawning
    if _is_frozen():
        import os

        os.environ["EXO_FROZEN"] = "1"

    main()
