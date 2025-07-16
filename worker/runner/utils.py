import sys


def get_runner_command() -> list[str]:
    python = sys.executable
    return [python, "-m", "worker.runner.runner"]
