"""exo --help works before the dashboard is built; serving the dashboard still fails with the hint
to build it."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from exo.shared import constants
from exo.utils import dashboard_path

# Runs exo's entry point with no dashboard to be found, wherever the tests run from
_EXO_WITHOUT_A_DASHBOARD = """
import sys
import exo.utils.dashboard_path as dashboard_path
dashboard_path._find_dashboard_in_repo = lambda: None
dashboard_path._find_dashboard_in_bundle = lambda: None
from exo.main import main
sys.argv = ["exo", *sys.argv[1:]]
main()
"""


def run_exo_without_a_dashboard(
    flag: str, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "EXO_HOME": str(tmp_path)}
    environment.pop("EXO_DASHBOARD_DIR", None)
    return subprocess.run(
        [sys.executable, "-c", _EXO_WITHOUT_A_DASHBOARD, flag],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_help_works_without_a_built_dashboard(tmp_path: Path):
    exo = run_exo_without_a_dashboard("--help", tmp_path)

    assert exo.returncode == 0, exo.stderr
    assert exo.stdout.startswith("usage: ")


def test_serving_the_dashboard_still_asks_for_it_to_be_built(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(constants, "_DASHBOARD_DIR_ENV", None)
    monkeypatch.setattr(dashboard_path, "_find_dashboard_in_repo", lambda: None)
    monkeypatch.setattr(dashboard_path, "_find_dashboard_in_bundle", lambda: None)

    with pytest.raises(FileNotFoundError, match="npm run build"):
        constants.dashboard_dir()
