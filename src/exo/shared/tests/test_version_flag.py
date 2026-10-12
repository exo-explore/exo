import sys

import pytest

from exo import __version__
from exo.main import Args


def test_version_flag_prints_the_version_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["exo", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        Args.parse()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"exo {__version__}\n"
