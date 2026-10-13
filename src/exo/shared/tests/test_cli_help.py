"""exo --help describes exo and every flag it shows, including the ports a firewall must allow, and hides
developer-only flags, which keep working."""

import sys

import pytest

from exo.main import Args

_DEVELOPER_FLAGS = (
    "--no-batch",
    "--bootstrap-peers",
    "--fast-synch",
    "--no-fast-synch",
)


def help_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> str:
    monkeypatch.setattr(sys, "argv", ["exo", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        Args.parse()
    assert exit_info.value.code == 0
    return capsys.readouterr().out


def help_by_option(help_text: str) -> dict[str, str]:
    """Each option shown under "options:" in --help output, mapped to its help text."""
    lines = help_text.splitlines()
    help_texts: dict[str, str] = {}
    current_option: str | None = None
    for line in lines[lines.index("options:") + 1 :]:
        if line.startswith("  -"):
            current_option, _, first_help_line = line.strip().partition("  ")
            help_texts[current_option] = first_help_line.strip()
        elif current_option is not None and line.startswith("   "):
            help_texts[current_option] = (
                f"{help_texts[current_option]} {line.strip()}".strip()
            )
    return help_texts


def test_help_has_a_description(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    help_text = help_output(monkeypatch, capsys)

    assert help_text.startswith("usage: exo ")
    assert "Run an exo node" in help_text


def test_every_shown_flag_has_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    help_texts = help_by_option(help_output(monkeypatch, capsys))
    shown_flags = {
        word
        for option in help_texts
        for word in option.replace(",", "").split()
        if word.startswith("-")
    }

    assert {"-q", "-v", "-m", "--no-api", "--api-port", "--no-worker"} <= shown_flags
    assert [option for option, text in help_texts.items() if not text] == []


def test_developer_flags_are_hidden(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    help_text = help_output(monkeypatch, capsys)

    for flag in _DEVELOPER_FLAGS:
        assert flag not in help_text


def test_hidden_flags_and_ports_still_work(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exo",
            "--no-batch",
            "--zenoh-port",
            "6000",
            "--discovery-port",
            "6001",
            "--no-fast-synch",
        ],
    )

    args = Args.parse()

    assert args.no_batch
    assert args.zenoh_port == 6000
    assert args.discovery_port == 6001
    assert args.fast_synch is False
