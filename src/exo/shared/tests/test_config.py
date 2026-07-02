from pathlib import Path

import pytest

from exo.shared.config import NodeConfig


@pytest.fixture(autouse=True)
def _clear_config_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_API_HOST", raising=False)
    monkeypatch.delenv("EXO_API_PORT", raising=False)


def test_node_config_reads_api_table(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [api]
        host = "127.0.0.1"
        port = 52416
        """,
        encoding="utf-8",
    )

    config = NodeConfig.load(config_path)

    assert config.api.host == "127.0.0.1"
    assert config.api.port == 52416


def test_node_config_applies_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [api]
        host = "127.0.0.1"
        port = 52416
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("EXO_API_HOST", "0.0.0.0")
    monkeypatch.setenv("EXO_API_PORT", "52417")

    config = NodeConfig.load(config_path)

    assert config.api.host == "0.0.0.0"
    assert config.api.port == 52417


def test_node_config_missing_file_uses_defaults(tmp_path: Path) -> None:
    config = NodeConfig.load(tmp_path / "missing.toml")

    assert config.api.host == "0.0.0.0"
    assert config.api.port == 52415
