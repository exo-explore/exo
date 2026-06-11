"""Tests for XDG Base Directory Specification compliance."""

import sys
from pathlib import Path

import pytest
from exo_rs import LocatorConfig


@pytest.mark.skipif(sys.platform != "linux", reason="XDG dirs are Linux-specific")
def test_xdg_paths_on_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that XDG paths are used on Linux when XDG env vars are set."""
    config_home = tmp_path / "test-config"
    data_home = tmp_path / "test-data"
    cache_home = tmp_path / "test-cache"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    exo_home = LocatorConfig.from_env_only().exo_home

    assert config_home / "exo" == exo_home.config
    assert data_home / "exo" == exo_home.data
    assert cache_home / "exo" == exo_home.cache


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS dirs are Darwin-specific")
def test_standard_directories_on_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that macOS standard directories are derived from HOME."""
    home = tmp_path / "home"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "ignored-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "ignored-cache"))

    exo_home = LocatorConfig.from_env_only().exo_home

    assert home / "Library" / "Application Support" / "exo" == exo_home.config
    assert home / "Library" / "Application Support" / "exo" == exo_home.data
    assert home / "Library" / "Caches" / "exo" == exo_home.cache


@pytest.mark.skipif(sys.platform != "linux", reason="XDG dirs are Linux-specific")
def test_xdg_default_paths_on_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that XDG default paths are used on Linux when env vars are not set."""
    home = tmp_path / "home"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    exo_home = LocatorConfig.from_env_only().exo_home

    assert home / ".config" / "exo" == exo_home.config
    assert home / ".local" / "share" / "exo" == exo_home.data
    assert home / ".cache" / "exo" == exo_home.cache


def test_legacy_exo_home_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that EXO_HOME environment variable takes precedence for backward compatibility."""
    exo_home_path = tmp_path / ".custom-exo"

    monkeypatch.setenv("EXO_HOME", str(exo_home_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "ignored-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "ignored-cache"))

    exo_home = LocatorConfig.from_env_only().exo_home

    assert exo_home_path == exo_home.config
    assert exo_home_path == exo_home.data
    assert exo_home_path == exo_home.cache


@pytest.mark.skipif(sys.platform != "linux", reason="XDG dirs are Linux-specific")
def test_models_in_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that default models directory is in the data directory."""
    data_home = tmp_path / "data"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_DEFAULT_MODELS_DIR", raising=False)
    monkeypatch.delenv("EXO_MODELS_DIRS", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    cfg = LocatorConfig.from_env_only()

    assert cfg.models_dirs.default_models_dir.parent == cfg.exo_home.data


def test_default_dir_always_prepended_to_models_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that the default models dir is always the first entry in EXO_MODELS_DIRS."""
    custom_models_dir = tmp_path / "custom-models"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_DEFAULT_MODELS_DIR", raising=False)
    monkeypatch.delenv("EXO_MODELS_READ_ONLY_DIRS", raising=False)
    monkeypatch.setenv("EXO_MODELS_DIRS", str(custom_models_dir))

    models_dirs = LocatorConfig.from_env_only().models_dirs

    assert models_dirs.models_dirs[0] == models_dirs.default_models_dir
    assert custom_models_dir in models_dirs.models_dirs


def test_default_models_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that EXO_DEFAULT_MODELS_DIR can be overridden via env var."""
    default_models_dir = tmp_path / "exo-models"

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_MODELS_DIRS", raising=False)
    monkeypatch.delenv("EXO_MODELS_READ_ONLY_DIRS", raising=False)
    monkeypatch.setenv("EXO_DEFAULT_MODELS_DIR", str(default_models_dir))

    models_dirs = LocatorConfig.from_env_only().models_dirs

    assert default_models_dir == models_dirs.default_models_dir
    assert models_dirs.models_dirs[0] == models_dirs.default_models_dir


def test_default_dir_only_entry_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    """Test that EXO_MODELS_DIRS contains only the default when env var is not set."""
    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_DEFAULT_MODELS_DIR", raising=False)
    monkeypatch.delenv("EXO_MODELS_DIRS", raising=False)
    monkeypatch.delenv("EXO_MODELS_READ_ONLY_DIRS", raising=False)

    models_dirs = LocatorConfig.from_env_only().models_dirs

    assert models_dirs.models_dirs == [models_dirs.default_models_dir]


def test_overlap_between_dirs_and_read_only_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that a directory in both lists is excluded from writable dirs."""
    shared = tmp_path / "shared"
    writable_only = tmp_path / "writable-only"
    read_only = tmp_path / "read-only"
    shared.mkdir()
    read_only.mkdir()

    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_DEFAULT_MODELS_DIR", raising=False)
    monkeypatch.setenv("EXO_MODELS_DIRS", f"{shared}:{writable_only}")
    monkeypatch.setenv("EXO_MODELS_READ_ONLY_DIRS", f"{shared}:{read_only}")

    models_dirs = LocatorConfig.from_env_only().models_dirs

    assert shared not in models_dirs.models_dirs
    assert writable_only in models_dirs.models_dirs
    assert shared in models_dirs.models_read_only_dirs
    assert read_only in models_dirs.models_read_only_dirs


def test_empty_read_only_dirs_when_unset(monkeypatch: pytest.MonkeyPatch):
    """Test that EXO_MODELS_READ_ONLY_DIRS is empty when env var is not set."""
    monkeypatch.delenv("EXO_HOME", raising=False)
    monkeypatch.delenv("EXO_DEFAULT_MODELS_DIR", raising=False)
    monkeypatch.delenv("EXO_MODELS_DIRS", raising=False)
    monkeypatch.delenv("EXO_MODELS_READ_ONLY_DIRS", raising=False)

    models_dirs = LocatorConfig.from_env_only().models_dirs

    assert models_dirs.models_read_only_dirs == []
