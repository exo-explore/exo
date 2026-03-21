"""Tests for XDG Base Directory Specification compliance."""

import os
import sys
from pathlib import Path
from unittest import mock


def test_xdg_paths_on_linux():
    """Test that XDG paths are used on Linux when XDG env vars are set."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "XDG_CONFIG_HOME": "/tmp/test-config",
                "XDG_DATA_HOME": "/tmp/test-data",
                "XDG_CACHE_HOME": "/tmp/test-cache",
            },
            clear=False,
        ),
        mock.patch.object(sys, "platform", "linux"),
    ):
        # Re-import to pick up mocked values
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert Path("/tmp/test-config/exo") == constants.EXO_CONFIG_HOME
        assert Path("/tmp/test-data/exo") == constants.EXO_DATA_HOME
        assert Path("/tmp/test-cache/exo") == constants.EXO_CACHE_HOME


def test_xdg_default_paths_on_linux():
    """Test that XDG default paths are used on Linux when env vars are not set."""
    # Remove XDG env vars and EXO_HOME
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("XDG_") and k != "EXO_HOME"
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "linux"),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".config" / "exo" == constants.EXO_CONFIG_HOME
        assert home / ".local/share" / "exo" == constants.EXO_DATA_HOME
        assert home / ".cache" / "exo" == constants.EXO_CACHE_HOME


def test_legacy_exo_home_takes_precedence():
    """Test that EXO_HOME environment variable takes precedence for backward compatibility."""
    with mock.patch.dict(
        os.environ,
        {
            "EXO_HOME": ".custom-exo",
            "XDG_CONFIG_HOME": "/tmp/test-config",
        },
        clear=False,
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".custom-exo" == constants.EXO_CONFIG_HOME
        assert home / ".custom-exo" == constants.EXO_DATA_HOME


def test_macos_uses_traditional_paths():
    """Test that macOS uses traditional ~/.exo directory."""
    # Remove EXO_HOME to ensure we test the default behavior
    env = {k: v for k, v in os.environ.items() if k != "EXO_HOME"}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "darwin"),
    ):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".exo" == constants.EXO_CONFIG_HOME
        assert home / ".exo" == constants.EXO_DATA_HOME
        assert home / ".exo" == constants.EXO_CACHE_HOME


def test_node_id_in_config_dir():
    """Test that node ID keypair is in the config directory."""
    import exo.shared.constants as constants

    assert constants.EXO_NODE_ID_KEYPAIR.parent == constants.EXO_CONFIG_HOME


def test_models_in_data_dir():
    """Test that default models directory is in the data directory."""
    # Clear EXO_MODELS_DIRS to test default behavior
    env = {k: v for k, v in os.environ.items() if k != "EXO_MODELS_DIRS"}
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.EXO_DEFAULT_MODELS_DIR.parent == constants.EXO_DATA_HOME


def test_default_dir_always_prepended_to_models_dirs():
    """Test that the default models dir is always the first entry in EXO_MODELS_DIRS."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS", "EXO_HOME")
    }
    env["EXO_MODELS_DIRS"] = "/tmp/custom-models"
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.EXO_MODELS_DIRS[0] == constants.EXO_DEFAULT_MODELS_DIR
        assert Path("/tmp/custom-models") in constants.EXO_MODELS_DIRS


def test_default_dir_only_entry_when_env_unset():
    """Test that EXO_MODELS_DIRS contains only the default when env var is not set."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS", "EXO_HOME")
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.EXO_MODELS_DIRS == (constants.EXO_DEFAULT_MODELS_DIR,)


def test_overlap_between_dirs_and_read_only_dirs():
    """Test that a directory in both lists is excluded from writable dirs."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS", "EXO_HOME")
    }
    env["EXO_MODELS_DIRS"] = "/tmp/shared:/tmp/writable-only"
    env["EXO_MODELS_READ_ONLY_DIRS"] = "/tmp/shared:/tmp/ro-only"
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        # /tmp/shared should be excluded from writable dirs
        assert Path("/tmp/shared") not in constants.EXO_MODELS_DIRS
        assert Path("/tmp/writable-only") in constants.EXO_MODELS_DIRS
        # /tmp/shared should still be in read-only dirs
        assert Path("/tmp/shared") in constants.EXO_MODELS_READ_ONLY_DIRS
        assert Path("/tmp/ro-only") in constants.EXO_MODELS_READ_ONLY_DIRS


def test_empty_read_only_dirs_when_unset():
    """Test that EXO_MODELS_READ_ONLY_DIRS is empty when env var is not set."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS", "EXO_HOME")
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import exo.shared.constants as constants

        importlib.reload(constants)

        assert constants.EXO_MODELS_READ_ONLY_DIRS == ()
