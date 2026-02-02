import os
import sys
from pathlib import Path

_EXO_HOME_ENV = os.environ.get("EXO_HOME", None)


def _resolve_home_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.home() / path


def _get_windows_dir(kind: str) -> Path:
    if kind == "config":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "exo"
        return Path.home() / "AppData" / "Roaming" / "exo"

    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "exo"
    return Path.home() / "AppData" / "Local" / "exo"


def _get_xdg_dir(env_var: str, fallback: str, *, kind: str) -> Path:
    """Get EXO directory for the platform, honoring EXO_HOME when set."""
    if _EXO_HOME_ENV is not None:
        return _resolve_home_path(_EXO_HOME_ENV)

    if sys.platform == "win32":
        return _get_windows_dir(kind)

    if sys.platform != "linux":
        return Path.home() / ".exo"

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config", kind="config")
EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share", kind="data")
EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache", kind="cache")

# Models directory (data)
_EXO_MODELS_DIR_ENV = os.environ.get("EXO_MODELS_DIR", None)
EXO_MODELS_DIR = (
    EXO_DATA_HOME / "models"
    if _EXO_MODELS_DIR_ENV is None
    else _resolve_home_path(_EXO_MODELS_DIR_ENV)
)

# Log files (data/logs or cache)
EXO_LOG = EXO_CACHE_HOME / "exo.log"
EXO_TEST_LOG = EXO_CACHE_HOME / "exo_test.log"

# Identity (config)
EXO_NODE_ID_KEYPAIR = EXO_CONFIG_HOME / "node_id.keypair"
EXO_CONFIG_FILE = EXO_CONFIG_HOME / "config.toml"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

EXO_MAX_CHUNK_SIZE = 512 * 1024

EXO_IMAGE_CACHE_DIR = EXO_CACHE_HOME / "images"

EXO_ENABLE_IMAGE_MODELS = (
    os.getenv("EXO_ENABLE_IMAGE_MODELS", "false").lower() == "true"
)
