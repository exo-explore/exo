import os
import sys
from pathlib import Path

from exo.utils.dashboard_path import find_dashboard, find_resources

_EXO_HOME_ENV = os.environ.get("EXO_HOME", "")


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory, prioritising EXO_HOME environment variable if its set. On non-Linux platforms, default to ~/.exo."""

    if _EXO_HOME_ENV != "":
        return Path.home() / _EXO_HOME_ENV

    if sys.platform != "linux":
        return Path.home() / ".exo"

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

# Default models directory (always included as first entry in writable dirs)
_EXO_DEFAULT_MODELS_DIR_ENV = os.environ.get("EXO_DEFAULT_MODELS_DIR", None)
EXO_DEFAULT_MODELS_DIR = (
    Path(_EXO_DEFAULT_MODELS_DIR_ENV).expanduser()
    if _EXO_DEFAULT_MODELS_DIR_ENV is not None
    else EXO_DATA_HOME / "models"
)


def _parse_colon_dirs(env_var: str) -> tuple[Path, ...]:
    raw = os.environ.get(env_var, None)
    if raw is None:
        return ()
    return tuple(Path(p).expanduser() for p in raw.split(":") if p)


# Read-only model directories (colon-separated). Never written to or deleted from.
_EXO_MODELS_READ_ONLY_DIRS_ENV = _parse_colon_dirs("EXO_MODELS_READ_ONLY_DIRS")
# Writable model directories (colon-separated). Default dir is always prepended.
_EXO_MODELS_DIRS_ENV = _parse_colon_dirs("EXO_MODELS_DIRS")

# If a directory appears in both lists, treat it as read-only.
_read_only_set = frozenset(_EXO_MODELS_READ_ONLY_DIRS_ENV)
EXO_MODELS_DIRS: tuple[Path, ...] = tuple(
    d
    for d in (EXO_DEFAULT_MODELS_DIR, *_EXO_MODELS_DIRS_ENV)
    if d not in _read_only_set
)
EXO_MODELS_READ_ONLY_DIRS: tuple[Path, ...] = _EXO_MODELS_READ_ONLY_DIRS_ENV

_RESOURCES_DIR_ENV = os.environ.get("EXO_RESOURCES_DIR", None)
RESOURCES_DIR = (
    find_resources() if _RESOURCES_DIR_ENV is None else Path.home() / _RESOURCES_DIR_ENV
)
_DASHBOARD_DIR_ENV = os.environ.get("EXO_DASHBOARD_DIR", None)
DASHBOARD_DIR = (
    find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
)

# Log files (data/logs or cache)
EXO_LOG_DIR = EXO_CACHE_HOME / "exo_log"
EXO_LOG = EXO_LOG_DIR / "exo.log"
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

EXO_CUSTOM_MODEL_CARDS_DIR = EXO_DATA_HOME / "custom_model_cards"

EXO_EVENT_LOG_DIR = EXO_DATA_HOME / "event_log"
EXO_IMAGE_CACHE_DIR = EXO_CACHE_HOME / "images"
EXO_TRACING_CACHE_DIR = EXO_CACHE_HOME / "traces"

EXO_ENABLE_IMAGE_MODELS = (
    os.getenv("EXO_ENABLE_IMAGE_MODELS", "false").lower() == "true"
)

EXO_OFFLINE = os.getenv("EXO_OFFLINE", "false").lower() == "true"

EXO_TRACING_ENABLED = os.getenv("EXO_TRACING_ENABLED", "false").lower() == "true"

ENABLE_DISAGGREGATION = os.getenv("ENABLE_DISAGGREGATION", "false").lower() == "true"

EXO_MAX_CONCURRENT_REQUESTS = int(os.getenv("EXO_MAX_CONCURRENT_REQUESTS", "8"))

EXO_MAX_INSTANCE_RETRIES = 5
