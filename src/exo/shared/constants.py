import os
import sys
from pathlib import Path

EXO_HOME_ENV = os.environ.get("EXO_HOME", None)


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory with fallback. On non-Linux platforms, use ~/.exo."""

    # Setting EXO_HOME takes precedence over setting XDG_HOME
    if EXO_HOME_ENV is not None:
        return Path.home() / EXO_HOME_ENV

    if sys.platform != "linux":
        return Path.home() / ".exo"

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

# Models directory (data)
EXO_MODELS_DIR_ENV = os.environ.get("EXO_MODELS_DIR")
EXO_MODELS_DIR = (
    Path(EXO_MODELS_DIR_ENV) if EXO_MODELS_DIR_ENV else EXO_DATA_HOME / "models"
)

# Log files (data/logs or cache)
EXO_LOG = EXO_CACHE_HOME / "exo.log"
EXO_TEST_LOG = EXO_CACHE_HOME / "exo_test.log"

# Identity (config)
EXO_NODE_ID_KEYPAIR = EXO_CONFIG_HOME / "node_id.keypair"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"
