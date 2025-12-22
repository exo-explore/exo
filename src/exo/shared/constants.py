import os
import sys
from pathlib import Path


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory with fallback. On non-Linux platforms, use ~/.exo."""
    # On macOS/Windows, use traditional ~/.exo directory
    if sys.platform != "linux":
        return Path.home() / ".exo"

    # On Linux, follow XDG Base Directory Specification
    xdg_value = os.environ.get(env_var)
    if xdg_value:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


# Support legacy EXO_HOME environment variable for backward compatibility
_EXO_HOME_ENV = os.environ.get("EXO_HOME")
if _EXO_HOME_ENV:
    # Legacy mode: use EXO_HOME for all files
    EXO_CONFIG_HOME = Path.home() / _EXO_HOME_ENV
    EXO_DATA_HOME = Path.home() / _EXO_HOME_ENV
    EXO_CACHE_HOME = Path.home() / _EXO_HOME_ENV
else:
    # XDG mode (Linux) or traditional mode (macOS/Windows)
    EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
    EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
    EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

# Legacy alias for backward compatibility
EXO_HOME = EXO_DATA_HOME

# Models directory (data)
EXO_MODELS_DIR_ENV = os.environ.get("EXO_MODELS_DIR")
EXO_MODELS_DIR = (
    Path(EXO_MODELS_DIR_ENV) if EXO_MODELS_DIR_ENV else EXO_DATA_HOME / "models"
)

# Database and state files (data)
EXO_GLOBAL_EVENT_DB = EXO_DATA_HOME / "global_events.db"
EXO_WORKER_EVENT_DB = EXO_DATA_HOME / "worker_events.db"
EXO_MASTER_STATE = EXO_DATA_HOME / "master_state.json"
EXO_WORKER_STATE = EXO_DATA_HOME / "worker_state.json"

# Log files (data/logs or cache)
EXO_MASTER_LOG = EXO_DATA_HOME / "master.log"
EXO_WORKER_LOG = EXO_DATA_HOME / "worker.log"
EXO_LOG = EXO_DATA_HOME / "exo.log"
EXO_TEST_LOG = EXO_DATA_HOME / "exo_test.log"

# Identity and keys (config)
EXO_NODE_ID_KEYPAIR = EXO_CONFIG_HOME / "node_id.keypair"
EXO_WORKER_KEYRING_FILE = EXO_CONFIG_HOME / "worker_keyring"
EXO_MASTER_KEYRING_FILE = EXO_CONFIG_HOME / "master_keyring"

# IPC directory (runtime/cache)
EXO_IPC_DIR = EXO_CACHE_HOME / "ipc"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

# lower bounds define timeouts for flops and memory bandwidth - these are the values for the M1 chip.
LB_TFLOPS = 2.3
LB_MEMBW_GBPS = 68
LB_DISK_GBPS = 1.5
