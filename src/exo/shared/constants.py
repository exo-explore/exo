import os
from pathlib import Path

EXO_HOME_RELATIVE_PATH = os.environ.get("EXO_HOME", ".exo")
EXO_HOME = Path.home() / EXO_HOME_RELATIVE_PATH

EXO_MODELS_DIR_ENV = os.environ.get("EXO_MODELS_DIR")
EXO_MODELS_DIR = Path(EXO_MODELS_DIR_ENV) if EXO_MODELS_DIR_ENV else EXO_HOME / "models"

EXO_GLOBAL_EVENT_DB = EXO_HOME / "global_events.db"
EXO_WORKER_EVENT_DB = EXO_HOME / "worker_events.db"
EXO_MASTER_STATE = EXO_HOME / "master_state.json"
EXO_WORKER_STATE = EXO_HOME / "worker_state.json"
EXO_MASTER_LOG = EXO_HOME / "master.log"
EXO_WORKER_LOG = EXO_HOME / "worker.log"
EXO_LOG = EXO_HOME / "exo.log"
EXO_TEST_LOG = EXO_HOME / "exo_test.log"

EXO_NODE_ID_KEYPAIR = EXO_HOME / "node_id.keypair"

EXO_WORKER_KEYRING_FILE = EXO_HOME / "worker_keyring"
EXO_MASTER_KEYRING_FILE = EXO_HOME / "master_keyring"

EXO_IPC_DIR = EXO_HOME / "ipc"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

# lower bounds define timeouts for flops and memory bandwidth - these are the values for the M1 chip.
LB_TFLOPS = 2.3
LB_MEMBW_GBPS = 68
LB_DISK_GBPS = 1.5
