import inspect
import os
from pathlib import Path

EXO_HOME_RELATIVE_PATH = os.environ.get("EXO_HOME", ".exo")
EXO_HOME = Path.home() / EXO_HOME_RELATIVE_PATH
EXO_GLOBAL_EVENT_DB = EXO_HOME / "global_events.db"
EXO_WORKER_EVENT_DB = EXO_HOME / "worker_events.db"
EXO_MASTER_STATE = EXO_HOME / "master_state.json"
EXO_WORKER_STATE = EXO_HOME / "worker_state.json"
EXO_MASTER_LOG = EXO_HOME / "master.log"
EXO_WORKER_LOG = EXO_HOME / "worker.log"

EXO_NODE_ID_KEYPAIR = EXO_HOME / "node_id.keypair"

EXO_WORKER_KEYRING_FILE = EXO_HOME / "worker_keyring"
EXO_MASTER_KEYRING_FILE = EXO_HOME / "master_keyring"

# libp2p topics for event forwarding
LIBP2P_WORKER_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"

# lower bounds define timeouts for flops and memory bandwidth - these are the values for the M1 chip.
LB_TFLOPS = 2.3
LB_MEMBW_GBPS = 68
LB_DISK_GBPS = 1.5

# little helper function to get the name of the module that raised the error
def get_caller_module_name() -> str:
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    if mod is None:
        return "UNKNOWN MODULE"
    return mod.__name__


def get_error_reporting_message() -> str:
    return (
        f"THIS IS A BUG IN THE EXO SOFTWARE, PLEASE REPORT IT AT https://github.com/exo-explore/exo/\n"
        f"The module that raised the error was: {get_caller_module_name()}"
    )
