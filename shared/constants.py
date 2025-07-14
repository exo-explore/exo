from pathlib import Path
import inspect

EXO_HOME = Path.home() / ".exo"
EXO_EVENT_DB = EXO_HOME / "event_db.sqlite3"
EXO_MASTER_STATE = EXO_HOME / "master_state.json"
EXO_WORKER_STATE = EXO_HOME / "worker_state.json"
EXO_MASTER_LOG = EXO_HOME / "master.log"
EXO_WORKER_LOG = EXO_HOME / "worker.log"

EXO_WORKER_KEYRING_FILE = EXO_HOME / "worker_keyring"
EXO_MASTER_KEYRING_FILE = EXO_HOME / "master_keyring"


# little helper function to get the name of the module that raised the error
def get_caller_module_name() -> str:
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    if mod is None:
        return "UNKNOWN MODULE"
    return mod.__name__


EXO_ERROR_REPORTING_MESSAGE = lambda: (
    f"THIS IS A BUG IN THE EXO SOFTWARE, PLEASE REPORT IT AT https://github.com/exo-explore/exo/\n"
    f"The module that raised the error was: {get_caller_module_name()}"
)
