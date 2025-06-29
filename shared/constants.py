from pathlib import Path

EXO_HOME = Path.home() / ".exo"
EXO_EVENT_DB = EXO_HOME / "event_db.sqlite3"
EXO_MASTER_CONFIG = EXO_HOME / "master.json"
EXO_WORKER_CONFIG = EXO_HOME / "worker.json"
EXO_MASTER_LOG = EXO_HOME / "master.log"
EXO_WORKER_LOG = EXO_HOME / "worker.log"

EXO_WORKER_KEYRING_FILE = EXO_HOME / "worker_keyring"
EXO_MASTER_KEYRING_FILE = EXO_HOME / "master_keyring"
