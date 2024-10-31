import json
from pathlib import Path
from platformdirs import user_data_dir
import requests
import traceback

from exo.config import session_config
from exo.telemetry.constants import TelemetryAction

class Logger:    
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, logging_url: str = None, node_id: str = None):
        if not self._initialized:
            assert logging_url is not None and node_id is not None, "logging_url and node_id are required for first initialization"
            self.logging_url = logging_url
            self.node_id = node_id
            self.log_file = self._init_log_file(node_id)
            self._initialized = True

    def _init_log_file(self, node_id: str) -> Path:
        app_data = Path(user_data_dir("exo", appauthor="exo_labs"))
        logs_dir = app_data / "logs" / node_id
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "logs.txt"
        # Open in write mode to clear/create the file
        with open(log_file, 'w') as f:
            pass

        # return the log file path
        return log_file

    def write_log(self, action: TelemetryAction, data: dict = {}) -> None:
        
        # Read existing lines if file exists
        lines = []
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                lines = f.readlines()
        
        # Add new line and keep only last 50 entries
        log_entry = {
            "commit_id": session_config.get("commit_id"),
            "action": action,
            "device_id": session_config.get("device_id"),
            "session_id": session_config.get("session_id"),
            "node_id": self.node_id,
            "topology": session_config.get("topology") or {},
            "data": data
        }
        
        # Filter out None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        lines.append(json.dumps(log_entry) + "\n")
        lines = lines[-50:]
        
        # Write back all lines
        with open(self.log_file, "w") as f:
            f.writelines(lines)


    def send_logs(self) -> None:
        headers = { 
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        # read the log file into a json array
        with open(self.log_file, "r") as f:
            log_entries = [json.loads(line) for line in f.readlines()]
        r = requests.post(self.logging_url, headers=headers, json=log_entries)
        if r.status_code != 200:
            print(f"Error reporting logs: {r.status_code} {r.text}")
        else:
            print("Logs reported successfully\n")
        

    def report_error(self, error: Exception) -> None:
        error_data = {
            "error": str(error),
            "stacktrace": traceback.format_exc()
        }
        self.write_log(TelemetryAction.ERROR, error_data)
        self.send_logs()
    

    