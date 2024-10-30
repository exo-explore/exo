import json
from pathlib import Path
from platformdirs import user_data_dir
import uuid

class PersistentConfig:
    """Persistent configuration that should be saved between sessions"""
    CONFIG_FILE_NAME = "config.json"
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._config_file = self.initialize()
            self._initialized = True

            # add default values
            self.set("device_id", str(uuid.uuid4()), replace_if_exists=False)
            self.set("node_id", str(uuid.uuid4()), replace_if_exists=False)

    def initialize(self):
        app_data = Path(user_data_dir("exo", appauthor="exo_labs"))
        app_data.mkdir(parents=True, exist_ok=True)
        print(f"Using app data directory: {app_data}")
        config_file = app_data / self.CONFIG_FILE_NAME
        
        if not config_file.exists():
            with config_file.open('w') as f:
                json.dump({}, f)
        
        return config_file

    def set(self, key: str, value: any, replace_if_exists: bool = True):
        print(f"Setting {key}={value} in config file")
        
        with self._config_file.open('r') as f:
            config = json.load(f)
        
        # Update config and write back to file
        if replace_if_exists or key not in config:
            config[key] = value
            with self._config_file.open('w') as f:
                json.dump(config, f, indent=4)
            
            # Sync to session config
            SessionConfig().set(key, value)


    def get(self, key: str):
        with self._config_file.open('r') as f:
            return json.load(f).get(key)

class SessionConfig:
    """
    Handles temporary configuration specific to the current session
    
    Note that this syncs with the PersistentConfig instance and is a 
    superset of the data there.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Load persistent config during initialization
            self._session_data = {}
            self.sync_with_persistent()
            self.set("session_id", str(uuid.uuid4()))
            self._initialized = True

    def sync_with_persistent(self):
        """Sync session data with persistent config"""
        persistent = PersistentConfig()
        with persistent._config_file.open('r') as f:
            persistent_data = json.load(f)
            self._session_data.update(persistent_data)

    def set(self, key: str, value: any, replace_if_exists: bool = True):
        if replace_if_exists or key not in self._session_data:
            self._session_data[key] = value

    def get(self, key: str):
        return self._session_data.get(key)

# Expose singleton instances
session_config = SessionConfig()
persistent_config = PersistentConfig()