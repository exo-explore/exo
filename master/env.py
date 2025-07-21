from pathlib import Path

from shared.env import BaseEnv


class MasterEnvironmentSchema(BaseEnv):
    # Master-specific: forwarder configuration
    FORWARDER_BINARY_PATH: Path | None = None
