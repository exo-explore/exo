from pathlib import Path

from exo.shared.env import BaseEnv


class MasterEnvironmentSchema(BaseEnv):
    # Master-specific: forwarder configuration
    # Default to build/forwarder if not explicitly set
    FORWARDER_BINARY_PATH: Path = Path("build/forwarder")
