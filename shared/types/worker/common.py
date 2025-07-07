from enum import Enum

from shared.types.common import NewUUID

class InstanceId(NewUUID):
    pass


class RunnerId(NewUUID):
    pass


class NodeStatus(str, Enum):
    Idle = "Idle"
    Running = "Running"
    Paused = "Paused"