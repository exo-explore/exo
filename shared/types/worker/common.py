from enum import Enum

from shared.types.common import ID


class InstanceId(ID):
    pass


class RunnerId(ID):
    pass


class NodeStatus(str, Enum):
    Idle = "Idle"
    Running = "Running"
