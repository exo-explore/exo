from enum import Enum

from shared.types.common import ID


class InstanceId(ID):
    pass


class RunnerId(ID):
    pass


class NodeStatus(str, Enum):
    Idle = "Idle"
    Running = "Running"

class RunnerError(Exception):
  """Exception raised when the runner process encounters an error."""
  
  def __init__(self, error_type: str, error_message: str, traceback: str):
    self.error_type = error_type
    self.error_message = error_message
    self.traceback = traceback
    super().__init__(f"{error_type}: {error_message}. Traceback: {traceback}")