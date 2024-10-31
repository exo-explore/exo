from enum import Enum

class TelemetryAction(str, Enum):
    START = "start"
    REQUEST_RECEIVED = "request_received"
    STOP = "stop"
    ERROR = "error"