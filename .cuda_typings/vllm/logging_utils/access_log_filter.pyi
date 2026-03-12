import logging
from _typeshed import Incomplete

class UvicornAccessLogFilter(logging.Filter):
    excluded_paths: Incomplete
    def __init__(self, excluded_paths: list[str] | None = None) -> None: ...
    def filter(self, record: logging.LogRecord) -> bool: ...

def create_uvicorn_log_config(
    excluded_paths: list[str] | None = None, log_level: str = "info"
) -> dict: ...
