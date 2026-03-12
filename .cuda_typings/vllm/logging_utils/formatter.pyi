import logging
from _typeshed import Incomplete
from vllm import envs as envs

class NewLineFormatter(logging.Formatter):
    use_relpath: Incomplete
    root_dir: Incomplete
    def __init__(self, fmt, datefmt=None, style: str = "%") -> None: ...
    def format(self, record): ...

class ColoredFormatter(NewLineFormatter):
    COLORS: Incomplete
    GREY: str
    RESET: str
    def __init__(self, fmt, datefmt=None, style: str = "%") -> None: ...
    def format(self, record): ...
