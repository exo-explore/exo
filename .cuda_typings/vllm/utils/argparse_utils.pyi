from _typeshed import Incomplete
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    RawDescriptionHelpFormatter,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

class SortedHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    def add_arguments(self, actions): ...

class FlexibleArgumentParser(ArgumentParser):
    add_json_tip: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    epilog: Incomplete
    def format_help(self): ...
    def parse_args(
        self, args: list[str] | None = None, namespace: Namespace | None = None
    ): ...
    def check_port(self, value): ...
    def load_config_file(self, file_path: str) -> list[str]: ...
