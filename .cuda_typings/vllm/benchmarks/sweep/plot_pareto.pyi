import argparse
from .plot import DummyExecutor as DummyExecutor
from .utils import sanitize_filename as sanitize_filename
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from vllm.utils.collection_utils import full_groupby as full_groupby
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

seaborn: Incomplete

def plot_pareto(
    output_dir: Path,
    user_count_var: str | None,
    gpu_count_var: str | None,
    label_by: list[str],
    *,
    dry_run: bool,
): ...
@dataclass
class SweepPlotParetoArgs:
    output_dir: Path
    user_count_var: str | None
    gpu_count_var: str | None
    label_by: list[str]
    dry_run: bool
    parser_name: ClassVar[str] = ...
    parser_help: ClassVar[str] = ...
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace): ...
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser): ...

def run_main(args: SweepPlotParetoArgs): ...
def main(args: argparse.Namespace): ...
