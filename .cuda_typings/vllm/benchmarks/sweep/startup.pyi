import argparse
import pandas as pd
from .param_sweep import (
    ParameterSweep as ParameterSweep,
    ParameterSweepItem as ParameterSweepItem,
)
from .utils import sanitize_filename as sanitize_filename
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

def run_benchmark(
    startup_cmd: list[str],
    *,
    serve_overrides: ParameterSweepItem,
    startup_overrides: ParameterSweepItem,
    run_number: int,
    output_path: Path,
    show_stdout: bool,
    dry_run: bool,
) -> dict[str, object] | None: ...
def run_comb(
    startup_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    startup_comb: ParameterSweepItem,
    base_path: Path,
    num_runs: int,
    show_stdout: bool,
    dry_run: bool,
) -> list[dict[str, object]] | None: ...
def run_combs(
    startup_cmd: list[str],
    *,
    serve_params: ParameterSweep,
    startup_params: ParameterSweep,
    experiment_dir: Path,
    num_runs: int,
    show_stdout: bool,
    dry_run: bool,
) -> pd.DataFrame | None: ...
@dataclass
class SweepStartupArgs:
    startup_cmd: list[str]
    serve_params: ParameterSweep
    startup_params: ParameterSweep
    output_dir: Path
    experiment_name: str
    num_runs: int
    show_stdout: bool
    dry_run: bool
    resume: bool
    parser_name: ClassVar[str] = ...
    parser_help: ClassVar[str] = ...
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace): ...
    @classmethod
    def add_cli_args(
        cls, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser: ...
    def resolve_experiment_dir(self) -> Path: ...
    @contextmanager
    def run_ctx(self, experiment_dir: Path): ...

def run_main(args: SweepStartupArgs): ...
def main(args: argparse.Namespace): ...
