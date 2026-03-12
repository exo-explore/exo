import argparse
import contextlib
from .param_sweep import (
    ParameterSweep as ParameterSweep,
    ParameterSweepItem as ParameterSweepItem,
)
from .server import ServerProcess as ServerProcess
from .utils import sanitize_filename as sanitize_filename
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

@contextlib.contextmanager
def run_server(
    serve_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_overrides: ParameterSweepItem,
    dry_run: bool,
    server_ready_timeout: int = 300,
): ...
def run_benchmark(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_overrides: ParameterSweepItem,
    bench_overrides: ParameterSweepItem,
    run_number: int,
    output_path: Path,
    dry_run: bool,
): ...
def server_ctx(
    serve_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    serve_comb: ParameterSweepItem,
    bench_params: ParameterSweep,
    experiment_dir: Path,
    dry_run: bool,
    server_ready_timeout: int = 300,
): ...
def run_comb(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    link_vars: list[tuple[str, str]],
    base_path: Path,
    num_runs: int,
    dry_run: bool,
): ...
def run_combs(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    server_ready_timeout: int,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    link_vars: list[tuple[str, str]],
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
): ...
@dataclass
class SweepServeArgs:
    serve_cmd: list[str]
    bench_cmd: list[str]
    after_bench_cmd: list[str]
    show_stdout: bool
    server_ready_timeout: int
    serve_params: ParameterSweep
    bench_params: ParameterSweep
    link_vars: list[tuple[str, str]]
    output_dir: Path
    experiment_name: str
    num_runs: int
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
    @staticmethod
    def parse_link_vars(s: str) -> list[tuple[str, str]]: ...
    def resolve_experiment_dir(self) -> Path: ...
    @contextmanager
    def run_ctx(self, experiment_dir: Path): ...

def run_main(args: SweepServeArgs): ...
def main(args: argparse.Namespace): ...
