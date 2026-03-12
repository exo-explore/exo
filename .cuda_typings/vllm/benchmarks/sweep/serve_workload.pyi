import argparse
from .param_sweep import (
    ParameterSweep as ParameterSweep,
    ParameterSweepItem as ParameterSweepItem,
)
from .serve import (
    SweepServeArgs as SweepServeArgs,
    run_comb as run_comb,
    server_ctx as server_ctx,
)
from .server import ServerProcess as ServerProcess
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from vllm.benchmarks.datasets import DEFAULT_NUM_PROMPTS as DEFAULT_NUM_PROMPTS
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

WorkloadVariable: Incomplete

def run_comb_workload(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    link_vars: list[tuple[str, str]],
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
    workload_var: WorkloadVariable,
    workload_value: int,
) -> list[dict[str, object]] | None: ...
def explore_comb_workloads(
    server: ServerProcess | None,
    bench_cmd: list[str],
    *,
    serve_comb: ParameterSweepItem,
    bench_comb: ParameterSweepItem,
    link_vars: list[tuple[str, str]],
    workload_var: WorkloadVariable,
    workload_iters: int,
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
): ...
def explore_combs_workloads(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    server_ready_timeout: int,
    serve_params: ParameterSweep,
    bench_params: ParameterSweep,
    link_vars: list[tuple[str, str]],
    workload_var: WorkloadVariable,
    workload_iters: int,
    experiment_dir: Path,
    num_runs: int,
    dry_run: bool,
): ...
@dataclass
class SweepServeWorkloadArgs(SweepServeArgs):
    workload_var: WorkloadVariable
    workload_iters: int
    parser_name: ClassVar[str] = ...
    parser_help: ClassVar[str] = ...
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace): ...
    @classmethod
    def add_cli_args(
        cls, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser: ...

def run_main(args: SweepServeWorkloadArgs): ...
def main(args: argparse.Namespace): ...
