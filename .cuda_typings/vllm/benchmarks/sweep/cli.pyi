import argparse
from .plot import SweepPlotArgs as SweepPlotArgs
from .plot_pareto import SweepPlotParetoArgs as SweepPlotParetoArgs
from .serve import SweepServeArgs as SweepServeArgs
from .serve_workload import SweepServeWorkloadArgs as SweepServeWorkloadArgs
from .startup import SweepStartupArgs as SweepStartupArgs
from _typeshed import Incomplete
from vllm.entrypoints.utils import (
    VLLM_SUBCMD_PARSER_EPILOG as VLLM_SUBCMD_PARSER_EPILOG,
)

SUBCOMMANDS: Incomplete

def add_cli_args(parser: argparse.ArgumentParser): ...
def main(args: argparse.Namespace): ...
