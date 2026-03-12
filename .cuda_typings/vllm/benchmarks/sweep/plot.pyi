import abc
import argparse
import pandas as pd
from .utils import sanitize_filename as sanitize_filename
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import ClassVar
from typing_extensions import Self, override
from vllm.utils.collection_utils import full_groupby as full_groupby
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

seaborn: Incomplete

@dataclass
class PlotFilterBase(ABC, metaclass=abc.ABCMeta):
    var: str
    target: str
    @classmethod
    def parse_str(cls, s: str): ...
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotEqualTo(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotNotEqualTo(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotLessThan(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotLessThanOrEqualTo(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotGreaterThan(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotGreaterThanOrEqualTo(PlotFilterBase):
    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

PLOT_FILTERS: dict[str, type[PlotFilterBase]]

class PlotFilters(list[PlotFilterBase]):
    @classmethod
    def parse_str(cls, s: str): ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class PlotBinner:
    var: str
    bin_size: float
    @classmethod
    def parse_str(cls, s: str): ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

PLOT_BINNERS: dict[str, type[PlotBinner]]

class PlotBinners(list[PlotBinner]):
    @classmethod
    def parse_str(cls, s: str): ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

class DummyExecutor:
    map = map
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None: ...

def plot(
    output_dir: Path,
    fig_dir: Path,
    fig_by: list[str],
    row_by: list[str],
    col_by: list[str],
    curve_by: list[str],
    *,
    var_x: str,
    var_y: str,
    filter_by: PlotFilters,
    bin_by: PlotBinners,
    scale_x: str | None,
    scale_y: str | None,
    dry_run: bool,
    fig_name: str = "FIGURE",
    error_bars: bool = True,
    fig_height: float = 6.4,
    fig_dpi: int = 300,
): ...
@dataclass
class SweepPlotArgs:
    output_dir: Path
    fig_dir: Path
    fig_by: list[str]
    row_by: list[str]
    col_by: list[str]
    curve_by: list[str]
    var_x: str
    var_y: str
    filter_by: PlotFilters
    bin_by: PlotBinners
    scale_x: str | None
    scale_y: str | None
    dry_run: bool
    fig_name: str = ...
    error_bars: bool = ...
    fig_height: float = ...
    fig_dpi: int = ...
    parser_name: ClassVar[str] = ...
    parser_help: ClassVar[str] = ...
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace): ...
    @classmethod
    def add_cli_args(
        cls, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser: ...

def run_main(args: SweepPlotArgs): ...
def main(args: argparse.Namespace): ...
