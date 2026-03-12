from pathlib import Path
from typing import Any
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

def generate_timeline_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    colors: list[str] | None = None,
    itl_thresholds: list[float] | None = None,
    labels: list[str] | None = None,
) -> None: ...
def construct_timeline_data(
    requests_data: list[dict[str, Any]], itl_thresholds: list[float], labels: list[str]
) -> list[dict[str, Any]]: ...
def generate_dataset_stats_plot(
    results: list[dict[str, Any]], output_path: Path
) -> None: ...
