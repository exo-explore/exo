"""Plot benchmark results — CLI subcommand.

    uv run python -m bench.cli plot bench/results/context_scaling/latest.json
    uv run python -m bench.cli plot run_a.json run_b.json --label-tag operator
    uv run python -m bench.cli plot latest.json --output /tmp/scaling.png

The benchmark type is detected from each JSON's ``metadata.benchmark`` —
all input files must share the same benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from loguru import logger

from bench.cli._common import get_arg_optional
from bench.lib.plotting import PlotInputs, render_context_scaling


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # type: ignore[type-arg]
) -> None:
    parser = subparsers.add_parser(
        "plot",
        help="Render benchmark JSON result(s) as a PNG.",
        description=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or more bench results JSON files. Multiple files are "
        "rendered as a comparison plot (one line per file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG output path. Default: replace the first JSON's '.json' "
        "suffix with '.png' (or '.compare.png' when multiple inputs).",
    )
    parser.add_argument(
        "--label-tag",
        type=str,
        default=None,
        help="Use metadata.tags[<KEY>] as the legend label for each run "
        "(falls back to run_id if unset or missing).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the auto-generated figure title.",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> Path:
    paths_raw = getattr(args, "paths", None)
    if not isinstance(paths_raw, list) or not paths_raw:
        raise SystemExit("plot: at least one JSON path is required")
    paths = [
        Path(str(p))  # type: ignore[reportUnknownArgumentType]
        for p in cast("list[Any]", paths_raw)  # type: ignore[reportAny]
    ]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"plot: file not found: {p}")

    benchmarks = {_benchmark_for(p) for p in paths}
    if len(benchmarks) != 1:
        raise SystemExit(
            f"plot: all input JSONs must share the same benchmark, got {benchmarks!r}"
        )
    benchmark = next(iter(benchmarks))

    output_arg = get_arg_optional(args, "output", str)
    output = Path(output_arg) if output_arg is not None else _default_output(paths)
    inputs = PlotInputs(
        results=paths,
        output=output,
        label_tag=get_arg_optional(args, "label_tag", str),
        title=get_arg_optional(args, "title", str),
    )

    if benchmark == "context_scaling":
        out_path = render_context_scaling(inputs)
    else:
        raise SystemExit(f"plot: no renderer registered for benchmark {benchmark!r}")

    logger.info(f"plot: wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------


def _benchmark_for(path: Path) -> str:
    with path.open() as f:
        loaded: Any = json.load(f)  # type: ignore[reportAny]
    if not isinstance(loaded, dict):
        raise SystemExit(f"plot: {path}: expected top-level JSON object")
    metadata: Any = loaded.get("metadata", {})  # type: ignore[reportAny]
    if not isinstance(metadata, dict):
        raise SystemExit(f"plot: {path}: metadata is not an object")
    benchmark: Any = metadata.get("benchmark")  # type: ignore[reportAny]
    if not isinstance(benchmark, str):
        raise SystemExit(f"plot: {path}: metadata.benchmark missing or not a string")
    return benchmark


def _default_output(paths: list[Path]) -> Path:
    """Auto-derive a PNG path next to the first JSON.

    Single input → ``<path>.png`` (replaces ``.json``).
    Multiple inputs → ``<path>.compare.png`` next to the first JSON.
    """
    first = paths[0]
    if len(paths) == 1:
        return first.with_suffix(".png")
    return first.with_name(first.stem + ".compare.png")
