"""Run a campaign of bench invocations from a single TOML file.

A campaign config has a ``[defaults]`` table (applied to every run) and a
list of ``[[runs]]`` entries (each a fully-formed invocation with its own
``subcommand``). The campaign runner merges defaults with each run's
overrides, dispatches to the matching subcommand handler, and collects
the output JSON paths.

Each run gets its own cluster — the deploy / teardown happens per-run.
After all runs finish, an optional ``[plot]`` table triggers a comparison
plot per benchmark group.

Schema::

    [defaults]
    nodes = 4
    num_steps = 8

    [[runs]]
    subcommand = "context-scaling"
    model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    [runs.tags]
    model_short = "llama-3.2-3b"

    [[runs]]
    subcommand = "context-scaling"
    model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    [runs.tags]
    model_short = "llama-3.1-8b"

    [plot]
    label_tag = "model_short"
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from loguru import logger

from bench.cli import context_scaling
from bench.cli._common import (
    _config_to_argv,  # type: ignore[reportPrivateUsage]
    get_arg,
)
from bench.lib.plotting import PlotInputs, render_context_scaling

# Each subcommand exposes its argparse via add_subparser. The campaign
# runner builds a one-off parser per run with only the chosen subcommand
# registered, parses the run-derived argv, and invokes the handler.
_SUBCOMMAND_PARSERS: dict[
    str,
    Callable[
        [Any], None
    ],  # subparsers action — argparse private; Any-typed at boundary
] = {
    "context-scaling": context_scaling.add_subparser,
}


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # type: ignore[type-arg]
) -> None:
    parser = subparsers.add_parser(
        "campaign",
        help="Run a list of bench invocations from a single TOML config.",
        description=__doc__,
    )
    parser.add_argument(
        "config",
        type=str,
        help="TOML campaign file (with [defaults] + [[runs]] tables).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the optional comparison plot at the end of the campaign.",
    )
    parser.set_defaults(handler=run)


# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Path | None:
    config_path = Path(get_arg(args, "config", str))
    if not config_path.is_file():
        raise SystemExit(f"campaign: file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    defaults = _table(raw, "defaults")
    runs_obj = raw.get("runs")
    if not isinstance(runs_obj, list) or not runs_obj:
        raise SystemExit(f"campaign: {config_path}: missing or empty [[runs]] list")
    runs_raw: list[Any] = cast("list[Any]", runs_obj)
    plot_cfg = _table(raw, "plot")

    n_runs = len(runs_raw)
    output_paths: dict[str, list[Path]] = {}
    for i, run_obj in enumerate(runs_raw):  # type: ignore[reportAny]
        if not isinstance(run_obj, dict):
            raise SystemExit(
                f"campaign: run #{i + 1}: expected a TOML table, "
                f"got {type(run_obj).__name__}"  # type: ignore[reportUnknownArgumentType]
            )
        run_cfg = cast("dict[str, Any]", run_obj)

        merged = _merge(defaults, run_cfg)
        subcommand_obj: Any = merged.pop("subcommand", None)  # type: ignore[reportAny]
        if not isinstance(subcommand_obj, str):
            raise SystemExit(
                f"campaign: run #{i + 1}: 'subcommand' field is required (str)"
            )
        if subcommand_obj not in _SUBCOMMAND_PARSERS:
            raise SystemExit(
                f"campaign: run #{i + 1}: unknown subcommand "
                f"{subcommand_obj!r} (have {sorted(_SUBCOMMAND_PARSERS)})"
            )

        argv_for_run = _config_to_argv(merged)
        sub_args = _parse_for_subcommand(subcommand_obj, argv_for_run)
        handler = getattr(sub_args, "handler", None)
        if not callable(handler):
            raise SystemExit(f"campaign: subcommand {subcommand_obj!r} has no handler")

        logger.info(
            f"campaign: starting run {i + 1}/{n_runs} "
            f"({subcommand_obj}; {len(merged)} flags)"
        )
        out = cast("Callable[[argparse.Namespace], Path]", handler)(sub_args)
        output_paths.setdefault(subcommand_obj, []).append(Path(out))
        logger.info(f"campaign: finished run {i + 1}/{n_runs} → {out}")

    last_path: Path | None = None
    for paths in output_paths.values():
        if paths:
            last_path = paths[-1]

    if get_arg(args, "no_plot", bool):
        return last_path

    comparison = _render_comparisons(output_paths, plot_cfg)
    return comparison or last_path


# ---------------------------------------------------------------------------


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``data[key]`` if it's a table, else an empty dict."""
    val: Any = data.get(key)
    return cast("dict[str, Any]", val) if isinstance(val, dict) else {}


def _merge(defaults: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``defaults`` with ``run``; ``run`` wins on conflict.

    The ``tags`` table is deep-merged (defaults' tags + run's tags) so a
    campaign-level operator tag and a per-run model_short tag both survive.
    """
    merged: dict[str, Any] = {**defaults, **run}
    default_tags = _table(defaults, "tags")
    run_tags = _table(run, "tags")
    if default_tags or run_tags:
        merged["tags"] = {**default_tags, **run_tags}
    return merged


def _parse_for_subcommand(
    subcommand: str, argv_for_run: list[str]
) -> argparse.Namespace:
    """Build a one-off parser with ``subcommand`` registered + parse argv."""
    parser = argparse.ArgumentParser(prog=f"bench campaign:{subcommand}")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    _SUBCOMMAND_PARSERS[subcommand](subparsers)
    return parser.parse_args([subcommand] + argv_for_run)


def _render_comparisons(
    output_paths: dict[str, list[Path]],
    plot_cfg: dict[str, Any],
) -> Path | None:
    """Render one comparison plot per benchmark group with ≥2 outputs."""
    label_tag = _str_or_none(plot_cfg.get("label_tag"))
    title = _str_or_none(plot_cfg.get("title"))

    last: Path | None = None
    for subcommand, paths in output_paths.items():
        if len(paths) < 2:
            continue
        if subcommand != "context-scaling":
            logger.warning(
                f"campaign: no comparison renderer registered for {subcommand!r}; "
                "skipping comparison plot"
            )
            continue
        out = paths[0].with_name(f"campaign_{subcommand}_compare.png")
        last = render_context_scaling(
            PlotInputs(
                results=paths,
                output=out,
                label_tag=label_tag,
                title=title,
            )
        )
        logger.info(f"campaign: wrote comparison plot {last}")
    return last


def _str_or_none(value: Any) -> str | None:  # type: ignore[reportAny]
    return value if isinstance(value, str) else None


__all__ = ["add_subparser", "run"]
