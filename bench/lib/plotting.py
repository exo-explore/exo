"""Typed matplotlib renderers for benchmark JSON results.

This module owns the *visualisation* of bench results, mirroring how
``bench/lib/<name>.py`` owns the methodology and ``bench/cli/<name>.py``
owns the orchestration. Adding plotting for a new benchmark = a new
``render_<name>`` function here + a dispatch entry in ``bench/cli/plot.py``.

Functions take typed inputs (``Path`` lists, options) and write a PNG.
They never touch argparse or stdout — that's the CLI's job.

matplotlib's type stubs are thin (most return values are ``Any``), so all
calls into ``pyplot`` are concentrated at the bottom of this file with
targeted ``# type: ignore[reportUnknownMemberType, reportAny]`` per line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Tab10 cycle from matplotlib's default; we pick colours by index ourselves
# instead of fishing them out of `Line2D.get_color()` so the strict-type
# fallout stays small and predictable.
_COLOR_CYCLE: tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
)


@dataclass(frozen=True)
class PlotInputs:
    """Inputs for any benchmark renderer.

    Attributes:
        results: One or more bench JSON files. The first is used to
            auto-derive the title when ``title`` is unset.
        output: Path to write the PNG to.
        label_tag: When set, use ``metadata.tags[label_tag]`` as the
            legend label for each run; otherwise use the run id.
        title: Override for the figure title.
    """

    results: list[Path]
    output: Path
    label_tag: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class _RunSeries:
    """Pre-extracted plot data for one results JSON.

    ``cached_prefill_seconds`` is the cumulative cold-prefill estimate
    (``T_cum`` from the methodology — read from ``derived.t_cum_seconds``).
    ``control_prefill_seconds`` is the actual cold prefill time per
    control (``pp_tokens / prompt_tps`` from the cold-control row).
    """

    label: str
    cached_pp: list[int]
    cached_prefill_seconds: list[float]
    cached_gen_tps: list[float]
    control_pp: list[int]
    control_prefill_seconds: list[float]


# ---------------------------------------------------------------------------
# Pure data extraction (strict-typed, no matplotlib)
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    """Read a bench JSON file and assert top-level shape."""
    with path.open() as f:
        loaded: Any = json.load(f)  # type: ignore[reportAny]
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return cast("dict[str, Any]", loaded)


# dict[str, Any].get(...) returns Any. The five _get_* helpers below
# concentrate the Any boundary so the rest of the module can be strict.


def _get_dict(d: dict[str, Any], key: str) -> dict[str, Any]:
    val: Any = d.get(key)
    return cast("dict[str, Any]", val) if isinstance(val, dict) else {}


def _get_list(d: dict[str, Any], key: str) -> list[Any]:
    val: Any = d.get(key)
    return cast("list[Any]", val) if isinstance(val, list) else []


def _get_str(d: dict[str, Any], key: str, default: str = "") -> str:
    val: Any = d.get(key, default)  # type: ignore[reportAny]
    return val if isinstance(val, str) else default


def _get_int(row: dict[str, Any], key: str) -> int:
    val: Any = row.get(key, 0)  # type: ignore[reportAny]
    if isinstance(val, bool):  # bool is int; reject explicitly
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return 0
    return 0


def _get_float(row: dict[str, Any], key: str) -> float:
    val: Any = row.get(key, 0.0)  # type: ignore[reportAny]
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def _label_for(data: dict[str, Any], label_tag: str | None) -> str:
    if label_tag is not None:
        tags = _get_dict(_get_dict(data, "metadata"), "tags")
        if label_tag in tags:
            return _get_str(tags, label_tag, "(unnamed)")
    return _get_str(_get_dict(data, "metadata"), "run_id", "(unnamed)")


def _extract_series(data: dict[str, Any], label: str) -> _RunSeries:
    """Pre-extract typed lists from a context-scaling bench JSON."""
    cached_pp: list[int] = []
    cached_gen_tps: list[float] = []
    for raw in _get_list(data, "runs"):  # type: ignore[reportAny]
        if not isinstance(raw, dict):
            continue
        row = cast("dict[str, Any]", raw)
        if _get_str(row, "phase") != "cached_sweep":
            continue
        cached_pp.append(_get_int(row, "pp_tokens"))
        cached_gen_tps.append(_get_float(row, "generation_tps"))

    # Cumulative cold-prefill estimate is computed in derive_summary and
    # written to derived.t_cum_seconds (parallel to the cached steps).
    derived = _get_dict(data, "derived")
    t_cum_raw = _get_list(derived, "t_cum_seconds")
    cached_prefill_seconds: list[float] = []
    for raw in t_cum_raw:  # type: ignore[reportAny]
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            cached_prefill_seconds.append(float(raw))

    # Cold controls give us the actual cold prefill time directly:
    # pp_tokens / prompt_tps. Skip rows with zero/missing prompt_tps.
    control_pp: list[int] = []
    control_prefill_seconds: list[float] = []
    for raw in _get_list(data, "cold_controls"):  # type: ignore[reportAny]
        if not isinstance(raw, dict):
            continue
        row = cast("dict[str, Any]", raw)
        pp = _get_int(row, "pp_tokens")
        tps = _get_float(row, "prompt_tps")
        if pp > 0 and tps > 0:
            control_pp.append(pp)
            control_prefill_seconds.append(pp / tps)

    return _RunSeries(
        label=label,
        cached_pp=cached_pp,
        cached_prefill_seconds=cached_prefill_seconds,
        cached_gen_tps=cached_gen_tps,
        control_pp=control_pp,
        control_prefill_seconds=control_prefill_seconds,
    )


def _auto_title(data: dict[str, Any]) -> str:
    metadata = _get_dict(data, "metadata")
    params = _get_dict(data, "params")
    model = (
        _get_str(params, "full_model_id") or _get_str(params, "model_id") or "(unknown)"
    )
    sha = _get_str(metadata, "exo_sha") or "(no-sha)"
    host = _get_str(metadata, "hostname") or "(no-host)"
    return f"{model}\n{sha} on {host}"


# ---------------------------------------------------------------------------
# Matplotlib boundary — each call site has a narrow, justified ignore.
# ---------------------------------------------------------------------------


def render_context_scaling(inputs: PlotInputs) -> Path:
    """Render a 2-panel context-scaling plot.

    Top:    pp_tokens vs prompt_tps (line per run; cold controls as 'x' scatter)
    Bottom: pp_tokens vs generation_tps (line per run)

    Each line is a separate result file. Multi-file mode is for comparing
    runs across exo SHAs / hosts / configs; the title is taken from the
    first file's metadata unless ``inputs.title`` is set.
    """
    if not inputs.results:
        raise ValueError("at least one results JSON path is required")

    # Validate + extract first so any data-shape error surfaces before we
    # even import matplotlib.
    first_data: dict[str, Any] | None = None
    series: list[_RunSeries] = []
    for path in inputs.results:
        data = _load(path)
        if first_data is None:
            first_data = data
        benchmark = _get_str(_get_dict(data, "metadata"), "benchmark")
        if benchmark != "context_scaling":
            raise ValueError(
                f"{path}: expected benchmark=='context_scaling', got {benchmark!r}"
            )
        series.append(_extract_series(data, _label_for(data, inputs.label_tag)))

    title = inputs.title
    if title is None and first_data is not None:
        title = _auto_title(first_data)
        if len(inputs.results) > 1:
            title = f"{title}\n(comparison of {len(inputs.results)} runs)"

    inputs.output.parent.mkdir(parents=True, exist_ok=True)
    _draw(series, inputs.output, title=title)
    return inputs.output


def _draw(series: list[_RunSeries], output: Path, *, title: str | None) -> None:
    """Concentrated matplotlib boundary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(  # type: ignore[reportUnknownMemberType]
        2, 1, figsize=(10, 8), sharex=True
    )
    top: Any = axes[0]  # type: ignore[reportAny]
    bottom: Any = axes[1]  # type: ignore[reportAny]

    for i, run in enumerate(series):
        color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        # Cumulative cold-prefill estimate (T_cum). Only plot points where
        # we have a t_cum value — skip if derived was empty for this run.
        n = min(len(run.cached_pp), len(run.cached_prefill_seconds))
        if n > 0:
            top.plot(  # type: ignore[reportAny, reportUnknownMemberType]
                run.cached_pp[:n],
                run.cached_prefill_seconds[:n],
                "-o",
                color=color,
                label=run.label,
            )
        if run.control_pp:
            top.scatter(  # type: ignore[reportAny, reportUnknownMemberType]
                run.control_pp,
                run.control_prefill_seconds,
                marker="x",
                s=80,
                color=color,
                label=f"{run.label} (cold one-shot)",
            )
        bottom.plot(  # type: ignore[reportAny, reportUnknownMemberType]
            run.cached_pp,
            run.cached_gen_tps,
            "-o",
            color=color,
            label=run.label,
        )

    top.set_ylabel("prefill time (s)")  # type: ignore[reportAny, reportUnknownMemberType]
    top.set_title(  # type: ignore[reportAny, reportUnknownMemberType]
        "cumulative cold-prefill time vs context size "
        "(line: T_cum estimate; ✕: cold one-shot control)"
    )
    top.grid(True, alpha=0.3)  # type: ignore[reportAny, reportUnknownMemberType]
    top.legend(loc="best", fontsize=8)  # type: ignore[reportAny, reportUnknownMemberType]

    bottom.set_xlabel("pp_tokens")  # type: ignore[reportAny, reportUnknownMemberType]
    bottom.set_ylabel("generation_tps (tok/s)")  # type: ignore[reportAny, reportUnknownMemberType]
    bottom.set_title("decode throughput vs context size")  # type: ignore[reportAny, reportUnknownMemberType]
    bottom.grid(True, alpha=0.3)  # type: ignore[reportAny, reportUnknownMemberType]

    if title is not None:
        fig.suptitle(title, fontsize=10)  # type: ignore[reportUnknownMemberType]

    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]
    plt.close(fig)
