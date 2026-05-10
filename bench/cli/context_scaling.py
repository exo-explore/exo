"""Context-scaling benchmark — CLI subcommand.

Wraps :func:`bench.lib.context_scaling.run` with:

  - HF model-metadata resolution
  - Auto-derived constraints (memory, disk) and context ramp (Δ, K)
  - eco cluster + instance lifecycle (managed_cluster + managed_instance)
  - Cold-control isolation (delete sweep instance before controls)
  - JSON results + ``latest.json`` symlink under ``<output-dir>/context_scaling/``
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from exo_tools.cluster import EcoSession
from loguru import logger

from bench.cli._common import (
    SharedOptions,
    add_shared_args,
    get_arg,
    get_arg_optional,
)
from bench.lib import context_scaling
from bench.lib.cluster import managed_cluster, managed_instance
from bench.lib.context_scaling import (
    ContextScalingParams,
    make_cold_control_factory,
)
from bench.lib.model_meta import (
    ModelMeta,
    derive_cold_controls,
    derive_context_ramp,
    fetch_model_meta,
)
from bench.lib.results import ResultsBundle, RunMetadata, find_repo_root


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # type: ignore[type-arg]
) -> None:
    parser = subparsers.add_parser(
        "context-scaling",
        help="Prompt-TPS / decode-TPS vs context-size sweep",
        description=__doc__,
    )
    add_shared_args(parser)
    g = parser.add_argument_group("context-scaling")
    g.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="Number of equally-spaced PP points in the ramp (K).",
    )
    g.add_argument(
        "--pp-step",
        type=int,
        default=None,
        help="Δ (token step). If unset, derived from the model's max context.",
    )
    g.add_argument(
        "--fraction-of-max",
        type=float,
        default=1.0,
        help="When Δ is auto-derived, use this fraction of the model's "
        "max_position_embeddings as the ramp's upper bound (0 < f ≤ 1).",
    )
    g.add_argument(
        "--tg",
        type=int,
        default=64,
        help="Tokens to generate per step (decode duration; constant across ramp).",
    )
    g.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup requests at pp=Δ before the measured ramp. "
        "First warmup is cache-disabled (kernel JIT only); subsequent "
        "warmups are cache-enabled (the second is the one that primes "
        "the cache entry with a hot-kernel rate). Default 2 is the "
        "sweet spot: warmup=0 leaves JIT cost in step 0; warmup=1 has "
        "step 0 as a 'none' hit (still hot-kernel cold prefill, just "
        "classified differently).",
    )
    g.add_argument(
        "--cold-controls",
        type=str,
        default=None,
        help="Cold-control pp values to take after the cached sweep. Either "
        "'auto' (4 evenly-spaced points across the ramp) or a comma-separated "
        "list of explicit pp values (e.g. '8192,32768,65536'). "
        "Default: no cold controls.",
    )
    g.add_argument(
        "--sleep-between-s",
        type=float,
        default=1.0,
        help="Seconds to sleep between consecutive sweep requests.",
    )
    parser.set_defaults(handler=run)


# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Path:
    """Execute the context-scaling benchmark per the parsed args.

    Returns the path of the JSON results file.
    """
    shared = SharedOptions.from_namespace(args)
    repo_root = find_repo_root()

    # 1. Fetch HF metadata up-front; everything else can be derived from it.
    logger.info(f"fetching HuggingFace metadata for {shared.model}")
    meta = fetch_model_meta(shared.model)
    logger.info(
        f"  weights: {meta.total_weight_gb:.1f}GB; "
        f"max context: {meta.max_position_embeddings} tokens; "
        f"layers: {meta.num_hidden_layers}"
    )

    # 2. Derive constraints (user values always win; otherwise fall back to
    # ModelMeta heuristics for the *minimums*).
    min_memory_gb = (
        shared.min_memory_gb
        if shared.min_memory_gb is not None
        else meta.memory_constraint_gb
    )
    min_disk_gb = (
        shared.min_disk_gb
        if shared.min_disk_gb is not None
        else meta.disk_constraint_gb
    )
    logger.info(f"  cluster constraint: min memory {min_memory_gb:.1f}GB")
    logger.info(f"  cluster constraint: min disk {min_disk_gb:.1f}GB")
    if shared.max_memory_gb is not None:
        logger.info(f"  cluster constraint: max memory {shared.max_memory_gb:.1f}GB")
    if shared.max_disk_gb is not None:
        logger.info(f"  cluster constraint: max disk {shared.max_disk_gb:.1f}GB")

    explicit_pp_step = get_arg_optional(args, "pp_step", int)
    num_steps = get_arg(args, "num_steps", int)
    if explicit_pp_step is not None:
        pp_step = explicit_pp_step
    else:
        pp_step, num_steps = derive_context_ramp(
            meta,
            num_steps=num_steps,
            fraction_of_max=get_arg(args, "fraction_of_max", float),
        )
        logger.info(
            f"  derived ramp: Δ={pp_step} × K={num_steps} "
            f"= {pp_step * num_steps} tokens (max {meta.max_position_embeddings})"
        )

    cold_controls = _resolve_cold_controls(
        args, meta, pp_step=pp_step, num_steps=num_steps
    )
    if cold_controls:
        logger.info(f"  cold controls: {list(cold_controls)}")

    # 3. Spin up cluster + instance + run.
    eco = EcoSession(user_prefix=shared.user_prefix)
    output_dir = (shared.output_dir / "context_scaling").resolve()
    metadata = RunMetadata.new(
        benchmark="context_scaling",
        repo_root=repo_root,
        tags={**shared.tags, "host_pool": ",".join(shared.hosts) or "<auto>"},
    )
    bundle = ResultsBundle(metadata=metadata)

    with (
        managed_cluster(
            eco,
            hosts=list(shared.hosts) or None,
            count=shared.nodes,
            thunderbolt=shared.thunderbolt,
            chip=shared.chip,
            min_memory_gb=min_memory_gb,
            max_memory_gb=shared.max_memory_gb,
            min_disk_gb=min_disk_gb,
            max_disk_gb=shared.max_disk_gb,
        ) as cluster,
        managed_instance(
            cluster,
            eco,
            shared.model,
            sharding=shared.sharding,
            comm=shared.comm,
            min_nodes=shared.min_nodes,
            evict_downloads=shared.evict_downloads,
            cleanup_on_exit=shared.cleanup_instance,
        ) as session,
    ):
        params = ContextScalingParams(
            pp_step=pp_step,
            num_steps=num_steps,
            tg=get_arg(args, "tg", int),
            warmup=get_arg(args, "warmup", int),
            cold_controls=cold_controls,
            sleep_between_s=get_arg(args, "sleep_between_s", float),
        )
        factory = (
            make_cold_control_factory(
                session, shared.sharding, shared.comm, shared.min_nodes
            )
            if cold_controls
            else None
        )
        context_scaling.run(session, params, bundle, cold_control_factory=factory)

    out_path = bundle.write_json(output_dir)
    _update_latest_symlink(out_path)
    logger.info(f"wrote results → {out_path}")

    _validate_partial_hits(bundle)
    return out_path


# ---------------------------------------------------------------------------


def _resolve_cold_controls(
    args: argparse.Namespace,
    meta: ModelMeta,
    *,
    pp_step: int,
    num_steps: int,
) -> tuple[int, ...]:
    raw = get_arg_optional(args, "cold_controls", str)
    if raw is None or not raw.strip():
        return ()
    if raw.strip().lower() == "auto":
        return derive_cold_controls(meta, pp_step=pp_step, num_steps=num_steps, count=4)
    return tuple(int(s.strip()) for s in raw.split(",") if s.strip())


def _update_latest_symlink(out_path: Path) -> None:
    """Update ``<dir>/latest.json`` to point at the newly-written file."""
    link = out_path.parent / "latest.json"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(out_path.name, link)
    except OSError as e:
        logger.warning(f"could not update latest.json symlink: {e}")


def _validate_partial_hits(bundle: ResultsBundle) -> None:
    """Hard-fail if the cached sweep didn't see ``partial`` on every step ≥ 1.

    Step 0 is allowed to be ``exact`` (warmup primed the cache at pp=Δ); a
    later ``exact`` means Δ was effectively absorbed into the cache and the
    cold-rate measurement is meaningless. ``none`` means the cache was
    discarded mid-sweep and ``T_cum`` is unreliable.
    """
    cached = [r for r in bundle.runs if r.get("phase") == "cached_sweep"]
    bad = [r for r in cached[1:] if r.get("prefix_cache_hit") != "partial"]
    if bad:
        bad_summary = [(r["step_index"], r["prefix_cache_hit"]) for r in bad]
        raise RuntimeError(
            f"{len(bad)} cached-sweep step(s) reported "
            f"prefix_cache_hit != 'partial': {bad_summary!r}; "
            "T_cum is unreliable."
        )
