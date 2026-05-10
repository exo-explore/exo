"""Prompt-TPS / decode-TPS vs context-size sweep.

Methodology (see also ``bench/METHODOLOGY.md``):

  Run a single ascending ramp of equally-spaced prompt lengths
  ``pp ∈ {Δ, 2Δ, …, K·Δ}`` with ``prefix_cache=enabled``, ``repeat=1``,
  ``concurrency=1`` and one warmup at ``pp=Δ``.

  Because each step's prefix is exactly what the previous step left in
  the cache, every step beyond the first is a *partial* hit and the
  server-reported ``prompt_tps`` reflects the true cold rate over the
  fresh ``Δ``-token suffix. We accept the warmup's reported rate as the
  cold equivalent for ``pp=Δ`` (the warmup itself is the cold prefill).

  ``decode TPS`` is independent of prefill mechanics — every step's
  ``generation_tps`` is a real decode-rate-at-N data point.

  Cumulative cold-prefill upper bound:
      ``T_cum(pp_k) = Σ_{i=1..k} (Δ_i / prompt_tps_i)``

  Optional cold-control points (``prefix_cache=disabled``) validate the
  approximation; the gap quantifies per-task overhead. To preserve the
  ``none`` cache-hit classification AND ensure the request actually
  hits a freshly-placed runner (the master picks the instance with the
  lowest in-flight task count, which is non-deterministic when multiple
  same-model instances exist), :func:`run` deletes the sweep instance
  *before* invoking the cold-control factory. The factory itself places
  a fresh instance per control and deletes it on exit; the
  :func:`bench.lib.cluster.managed_instance` ctx-manager calls
  ``cleanup_all_instances`` on exit as a final safety net.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from exo_tools.client import ExoClient, ExoHttpError
from exo_tools.harness import (
    Comm,
    Sharding,
    place_instance,
    wait_for_instance_gone,
)
from loguru import logger

from .completion import GenerationStats, PrefixCacheHit, run_one_completion
from .prompt import PromptSizer
from .results import ResultsBundle
from .session import BenchSession


@dataclass(frozen=True)
class ContextScalingParams:
    """Inputs for a single context-scaling sweep."""

    pp_step: int
    num_steps: int
    tg: int
    warmup: int = 1
    cold_controls: tuple[int, ...] = ()
    sleep_between_s: float = 1.0


@dataclass
class StepResult:
    pp_tokens: int
    delta_tokens: int
    prompt_tps: float
    generation_tps: float
    prefix_cache_hit: PrefixCacheHit | str
    prompt_tokens: int
    generation_tokens: int
    elapsed_s: float
    peak_memory_bytes: int = 0
    output_text_preview: str = ""


def _peak_bytes(stats: GenerationStats) -> int:
    pm = stats.get("peak_memory_usage") or {}
    return int(pm.get("inBytes") or pm.get("in_bytes") or 0)


def _build_step_result(
    pp_tokens: int,
    delta_tokens: int,
    elapsed_s: float,
    output_text_preview: str,
    stats: GenerationStats,
) -> StepResult:
    return StepResult(
        pp_tokens=pp_tokens,
        delta_tokens=delta_tokens,
        prompt_tps=float(stats.get("prompt_tps") or 0.0),
        generation_tps=float(stats.get("generation_tps") or 0.0),
        prefix_cache_hit=stats.get("prefix_cache_hit") or "unknown",
        prompt_tokens=int(stats.get("prompt_tokens") or pp_tokens),
        generation_tokens=int(stats.get("generation_tokens") or 0),
        elapsed_s=elapsed_s,
        peak_memory_bytes=_peak_bytes(stats),
        output_text_preview=output_text_preview[:200],
    )


def _run_request(
    client: ExoClient,
    full_model_id: str,
    pp: int,
    tg: int,
    sizer: PromptSizer,
    *,
    use_prefix_cache: bool,
) -> tuple[StepResult, int]:
    """Send one request and return ``(StepResult, actual_pp_tokens)``."""
    row, actual_pp = run_one_completion(
        client,
        full_model_id,
        pp,
        tg,
        sizer,
        use_prefix_cache=use_prefix_cache,
        stream=False,
    )
    step = _build_step_result(
        pp_tokens=actual_pp,
        delta_tokens=actual_pp,  # caller overrides for cached sweep
        elapsed_s=row["elapsed_s"],
        output_text_preview=row["output_text_preview"],
        stats=row["stats"],
    )
    return step, actual_pp


def _compute_t_cum(steps: list[StepResult]) -> list[float]:
    t_cum = 0.0
    out: list[float] = []
    for s in steps:
        if s.prompt_tps > 0 and s.delta_tokens > 0:
            t_cum += s.delta_tokens / s.prompt_tps
        out.append(round(t_cum, 6))
    return out


def run_cached_sweep(
    session: BenchSession,
    params: ContextScalingParams,
    bundle: ResultsBundle,
) -> list[StepResult]:
    """Run the ascending PP sweep with ``prefix_cache=enabled``.

    Mutates ``bundle.runs`` in place and returns the typed step list.
    """
    if session.full_model_id is None:
        raise RuntimeError(
            "BenchSession.full_model_id must be set for context-scaling."
        )

    sizer = session.get_prompt_sizer()
    client = session.client
    pp_targets = [params.pp_step * i for i in range(1, params.num_steps + 1)]
    logger.info(
        f"context-scaling: K={params.num_steps} steps, Δ={params.pp_step} tokens, "
        f"tg={params.tg}, warmup={params.warmup}, cached"
    )

    # Warmup discipline:
    #   - First warmup runs with the prefix cache DISABLED. This triggers
    #     the MLX kernel JIT compile + KV-buffer alloc for this exact
    #     (Δ, dtype, batch) shape, but does NOT write a cache entry — so
    #     the cold-with-JIT rate isn't fossilised.
    #   - Subsequent warmups run with the prefix cache ENABLED. The
    #     second one finds an empty cache, does a real cold prefill with
    #     a HOT kernel, and writes the resulting rate into the cache
    #     entry at pp=Δ.
    #   - Step 0 (also cache-enabled) is then an exact hit on that entry
    #     and reports the hot rate.
    # Default warmup=2 gives both effects; warmup=1 still does the JIT
    # warmup but leaves step 0 as a "none" hit (cold prefill at the hot
    # kernel, creates the cache entry on the way through).
    for w in range(params.warmup):
        is_jit_warmup = w == 0
        kind = "JIT warmup" if is_jit_warmup else "cache-prime warmup"
        logger.info(
            f"  warmup {w + 1}/{params.warmup} ({kind}, pp={params.pp_step})"
        )
        _run_request(
            client,
            session.full_model_id,
            params.pp_step,
            params.tg,
            sizer,
            use_prefix_cache=not is_jit_warmup,
        )

    steps: list[StepResult] = []
    prev_pp = 0
    for i, pp in enumerate(pp_targets):
        time.sleep(params.sleep_between_s)
        try:
            step, actual_pp = _run_request(
                client,
                session.full_model_id,
                pp,
                params.tg,
                sizer,
                use_prefix_cache=True,
            )
        except Exception as e:
            logger.error(f"step {i + 1}/{params.num_steps} (pp={pp}) failed: {e}")
            raise

        step.delta_tokens = actual_pp - prev_pp
        steps.append(step)
        bundle.runs.append({"step_index": i, "phase": "cached_sweep", **asdict(step)})
        logger.info(
            f"  step {i + 1}/{params.num_steps} pp={actual_pp} Δ={step.delta_tokens} "
            f"prompt_tps={step.prompt_tps:.1f} gen_tps={step.generation_tps:.2f} "
            f"hit={step.prefix_cache_hit}"
        )
        prev_pp = actual_pp

    return steps


def run_cold_controls(
    factory: Callable[[], AbstractContextManager[ExoClient]],
    session: BenchSession,
    params: ContextScalingParams,
    bundle: ResultsBundle,
) -> list[StepResult]:
    """Run cold-control points on a fresh instance to preserve ``none`` hits.

    A cold control is a single request at ``pp=N`` with
    ``prefix_cache=disabled``, executed against a freshly-placed instance
    (and with no other same-model instance live, so the master's task
    routing is deterministic). The caller is expected to delete the
    sweep instance before invoking this — see :func:`run`.
    """
    if not params.cold_controls:
        return []
    if session.full_model_id is None:
        raise RuntimeError("BenchSession.full_model_id must be set for cold controls.")

    sizer = session.get_prompt_sizer()
    out: list[StepResult] = []
    for control_pp in params.cold_controls:
        logger.info(f"cold control: pp={control_pp} (fresh instance, cache disabled)")
        with factory() as fresh_client:
            step, actual_pp = _run_request(
                fresh_client,
                session.full_model_id,
                control_pp,
                params.tg,
                sizer,
                use_prefix_cache=False,
            )
        step.delta_tokens = actual_pp
        out.append(step)
        bundle.cold_controls.append({"phase": "cold_control", **asdict(step)})
        logger.info(
            f"  cold pp={actual_pp} prompt_tps={step.prompt_tps:.1f} "
            f"gen_tps={step.generation_tps:.2f} hit={step.prefix_cache_hit}"
        )
        if step.prefix_cache_hit != "none":
            logger.warning(
                f"cold control at pp={actual_pp} reported "
                f"prefix_cache_hit={step.prefix_cache_hit!r}; "
                f"control may not be cold."
            )
    return out


def derive_summary(
    steps: list[StepResult],
    cold_controls: list[StepResult],
) -> dict[str, Any]:
    """Compute the cumulative cold-prefill upper bound + control gaps."""
    t_cum = _compute_t_cum(steps)
    bracketed = sorted(
        ((s.pp_tokens, t) for s, t in zip(steps, t_cum, strict=True)),
        key=lambda x: x[0],
    )
    control_gaps: list[dict[str, float]] = []
    for ctrl in cold_controls:
        cold_t = ctrl.pp_tokens / ctrl.prompt_tps if ctrl.prompt_tps > 0 else 0.0
        cum_t = _interp(bracketed, ctrl.pp_tokens)
        gap = cum_t - cold_t
        control_gaps.append(
            {
                "pp_tokens": ctrl.pp_tokens,
                "cold_t_seconds": round(cold_t, 4),
                "t_cum_seconds_at_pp": round(cum_t, 4),
                "gap_seconds": round(gap, 4),
                "gap_fraction": round(gap / cold_t, 4) if cold_t > 0 else 0.0,
            }
        )
    return {
        "t_cum_seconds": t_cum,
        "control_gaps": control_gaps,
    }


def _interp(points: list[tuple[int, float]], x: int) -> float:
    """Linear interpolate y at x, given sorted ``(x, y)`` points."""
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= x <= x1 and x1 != x0:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# ---------------------------------------------------------------------------
# Cold-control instance factory
# ---------------------------------------------------------------------------


def make_cold_control_factory(
    session: BenchSession,
    sharding: Sharding,
    comm: Comm,
    min_nodes: int,
    instance_timeout_s: float = 1800.0,
) -> Callable[[], AbstractContextManager[ExoClient]]:
    """Return a callable yielding a context manager that places a fresh instance.

    Each ``with factory() as client:`` block places a brand-new instance,
    yields its client, then deletes the instance on exit. Used to isolate
    cold-control runs.

    The caller is responsible for ensuring no other same-model instance is
    live during the ``with`` block — otherwise master routing is
    non-deterministic and the cold control may be served by a stale runner.
    See :func:`run` for the orchestration.
    """

    @contextmanager
    def factory() -> Iterator[ExoClient]:
        if session.full_model_id is None:
            raise RuntimeError("session.full_model_id is unset")
        client = session.client
        instance_id = place_instance(
            client,
            session.full_model_id,
            sharding=sharding,
            comm=comm,
            min_nodes=min_nodes,
            timeout=instance_timeout_s,
        )
        try:
            yield client
        finally:
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{instance_id}")
            with contextlib.suppress(Exception):
                wait_for_instance_gone(client, instance_id, timeout=60.0)

    return factory


def _delete_instance(client: ExoClient, instance_id: str) -> None:
    """Best-effort delete of a placed instance."""
    with contextlib.suppress(ExoHttpError):
        client.request_json("DELETE", f"/instance/{instance_id}")
    with contextlib.suppress(Exception):
        wait_for_instance_gone(client, instance_id, timeout=60.0)


def run(
    session: BenchSession,
    params: ContextScalingParams,
    bundle: ResultsBundle,
    *,
    cold_control_factory: Callable[[], AbstractContextManager[ExoClient]] | None = None,
) -> ResultsBundle:
    """End-to-end: cached sweep + optional cold controls + derived summary.

    To make cold controls truly isolated from the sweep instance, we
    delete the sweep instance *before* running the controls (otherwise
    the master might route a control's request to the stale sweep
    instance, since both match the same ``model_id``). The controls then
    each place their own fresh instance via the factory.
    """
    bundle.params.update(
        {
            "pp_step": params.pp_step,
            "num_steps": params.num_steps,
            "tg": params.tg,
            "warmup": params.warmup,
            "cold_controls": list(params.cold_controls),
            "sleep_between_s": params.sleep_between_s,
            "model_id": session.model_id,
            "full_model_id": session.full_model_id,
        }
    )
    bundle.capture_cluster(session.client)

    cached_steps = run_cached_sweep(session, params, bundle)

    cold_steps: list[StepResult] = []
    if params.cold_controls and cold_control_factory is not None:
        # Delete the sweep instance so the cold-control fresh instance is
        # the only same-model instance live for the duration of the controls.
        if session.instance_id is not None:
            logger.info(
                f"cold controls: deleting sweep instance {session.instance_id} "
                "to isolate fresh instance routing"
            )
            _delete_instance(session.client, session.instance_id)
            session.instance_id = None
        cold_steps = run_cold_controls(cold_control_factory, session, params, bundle)
    elif params.cold_controls and cold_control_factory is None:
        logger.warning(
            "Cold controls requested but no cold_control_factory supplied; skipping."
        )

    bundle.derived.update(derive_summary(cached_steps, cold_steps))
    return bundle
