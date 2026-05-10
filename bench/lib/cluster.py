"""Eco-managed cluster + instance lifecycle helpers for the bench CLI.

Two context managers:

- :func:`managed_cluster` deploys exo on the requested hosts (or via
  constraint-based reservation) and tears it down on exit.
- :func:`managed_instance` resolves the model on the cluster, optionally
  frees disk via ``--danger-delete-downloads`` (default on for benches),
  places the instance, and deletes it on exit.

The library never reaches for global state — every call takes an
explicit :class:`EcoSession`. Callers are expected to instantiate one
session per CLI invocation and use it across both context managers.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from exo_tools.client import ExoClient
from exo_tools.cluster import Chip, ClusterInfo, EcoSession, Thunderbolt
from exo_tools.harness import (
    Comm,
    Sharding,
    cleanup_all_instances,
    place_instance,
    resolve_model_short_id,
    run_planning_phase,
)
from loguru import logger

from .session import BenchSession


@contextmanager
def managed_cluster(
    eco: EcoSession,
    *,
    hosts: list[str] | None = None,
    count: int = 1,
    thunderbolt: Thunderbolt | None = None,
    chip: Chip | None = None,
    min_memory_gb: float | None = None,
    max_memory_gb: float | None = None,
    min_disk_gb: float | None = None,
    max_disk_gb: float | None = None,
    deploy_timeout_s: int = 600,
) -> Iterator[ClusterInfo]:
    """Deploy exo for the duration of the ``with`` block, then ``eco stop``.

    If ``hosts`` is given, deploys on exactly those hosts (constraint flags
    are ignored — eco doesn't re-validate the explicit list). Otherwise eco
    reserves any matching hosts that satisfy all of:

      - ``count`` (number of hosts)
      - ``thunderbolt`` topology (``A2A``, ``RING``, or ``NONE`` to
        exclude TB-connected hosts)
      - ``chip`` (substring match against eco's chip names)
      - memory bounds (``min_memory_gb`` / ``max_memory_gb``)
      - disk bounds (``min_disk_gb`` / ``max_disk_gb``)
    """
    if hosts:
        cluster = eco.start_deploy(
            hosts=hosts[:count],
            wait=True,
            timeout=deploy_timeout_s,
        )
    else:
        cluster = eco.start_deploy(
            count=count,
            thunderbolt=thunderbolt,
            chip=chip,
            min_memory_gb=min_memory_gb,
            max_memory_gb=max_memory_gb,
            min_disk_gb=min_disk_gb,
            max_disk_gb=max_disk_gb,
            wait=True,
            timeout=deploy_timeout_s,
        )
    logger.info(
        f"cluster deployed: {len(cluster.hosts)} host(s) "
        f"({', '.join(cluster.hosts)}); namespace={cluster.namespace}"
    )
    try:
        yield cluster
    finally:
        with contextlib.suppress(Exception):
            eco.stop(cluster.hosts)
        logger.info("cluster stopped")


@contextmanager
def managed_instance(
    cluster: ClusterInfo,
    eco: EcoSession,
    model_id: str,
    *,
    sharding: Sharding = Sharding.PIPELINE,
    comm: Comm = Comm.RING,
    min_nodes: int = 1,
    evict_downloads: bool = True,
    cleanup_on_exit: bool = True,
    instance_timeout_s: float = 7200.0,
    settle_timeout_s: float = 60.0,
) -> Iterator[BenchSession]:
    """Resolve the model on the cluster, place an instance, yield a session.

    Steps on entry:
      1. Resolve ``model_id`` to ``(short_id, full_id)`` against the cluster's
         ``/models`` endpoint (auto-adds from HuggingFace if missing).
      2. Run the harness's planning phase: validates each node has enough
         disk for the model and starts the download (or reuses an existing
         download). When ``evict_downloads=True`` (the default for benches),
         this also evicts smaller existing models if disk is short.
      3. Place the instance, wait for it to be ``RunnerReady``.
      4. Yield a :class:`BenchSession` pointing at the cluster's primary API.

    On exit: deletes the placed instance (and any other lingering
    instances) so the cluster is clean for the next benchmark.
    """
    client = cluster.make_client(timeout_s=instance_timeout_s)

    short_id, full_id = resolve_model_short_id(client, model_id, force_download=True)
    logger.info(f"resolved model: short_id={short_id} full_id={full_id}")

    # The planning phase needs a concrete preview (instance + runner-to-shard
    # mapping) to know which nodes to download to. Pull the placements API
    # directly and take the first valid one — bench cares about disk +
    # download, not the specific shard mapping.
    preview = _first_valid_preview(client, full_id, settle_timeout_s)
    if preview is None:
        raise RuntimeError(
            f"No placement available for {full_id} on cluster {cluster.hosts}"
        )

    duration = run_planning_phase(
        client,
        full_id,
        preview,
        danger_delete=evict_downloads,
        timeout=instance_timeout_s,
        settle_deadline=None,
    )
    if duration is not None:
        logger.info(f"download: {duration:.1f}s (freshly downloaded)")
    else:
        logger.info("download: model already cached on all nodes")

    instance_id = place_instance(
        client,
        model_id,
        sharding=sharding,
        comm=comm,
        min_nodes=min_nodes,
        timeout=instance_timeout_s,
    )
    logger.info(f"placed instance {instance_id} ({sharding.value}/{comm.value})")

    sess = BenchSession(
        cluster=cluster,
        eco=eco,
        instance_id=instance_id,
        model_id=short_id,
        full_model_id=full_id,
    )
    try:
        yield sess
    finally:
        if cleanup_on_exit:
            with contextlib.suppress(Exception):
                cleanup_all_instances(sess.client)
        else:
            logger.info(
                f"cleanup_on_exit=False: leaving instance(s) on {cluster.hosts}"
            )


def _first_valid_preview(
    client: ExoClient, full_model_id: str, settle_timeout_s: float
) -> dict[str, Any] | None:
    """Poll ``/instance/previews`` until at least one valid preview comes back."""
    deadline = time.monotonic() + settle_timeout_s
    backoff_s = 1.0
    while True:
        resp_obj: Any = client.request_json(  # type: ignore[reportAny]
            "GET", "/instance/previews", params={"model_id": full_model_id}
        )
        resp: dict[str, Any] = (
            cast("dict[str, Any]", resp_obj) if isinstance(resp_obj, dict) else {}
        )
        previews_raw: object = resp.get("previews") or []
        previews: list[Any] = (
            cast("list[Any]", previews_raw) if isinstance(previews_raw, list) else []
        )
        for raw in previews:  # type: ignore[reportAny]
            if not isinstance(raw, dict):
                continue
            entry = cast("dict[str, Any]", raw)
            if entry.get("error") is not None:
                continue
            instance = entry.get("instance")
            if isinstance(instance, dict):
                return entry
        if time.monotonic() >= deadline:
            return None
        logger.info(
            f"waiting for placement to appear for {full_model_id} "
            f"({deadline - time.monotonic():.0f}s remaining)..."
        )
        time.sleep(min(backoff_s, max(0.0, deadline - time.monotonic())))
        backoff_s = min(backoff_s * 2, 30.0)
