import errno
import ipaddress
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast, final

if TYPE_CHECKING:
    import socket as _socket_module

    from exo.worker.engines.mlx.vision import VisionProcessor

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path, resolve_existing_model
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    DrafterPlacement,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import (
    AsymmetricTensorShardMetadata,
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.asymmetric_parallel import (
    asymmetric_tensor_auto_parallel,
)
from exo.worker.engines.mlx.auto_parallel import (
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    DFlashHooksNotImplementedError as _DFlashHooksNotImplementedError,
)
from exo.worker.runner.bootstrap import logger


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    if isinstance(model_shard_meta, AsymmetricTensorShardMetadata):
        rank_weight_fraction = (
            model_shard_meta.ratio
            if model_shard_meta.device_rank == 0
            else 1.0 - model_shard_meta.ratio
        )
        return Memory.from_float_kb(
            (model_shard_meta.end_layer - model_shard_meta.start_layer)
            / model_shard_meta.n_layers
            * model_shard_meta.model_card.storage_size.in_kb
            * rank_weight_fraction
        )

    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def _bound_rank(bound_instance: BoundInstance) -> int:
    """Rank of this runner inside the parent ``mx.distributed`` group.

    Target ranks read this from their bound shard metadata; the drafter
    rank reads it from :class:`DrafterPlacement` since the drafter has
    no target shard.
    """
    if bound_instance.is_drafter_rank:
        placement = bound_instance.instance.drafter_placement
        assert placement is not None  # type narrowed by is_drafter_rank
        return placement.drafter_rank
    return bound_instance.bound_shard.device_rank


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """Initialize MLX distributed for this rank's parent group.

    The parent group spans every rank declared by the instance: target
    ranks plus, for asymmetric placement, the trailing drafter rank.
    Target ranks split off into a subgroup at runtime via
    :func:`initialize_mlx`; this helper just brings up the parent.
    """
    rank = _bound_rank(bound_instance)
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        group: mx.distributed.Group | None = None
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator

                max_jaccl_attempts = 8
                for attempt in range(1, max_jaccl_attempts + 1):
                    try:
                        group = mx.distributed.init(backend="jaccl", strict=True)
                        break
                    except (RuntimeError, ValueError) as exc:
                        if attempt == max_jaccl_attempts:
                            raise
                        backoff = min(2.0 * attempt, 10.0)
                        logger.warning(
                            f"rank {rank} JACCL init attempt {attempt}/{max_jaccl_attempts} "
                            f"failed ({exc}), retrying in {backoff:.0f}s"
                        )
                        time.sleep(backoff)

        logger.info(f"Rank {rank} mlx distributed initialization complete")
        if group is None:
            raise RuntimeError("MLX distributed initialization did not return a group")

        return group


@final
@dataclass(frozen=True)
class MlxGroupSplit:
    """Target-side view of an instance's distributed wiring.

    Pre-v3 the asymmetric drafter rank was a member of the parent
    ``mx.distributed`` group, and this struct carried the parent + a
    target-only subgroup. Under the v3+ wire the drafter is NOT in any
    ``mx.distributed.Group`` -- target ranks form their own group of
    size ``target_world_size`` and the drafter dials a TCP socket. The
    struct now carries:

      * ``parent`` / ``target_subgroup`` -- aliases for the same target
        group (``parent is target_subgroup`` always under v3). Both
        fields are retained so existing callers (builder.py, image
        builder, generate.py) keep working without rev. ``None`` when
        the target world size is 1 (the well-known "single rank, no
        collectives needed" signal that
        :func:`load_mlx_items`, :func:`mx_barrier`, :func:`mx_any`
        already short-circuit on).
      * ``drafter_socket`` -- the connected TCP socket between target
        rank 0 and the drafter rank. Set ONLY on target rank 0 of an
        asymmetric placement; ``None`` for any other rank.
      * ``drafter_rank_in_parent`` -- advisory placement index of the
        drafter (``placement.drafter_rank``). Carried for telemetry
        and the few legacy call sites that branch on "is asymmetric";
        ``None`` for symmetric placement.
      * ``target_peer_fanout`` -- inter-target-rank TCP fanout for
        spec-decode int broadcasts (see :class:`TargetPeerFanout`).
        ``None`` for single-target instances or symmetric placements
        without a drafter (no spec-decode hot path; legacy
        ``mx_broadcast_int_list`` is sufficient).
    """

    parent: mx.distributed.Group | None
    target_subgroup: mx.distributed.Group | None
    drafter_rank_in_parent: int | None
    drafter_socket: object | None = None
    """Connected ``socket.socket`` from target rank 0 to the drafter.

    Typed as ``object`` to keep the dataclass importable from modules
    that don't import ``socket`` directly. Runtime callers
    (:mod:`builder`) cast back to ``socket.socket`` before passing to
    :func:`make_remote_transport`."""

    target_peer_fanout: "TargetPeerFanout | None" = None
    """Inter-target-rank TCP fanout for spec-decode int broadcasts.

    Allocated alongside the drafter socket on multi-target asymmetric
    placements. ``None`` for single-target or symmetric instances.
    Built once at bootstrap; the spec-decode loop reuses it for every
    round."""

    @property
    def is_asymmetric(self) -> bool:
        return self.drafter_rank_in_parent is not None


@final
@dataclass(frozen=True)
class TargetPeerFanout:
    """Direct TCP int-broadcast wire between target rank 0 and its peers.

    Replaces :func:`mx.distributed.send` / :func:`recv` on the
    spec-decode hot path. JACCL on Apple Silicon conflates int32
    broadcasts on the target group with the model's float32 TP
    ``all_sum`` collectives; the former occasionally returns the
    latter's logit memory reinterpreted as int32, surfacing as
    out-of-vocab token ids (~``10^9``) deep in the SPM detokenizer.

    The model's TP ``all_sum`` collectives stay on JACCL/RDMA -- they
    carry multi-MB tensor reductions where vendor RDMA wins
    decisively. Only the tiny (~24-byte) int32 broadcasts move to TCP,
    where Thunderbolt with ``TCP_NODELAY`` adds <100µs per round
    (negligible against a ~30ms verifier forward).

    Topology:
      * On target rank 0: ``peer_sockets`` holds one connection per
        non-zero peer rank, indexed by peer rank.
      * On a peer target rank (rank > 0): ``rank_zero_socket`` holds
        the single connection back to rank 0.

    Both shapes are produced by :func:`_setup_target_peer_fanout` at
    instance bootstrap and are immutable for the runner's lifetime.
    Reconnect-on-failure is intentionally NOT supported: a transport
    failure on this wire is treated as a hard runner failure (same as
    a TP all-reduce failure) and the supervisor rebuilds the instance.
    """

    rank: int
    """Caller's target rank inside the parent group; matches
    ``MlxGroupSplit.parent.rank()`` when ``parent`` is set."""

    peer_sockets: dict[int, object] = field(default_factory=dict)
    """Rank 0 only: ``{peer_rank: socket.socket}``. Empty on rank > 0."""

    rank_zero_socket: object | None = None
    """Rank > 0 only: connected socket back to rank 0. ``None`` on rank 0."""

    expected_world_size: int = 1
    """Target world size (every rank in the fanout sees the same value).

    Stored explicitly so the broadcast helpers can sanity-check that
    rank 0's ``peer_sockets`` cover all peers without re-deriving the
    world size from a possibly-discarded group handle."""


def initialize_mlx(bound_instance: BoundInstance) -> MlxGroupSplit:
    """Bring up the target ``mx.distributed`` group + (rank 0) drafter socket.

    Target ranks: initialise an ``mx.distributed.Group`` of size
    ``parent_group_size`` (which under v3+ equals the number of target
    shards -- the drafter is NOT a member of this group). Single-target
    instances (``parent_group_size == 1``) short-circuit and return a
    split with ``parent / target_subgroup = None``.

    Target rank 0 of an asymmetric placement additionally binds a TCP
    listener on ``DrafterPlacement.drafter_socket_port`` and accepts
    the drafter's incoming connection. The connected socket flows
    through :class:`MlxGroupSplit.drafter_socket` to the builder, which
    hands it to :func:`make_remote_transport`.

    The drafter rank does NOT call this function; its bootstrap
    (:class:`DrafterRunner._handle_connect`) dials the socket directly
    without touching ``mx.distributed`` at all.
    """
    assert not bound_instance.is_drafter_rank, (
        "initialize_mlx should not be called on a drafter rank under "
        "the v3+ asymmetric wire; DrafterRunner._handle_connect dials "
        "the drafter socket directly without joining mx.distributed."
    )
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    target_world_size = bound_instance.instance.parent_group_size
    placement = bound_instance.instance.drafter_placement

    # Single-target instance: no mx.distributed group needed (other
    # ranks short-circuit on the ``group is None`` signal). Drafter
    # wire still exists for asymmetric placement.
    parent: mx.distributed.Group | None = (
        None if target_world_size <= 1 else mlx_distributed_init(bound_instance)
    )

    drafter_rank_in_parent = placement.drafter_rank if placement is not None else None

    drafter_socket = _maybe_accept_drafter_socket(
        bound_instance=bound_instance,
        target_world_size=target_world_size,
        placement=placement,
    )

    target_peer_fanout = _maybe_setup_target_peer_fanout(
        bound_instance=bound_instance,
        target_world_size=target_world_size,
        placement=placement,
    )

    return MlxGroupSplit(
        parent=parent,
        target_subgroup=parent,
        drafter_rank_in_parent=drafter_rank_in_parent,
        drafter_socket=drafter_socket,
        target_peer_fanout=target_peer_fanout,
    )


def _maybe_accept_drafter_socket(
    *,
    bound_instance: BoundInstance,
    target_world_size: int,
    placement: object,
) -> object | None:
    """Bind + accept the drafter dial on target rank 0; otherwise return ``None``.

    Only target rank 0 of an asymmetric placement owns the drafter
    wire. Other target ranks (rank >= 1) and symmetric placements
    return ``None``. The caller embeds the result in
    :class:`MlxGroupSplit.drafter_socket`.

    The accept call is sequential after :func:`mlx_distributed_init`
    in the parent function. The drafter's :func:`dial_target` retries
    with backoff for up to two minutes, which comfortably covers the
    target group's bootstrap latency. If accept times out (drafter
    unreachable / crashed), this raises :class:`socket.timeout`; the
    runner surface bubbles it up as a connect-task failure so the
    cluster doesn't sit silently wedged.
    """
    from exo.shared.types.worker.instances import DrafterPlacement

    if placement is None:
        return None
    if not isinstance(placement, DrafterPlacement):
        raise TypeError(
            f"drafter_placement must be DrafterPlacement, got {type(placement)!r}"
        )
    # Target rank 0 binds; other target ranks no-op. Symmetric placements
    # land in the ``placement is None`` branch above.
    if bound_instance.parent_rank != 0:
        return None
    del target_world_size  # not needed once we know we're rank 0
    # Imported lazily to avoid pulling the socket transport into module
    # import unless this code path is exercised.
    from exo.worker.engines.mlx.generator.drafter_socket import (
        accept_drafter,
        bind_target_listener,
    )

    # Bind to all interfaces so the drafter can dial whichever address
    # ``DrafterPlacement.drafter_socket_host`` resolves to (LAN,
    # Thunderbolt-bridge, Tailscale, etc.). The placement-time IP only
    # serves as the address the drafter dials; target rank 0 doesn't
    # need to advertise a specific bind address.
    #
    # Codex P2 (PR #20 round-(N+9), drafter_socket.py:106): pre-fix the
    # listener was hard-coded to ``AF_INET``/``0.0.0.0``, so an IPv6
    # advertised host (Tailscale ULA, link-local IPv6, IPv6-only LAN)
    # could never accept the drafter's dial. Pick the wildcard whose
    # family matches the advertised host: ``::`` for IPv6 (with
    # IPV6_V6ONLY=0 inside ``bind_target_listener`` so IPv4-mapped
    # connects still land), ``0.0.0.0`` for IPv4 or unparseable
    # hostnames.
    advertised = placement.drafter_socket_host
    try:
        parsed = ipaddress.ip_address(advertised)
        bind_host = "::" if isinstance(parsed, ipaddress.IPv6Address) else "0.0.0.0"
    except ValueError:
        # ``find_ip_prioritised`` should always return an IP literal,
        # but defensively handle a hostname by binding to the IPv6
        # wildcard (dual-stack via IPV6_V6ONLY=0). If IPv6 is not
        # available on the host, ``bind_target_listener`` will raise
        # and the failure is loud rather than silent.
        bind_host = "::"
    listener = _bind_drafter_listener_same_port_retry(
        bind_host=bind_host,
        bind_target_listener=bind_target_listener,
        port=placement.drafter_socket_port,
        advertised_host=placement.drafter_socket_host,
    )
    try:
        logger.info(
            f"target rank 0 listening for drafter on "
            f"{bind_host}:{listener.getsockname()[1]} "
            f"(advertised {placement.drafter_socket_host}:"
            f"{placement.drafter_socket_port})"
        )
        conn = accept_drafter(listener, timeout_seconds=180.0)
        logger.info("target rank 0 accepted drafter connection")
        return conn
    finally:
        # Listener is single-shot (drafter dials once and stays
        # connected for the instance lifetime); close it as soon as
        # accept returns to free the port.
        listener.close()


_DRAFTER_BIND_RETRY_BUDGET: Final[int] = 8
"""Number of bind attempts tolerated before giving up on the drafter listener.

Codex P1.2 (PR #20, round-2 fix): the master allocates
``drafter_socket_port`` via :func:`exo.utils.ports.random_ephemeral_port`,
which kernel-vets the port on the master's host. In cross-host deploys
the master cannot vet the target's port allocations, so
``bind_target_listener`` may still hit ``EADDRINUSE``; the most common
cause is a TIME_WAIT residue from a previous instance on the same port,
which clears within ~100 ms. Eight same-port retries with brief sleeps
absorb that without breaking the placement contract (the drafter is
told to dial ``placement.drafter_socket_port`` and retry must keep
listening on that exact port).
"""

_DRAFTER_BIND_RETRY_SLEEP_SECONDS: Final[float] = 0.1


def _bind_drafter_listener_same_port_retry(
    *,
    bind_host: str,
    bind_target_listener: Callable[[str, int], "_socket_module.socket"],
    port: int,
    advertised_host: str,
) -> "_socket_module.socket":
    """Bind the drafter listener on ``port``, retrying transient EADDRINUSE.

    Round-1 (Codex P1.2 PR #20) attempted to re-roll the port on
    ``EADDRINUSE``, but that broke the placement contract: the drafter
    dials ``DrafterPlacement.drafter_socket_port`` (master-announced),
    so a re-rolled listener accepts on a port the drafter never tries
    and the connection stalls until ``accept_drafter``'s 180 s timeout
    (Codex P1, round-2). We instead retry the SAME port with short
    backoff: a TIME_WAIT residue from a previous generator on the same
    port (the realistic ``EADDRINUSE`` case in cross-host deploys)
    clears within ~100 ms, and persistent collisions surface a clean
    ``EADDRINUSE`` to the runner so the master can re-place with a
    new port.

    Non-``EADDRINUSE`` ``OSError`` (Codex P2 round-2: e.g.
    ``EAFNOSUPPORT`` for an IPv6 wildcard on an IPv4-only host,
    ``EACCES`` for a privileged port) is surfaced immediately so the
    operator sees the actual root cause instead of a misleading
    "port range exhausted" message after the retry budget.
    """
    last_error: OSError | None = None
    for attempt in range(1, _DRAFTER_BIND_RETRY_BUDGET + 1):
        try:
            return bind_target_listener(bind_host, port)
        except OSError as bind_error:
            if bind_error.errno != errno.EADDRINUSE:
                # Non-collision error: surface immediately. Retrying an
                # ``EAFNOSUPPORT`` or ``EACCES`` would just hide the
                # root cause behind a misleading retry log.
                raise
            last_error = bind_error
            if attempt >= _DRAFTER_BIND_RETRY_BUDGET:
                break
            logger.warning(
                f"bind_target_listener({bind_host}, {port}) raised "
                f"{bind_error!r} (attempt {attempt}/"
                f"{_DRAFTER_BIND_RETRY_BUDGET}, advertised host "
                f"{advertised_host}); retrying same port after "
                f"{_DRAFTER_BIND_RETRY_SLEEP_SECONDS}s"
            )
            time.sleep(_DRAFTER_BIND_RETRY_SLEEP_SECONDS)
    raise OSError(
        last_error.errno if last_error is not None else errno.EADDRINUSE,
        f"failed to bind drafter listener on {bind_host}:{port} after "
        f"{_DRAFTER_BIND_RETRY_BUDGET} same-port retries (last error: "
        f"{last_error!r}). The placement-announced port is held by "
        f"another process on this host; re-place the instance to "
        f"draw a fresh port.",
    ) from last_error


def _maybe_setup_target_peer_fanout(
    *,
    bound_instance: BoundInstance,
    target_world_size: int,
    placement: object,
) -> TargetPeerFanout | None:
    """Bring up the inter-target-rank TCP int-broadcast wire.

    Multi-target asymmetric placements need a TCP fanout between
    target rank 0 and its peers because the JACCL backend conflates
    the model's float32 TP ``all_sum`` with int32 broadcasts on the
    same group (see :class:`TargetPeerFanout` docstring). Single-rank
    targets and symmetric placements (no drafter) have no spec-decode
    hot path, so they don't need this wire and the function returns
    ``None``.

    Bootstrap protocol:

      * Target rank 0 binds 0.0.0.0:``placement.target_peer_socket_port``
        and accepts ``target_world_size - 1`` incoming connections.
      * Each non-zero target rank dials
        ``placement.target_peer_hosts_by_rank[my_rank]:target_peer_socket_port``
        with bounded retry (the listener may not be up yet on the
        first attempt because ``accept`` and ``connect`` race during
        bootstrap).

    The drafter rank is NOT in this fanout: it has its own dedicated
    wire to target rank 0 (see :func:`_maybe_accept_drafter_socket`).
    Skipping the fanout for the drafter rank is the right call
    because the drafter never broadcasts int frames to target peers
    -- it only exchanges drafts/verify with rank 0.

    Failure mode: a dial timeout / accept timeout raises
    :class:`ConnectionError` or :class:`socket.timeout`, which
    bubbles up to the runner and surfaces as a connect-task failure.
    The cluster does not silently wedge.
    """
    from exo.shared.types.worker.instances import DrafterPlacement

    if placement is None or not isinstance(placement, DrafterPlacement):
        return None
    if target_world_size <= 1:
        return None
    if bound_instance.is_drafter_rank:
        return None
    # Codex P1 (PR #21 round-(N+9), instances.py:97):
    # ``target_peer_socket_port`` is optional for wire-schema
    # compatibility with pre-fanout placements (rolling upgrades,
    # replayed historical events). When the field is absent we cannot
    # bind a fanout listener, so degrade gracefully to the legacy
    # behavior: no peer wire, no spec-decode int broadcasts. Multi-rank
    # asymmetric instances produced by current placement always include
    # the port, so this branch only fires for legacy payloads.
    if placement.target_peer_socket_port is None:
        logger.warning(
            "DrafterPlacement.target_peer_socket_port is unset (legacy "
            "or rolling-upgrade payload); skipping target-peer fanout. "
            "Spec-decode int broadcasts will fall back to the parent "
            "mx.distributed group, which is bandwidth-suboptimal but "
            "functionally correct."
        )
        return None

    rank = bound_instance.parent_rank
    expected_world_size = target_world_size
    target_peer_socket_port = placement.target_peer_socket_port

    # Imported lazily to avoid pulling the socket module into module
    # import for runners that never reach this code path.
    from exo.worker.engines.mlx.generator.target_peer_socket import (
        accept_target_peers,
        bind_target_peer_listener,
        dial_target_zero,
    )

    if rank == 0:
        listener = bind_target_peer_listener(
            "0.0.0.0",
            target_peer_socket_port,
            backlog=expected_world_size - 1,
        )
        try:
            logger.info(
                f"target rank 0 listening for {expected_world_size - 1} "
                f"target peers on 0.0.0.0:{target_peer_socket_port}"
            )
            conns = accept_target_peers(
                listener,
                expected_peers=expected_world_size - 1,
                timeout_seconds=180.0,
            )
            logger.info(
                f"target rank 0 accepted {len(conns)} target-peer connection(s)"
            )
        finally:
            listener.close()
        # The peer rank that wrote each connection is implicit (we
        # accept in connection order, but peers can dial in any
        # order). Spec-decode broadcasts don't need rank-indexed
        # peers -- rank 0 sends the same payload to every peer per
        # round -- so we store sockets in arbitrary stable order
        # keyed by accept order. The spec-decode broadcast helper
        # iterates ``peer_sockets.values()`` and ignores keys.
        peer_sockets: dict[int, object] = {idx: c for idx, c in enumerate(conns)}
        return TargetPeerFanout(
            rank=0,
            peer_sockets=peer_sockets,
            rank_zero_socket=None,
            expected_world_size=expected_world_size,
        )

    rank_zero_host = placement.target_peer_hosts_by_rank.get(str(rank))
    if rank_zero_host is None:
        raise RuntimeError(
            f"target peer rank {rank} (key={str(rank)!r}) has no entry "
            f"in DrafterPlacement.target_peer_hosts_by_rank "
            f"({placement.target_peer_hosts_by_rank}); placement is "
            "malformed"
        )
    logger.info(
        f"target peer rank {rank} dialing target rank 0 at "
        f"{rank_zero_host}:{target_peer_socket_port}"
    )
    conn = dial_target_zero(
        rank_zero_host,
        target_peer_socket_port,
        total_timeout_seconds=180.0,
    )
    logger.info(f"target peer rank {rank} connected to target rank 0")
    return TargetPeerFanout(
        rank=rank,
        peer_sockets={},
        rank_zero_socket=conn,
        expected_world_size=expected_world_size,
    )


EXO_DISABLE_DRAFTER_ENV = "EXO_DISABLE_DRAFTER"
EXO_DRAFTER_PREFERENCE_ENV = "EXO_DRAFTER_PREFERENCE"

# Allowed values for ``EXO_DRAFTER_PREFERENCE``. ``fastest`` picks the first
# drafter declared on the card (smallest by convention); ``highest_acceptance``
# picks the last (largest by convention); ``auto`` defaults to ``fastest`` but
# may be tuned by future heuristics (e.g. observed acceptance rate).
_DRAFTER_PREFERENCE_VALUES: frozenset[str] = frozenset(
    {"fastest", "highest_acceptance", "auto"}
)


def _drafter_disabled_by_env() -> bool:
    return os.environ.get(EXO_DISABLE_DRAFTER_ENV, "").lower() in {"1", "true", "yes"}


def _drafter_preference() -> str:
    raw = os.environ.get(EXO_DRAFTER_PREFERENCE_ENV, "auto").lower()
    if raw not in _DRAFTER_PREFERENCE_VALUES:
        logger.warning(
            f"Unknown {EXO_DRAFTER_PREFERENCE_ENV}={raw!r}, falling back to 'auto'"
        )
        return "auto"
    return raw


# Drafter kinds the loader recognises. ``"standard"`` is the existing
# external-drafter path (independent sibling LM via mlx-lm). ``"mtp"`` and
# ``"dflash"`` are the coupled-drafter kinds shipped by mlx-vlm 0.5+ that
# attach to the target architecturally (consume the target's hidden state /
# KV cache every draft step) and only run on single-node placements.
CoupledDrafterKind = Literal["mtp", "dflash"]
_KNOWN_COUPLED_DRAFTER_KINDS: Final[frozenset[CoupledDrafterKind]] = frozenset(
    {"mtp", "dflash"}
)


@final
@dataclass(frozen=True, kw_only=True)
class CoupledDrafter:
    """A loaded MTP/DFlash-kind coupled drafter, ready for the generator.

    Coupled drafters consume the target's hidden state every draft step and
    (for ``kind="mtp"``) read the target's KV cache directly via
    ``set_shared_kv``. They cannot decode independently the way standard
    external drafters can, so this loader path runs only when the placement
    collocates target + drafter on the same node (i.e. the target is not
    asymmetrically split via ``DrafterPlacement`` and the runner is loading
    both halves locally).

    The model object is typed ``object`` because the concrete class
    (``Gemma4AssistantDraftModel`` for ``mtp``, ``DFlashDraftModel`` for
    ``dflash``) lives in mlx-vlm and importing it in the worker hot path
    would force every linux/CPU build to drag mlx-vlm into the type
    surface. Generator-side dispatch narrows the type at the use site.
    """

    model_id: ModelId
    kind: CoupledDrafterKind
    model: object


# Exceptions :func:`_dispatch_attach_coupled_hooks` may raise that the
# loader caller should treat as "drafter loaded but not dispatchable on
# this target -- degrade to standard drafting" rather than crashes:
#
# - :class:`TypeError` -- right kind, wrong target architecture (e.g.
#   card declared a ``coupled_drafter`` of kind ``"mtp"`` but the target
#   loaded as something other than a Gemma 4 ``Model``).
# - :class:`exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks.DFlashHooksNotImplementedError`
#   -- right kind, hooks not yet vendored for that kind. Today raised by
#   the dflash skeleton; deletion follows the qwen3_5 vendor work.
#
# Listed at module scope (rather than caught inline) so the exception
# tuple stays a single source of truth -- adding a future coupled-drafter
# kind extends the tuple here once and the loader picks it up automatically.
# ``_DFlashHooksNotImplementedError`` is imported at the top of the file
# alongside other vendor imports so ruff E402 stays happy.
_COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = (
    TypeError,
    _DFlashHooksNotImplementedError,
)


def _dispatch_attach_coupled_hooks(kind: CoupledDrafterKind, model: object) -> None:
    """Mark ``model`` as wired for ``kind``'s coupled-drafter hooks.

    Per-kind dispatcher around the vendor modules' ``attach_*_hooks``
    helpers. Splitting the dispatch out of the load path lets the
    loader stay kind-agnostic -- adding a new coupled-drafter kind
    only requires extending this match plus the vendor module, not
    touching :func:`load_mlx_items`.

    Raises:
        TypeError: ``model`` is the wrong target architecture for
            the declared ``kind``. Caller falls back to standard
            drafting (see :data:`_COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS`).
        DFlashHooksNotImplementedError: ``kind == "dflash"`` and the
            qwen3_5 hook surface is still a skeleton. Same fallback.
    """
    match kind:
        case "mtp":
            from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
                attach_mtp_hooks,
            )

            attach_mtp_hooks(model)
        case "dflash":
            from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
                attach_dflash_hooks,
            )

            attach_dflash_hooks(model)


def _coupled_drafter_weight_size_bytes(coupled_id: ModelId) -> int:
    """Best-effort coupled-drafter on-disk size for the wired-memory bump.

    Mirrors :func:`_drafter_weight_size_bytes`: walk the drafter directory
    and sum file sizes; return 0 on any error. Coupled drafters are tiny
    (~158MB for the Gemma 4 E2B assistant) so under-wiring here is cheap
    even if the helper falls through; we just want a reasonable hint to
    ``set_wired_limit_for_model`` so the OS doesn't page the drafter
    weights out between requests.
    """
    drafter_path = resolve_existing_model(coupled_id)
    if drafter_path is None:
        return 0
    try:
        return sum(p.stat().st_size for p in drafter_path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _try_load_coupled_drafter(model_card: ModelCard) -> CoupledDrafter | None:
    """Attempt to load the coupled drafter declared on ``model_card``.

    Returns the loaded drafter on success, or ``None`` when:
    - the card declares no ``coupled_drafter``,
    - ``EXO_DISABLE_DRAFTER`` is set,
    - mlx-vlm is unavailable (e.g. linux build without the speculative
      drafters extra) or too old to expose ``load_drafter``,
    - the drafter's weights are not on disk,
    - mlx-vlm resolves an unknown / unsupported drafter kind, or
    - the load itself raises.

    Failures are logged at warning level and swallowed so that single-node
    deployments degrade to the standard external-drafter list (or to plain
    decoding) instead of crashing the runner. The caller is responsible
    for that fallback.
    """
    coupled_id = model_card.coupled_drafter
    if coupled_id is None:
        return None
    if _drafter_disabled_by_env():
        logger.info(
            f"Coupled drafter declared by {model_card.model_id} but "
            f"{EXO_DISABLE_DRAFTER_ENV} is set; skipping coupled drafter load."
        )
        return None

    # mlx-vlm's speculative-drafter API is partially typed (its
    # ``load_drafter`` signature uses ``**kwargs`` with no annotation),
    # so we cast at the import boundary to give the rest of this
    # function a well-typed surface. ``KNOWN_DRAFTER_KINDS`` is an
    # iterable of upstream kind strings -- declared as ``Iterable[str]``
    # because mlx-vlm uses ``frozenset[str]`` today but a future
    # release could swap it for a list without breaking us.
    #
    # Codex P2 (PR #23 round-(N+0), utils_mlx.py:809): we also catch
    # ``AttributeError`` so a partial / mismatched mlx-vlm install (the
    # ``speculative`` package imports cleanly but is missing
    # ``load_drafter`` / ``KNOWN_DRAFTER_KINDS`` -- e.g. an old release
    # with the namespace package but pre-drafter API, or a future
    # release that renames the symbols) degrades to the standard
    # drafter path instead of crashing the runner.
    try:
        from mlx_vlm.speculative import (  # pyright: ignore[reportMissingTypeStubs]
            drafters as _mlxvlm_drafters,
        )

        load_drafter = cast(
            Callable[..., tuple[object, str]],
            _mlxvlm_drafters.load_drafter,
        )
        known_drafter_kinds = cast(
            "Iterable[str]",
            _mlxvlm_drafters.KNOWN_DRAFTER_KINDS,
        )
    except (ImportError, AttributeError) as exc:
        logger.warning(
            f"Coupled drafter declared by {model_card.model_id} requires "
            f"mlx-vlm with speculative-drafter support (>=0.5.0) exposing "
            f"``load_drafter`` and ``KNOWN_DRAFTER_KINDS``, but resolving "
            f"those symbols failed ({type(exc).__name__}: {exc}); falling "
            f"back to the standard drafter path."
        )
        return None

    drafter_path = resolve_existing_model(coupled_id)
    if drafter_path is None:
        logger.warning(
            f"Coupled drafter {coupled_id} declared by {model_card.model_id} "
            "is not downloaded; pre-download it to enable coupled "
            "speculative decoding. Falling back to the standard drafter "
            "path for this load."
        )
        return None

    drafter_start = time.perf_counter()
    try:
        loaded_model, resolved_kind = load_drafter(str(drafter_path), kind=None)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            f"Failed to load coupled drafter {coupled_id} via mlx-vlm; "
            "falling back to the standard drafter path."
        )
        return None

    if resolved_kind not in _KNOWN_COUPLED_DRAFTER_KINDS:
        # mlx-vlm may evolve to recognise more kinds before exo's loader
        # learns to dispatch them; refuse rather than load a model the
        # generator cannot drive.
        known_upstream: list[str] = sorted(known_drafter_kinds)
        logger.warning(
            f"Coupled drafter {coupled_id} resolved to kind "
            f"{resolved_kind!r}, which exo's generator does not yet "
            f"support (known kinds: {sorted(_KNOWN_COUPLED_DRAFTER_KINDS)}; "
            f"mlx-vlm reports: {known_upstream}). Falling "
            "back to the standard drafter path."
        )
        return None

    logger.info(
        f"Loaded coupled drafter {coupled_id} (kind={resolved_kind!r}) "
        f"for {model_card.model_id} in "
        f"{(time.perf_counter() - drafter_start):.2f}s"
    )
    return CoupledDrafter(
        model_id=coupled_id,
        kind=resolved_kind,
        model=loaded_model,
    )


def _select_drafter_id(candidates: list[ModelId], preference: str) -> ModelId | None:
    """Pick a drafter id from a card's preference-ordered list.

    The card lists drafters in `[fastest, ..., highest_acceptance]` order. We
    prefer drafters that are already on disk (so the chooser doesn't force a
    surprise download); within the on-disk subset we honor the user's
    preference. If nothing is on disk we fall back to the head of the list,
    leaving the loader to log a "weights missing" warning.
    """
    if not candidates:
        return None

    on_disk = [cid for cid in candidates if resolve_existing_model(cid) is not None]
    pool = on_disk if on_disk else candidates

    if preference == "highest_acceptance":
        return pool[-1]
    return pool[0]


def _maybe_load_drafter(model_card: ModelCard) -> tuple[ModelId, Model] | None:
    """Load a drafter model declared on ``model_card``, if any.

    Returns the chosen ``(drafter_id, drafter_model)`` pair on success, or
    ``None`` when the card declares no drafter, the chosen drafter's weights
    are not on disk, ``EXO_DISABLE_DRAFTER`` is set, or the load itself
    fails. Drafter loading failures are logged and swallowed: the target
    model continues to load and inference falls back to standard
    (non-speculative) decoding.

    This helper is intentionally single-device only. Multi-device distributed
    inference does not pass ``draft_model`` through to ``stream_generate``
    today (see ``mlx_generate``), so loading a drafter on those ranks would
    just waste memory.
    """
    candidates = list(model_card.drafter_model_ids)
    if not candidates:
        return None
    if _drafter_disabled_by_env():
        logger.info(
            f"Drafter declared by {model_card.model_id} but "
            f"{EXO_DISABLE_DRAFTER_ENV} is set; skipping drafter load."
        )
        return None

    preference = _drafter_preference()
    drafter_id = _select_drafter_id(candidates, preference)
    if drafter_id is None:
        return None

    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        logger.warning(
            f"Drafter {drafter_id} (preferred '{preference}') declared by "
            f"{model_card.model_id} is not downloaded; falling back to "
            "standard decoding. Pre-download the drafter to enable "
            "speculative decoding."
        )
        return None

    drafter_start = time.perf_counter()
    try:
        drafter_model, _ = load_model(drafter_path, lazy=True, strict=False)
        mx.eval(drafter_model)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            f"Failed to load drafter {drafter_id}; continuing without "
            "speculative decoding."
        )
        return None
    logger.info(
        f"Loaded drafter {drafter_id} (preferred '{preference}') for "
        f"{model_card.model_id} in {(time.perf_counter() - drafter_start):.2f}s"
    )
    return drafter_id, cast(Model, drafter_model)


def _try_load_collocated_drafter(
    target_card: ModelCard,
    model: nn.Module,
    *,
    allow_standard_drafter_fallback: bool,
) -> tuple[CoupledDrafter | None, ModelId | None, Model | None]:
    """Resolve the collocated drafter (coupled or standard) for ``model``.

    Coupled-drafter precedence: when the card declares
    ``coupled_drafter`` we try it first because it's the path that
    yields the multi-x DFlash / MTP speedup. If the coupled load
    fails (mlx-vlm missing, weights absent, kind unrecognised, target
    type unsupported) we either fall through to the standard
    external-drafter list (single-device, where the standard drafter
    *is* dispatchable) or return empty-handed (multi-device, where
    the generator can't dispatch standard drafters yet so loading
    one would just waste memory).

    On a successful coupled load we ALSO attach the target-side hooks
    (``attach_mtp_hooks`` / ``attach_dflash_hooks``). The hook is the
    *capability gate* that :func:`mlx_generate` reads -- without it,
    the dispatch declines to route the request through the coupled
    path and the loaded coupled drafter stays passive. Hook
    attachment can fail on its own (e.g. the card incorrectly pairs a
    Gemma 4 ``coupled_drafter`` with a non-Gemma target); we treat
    that as another degrade-to-standard signal rather than a hard
    load failure so traffic keeps flowing through whichever drafter
    path is available.

    Used by both single-device and symmetric multi-rank (tensor-
    parallel) placements. Tensor parallel works because coupled
    drafters (~0.5-3 GB) replicate per rank and consume the post-
    all-reduce hidden state, which is identical on every rank. The
    drafter's own KV / SSM state replicates with the same logic.
    Asymmetric multi-rank uses a separate ``DrafterRunner`` reachable
    over the parent group and is handled by the caller (the
    ``drafter_placement is not None`` branch).

    Args:
        target_card: The target model card; supplies the
            ``coupled_drafter`` and ``drafter_model_ids`` declarations.
        model: The (possibly sharded) loaded target. Coupled hooks
            attach to this object's wrapper / inner-text-model
            sentinel attributes.
        allow_standard_drafter_fallback: Whether to fall back to
            :func:`_maybe_load_drafter` when no coupled drafter loads.
            Pass ``True`` for single-device placements (the standard
            drafter is dispatchable). Pass ``False`` for multi-device
            placements -- :func:`mlx_generate` declines to dispatch
            standard drafters when ``group is not None`` today, so a
            loaded standard drafter would just sit in memory unused.

    Returns:
        ``(coupled_drafter, drafter_id, drafter_model)`` where at
        most one of ``coupled_drafter`` and ``drafter_model`` is
        non-None. ``drafter_id`` is populated only on a successful
        standard-drafter load -- coupled-drafter attribution is
        threaded through ``GenerationStats`` from
        :data:`CoupledDrafter.model_id` instead, see
        :func:`_resolve_coupled_drafter_telemetry`.
    """
    coupled_drafter = _try_load_coupled_drafter(target_card)
    if coupled_drafter is not None:
        try:
            _dispatch_attach_coupled_hooks(coupled_drafter.kind, model)
        except _COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS as e:
            logger.warning(
                f"Coupled drafter loaded for "
                f"{target_card.model_id} but target type "
                f"{type(model).__name__!r} is incompatible "
                f"with the {coupled_drafter.kind} hooks "
                f"(error: {e}). Discarding coupled drafter "
                "and falling back to standard drafting."
            )
            coupled_drafter = None
    if coupled_drafter is not None:
        return coupled_drafter, None, None
    if not allow_standard_drafter_fallback:
        return None, None, None
    drafter_pair = _maybe_load_drafter(target_card)
    if drafter_pair is None:
        return None, None, None
    drafter_id, drafter_model = drafter_pair
    return None, drafter_id, drafter_model


def _drafter_weight_size_bytes(drafter_id: ModelId) -> int:
    """Best-effort drafter-on-disk size for the wired-memory bump.

    Walks the drafter directory and sums file sizes. Returns 0 on any error
    (the drafter weights aren't critical-path so we'd rather under-wire than
    crash).
    """
    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        return 0
    try:
        return sum(p.stat().st_size for p in drafter_path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _collocated_drafter_wired_bytes(
    *,
    target_card: ModelCard,
    group: mx.distributed.Group | None,
    drafter_placement: DrafterPlacement | None,
) -> Memory:
    """Bytes to add to the wired-memory limit for a collocated drafter.

    Mirrors :func:`_try_load_collocated_drafter`'s "will any drafter
    weights end up in this rank's address space?" decision exactly, so
    the wired bump matches what actually gets loaded:

    - ``drafter_placement is not None`` (asymmetric remote drafter) →
      0. The drafter lives on another node; its weights never enter
      this rank's wired pool.
    - ``EXO_DISABLE_DRAFTER=1`` → 0. The loader returns early before
      pulling any drafter weights.
    - ``group is None`` (single-device): tries coupled first then
      standard. The wired bump is the LARGER of the two on-disk sizes
      because the coupled load can fail at runtime (mlx-vlm missing,
      weights absent, unknown kind) and fall through to the standard
      drafter -- under-wiring there would page out the standard
      drafter between requests and undo the whole speedup. Over-wiring
      is cheap (the limit is a *minimum* on the wired pool, not a cap
      on total usage), so :func:`max` is the safe choice.
    - ``group is not None`` (symmetric tensor-parallel): only the
      coupled load runs (:func:`_try_load_collocated_drafter` is
      called with ``allow_standard_drafter_fallback=False``), so only
      the coupled size feeds the bump. The standard-drafter on-disk
      size is excluded to keep the wired limit minimal on the TP
      rank, which is already memory-tight for the 122B-class targets
      that motivate multi-device coupled dispatch in the first place.

    Note that the coupled drafter REPLICATES per TP rank rather than
    sharding: each rank loads the full drafter weights, KV cache, and
    SSM state in-process so it can consume its post-all-reduce hidden
    state without any cross-rank routing. The bump on a TP rank
    therefore reserves the *full* coupled-drafter size, not a shard
    of it.

    Args:
        target_card: The target model card.
        group: The MLX distributed group, or ``None`` for single-device.
        drafter_placement: ``bound_instance.instance.drafter_placement``,
            an asymmetric :class:`DrafterPlacement` or ``None``.

    Returns:
        ``Memory.from_bytes(0)`` when no bump is needed; otherwise the
        bytes to add to ``target_size`` before calling
        :func:`set_wired_limit_for_model`.
    """
    if drafter_placement is not None or _drafter_disabled_by_env():
        return Memory.from_bytes(0)
    candidate_bytes = 0
    if target_card.coupled_drafter is not None:
        candidate_bytes = max(
            candidate_bytes,
            _coupled_drafter_weight_size_bytes(target_card.coupled_drafter),
        )
    if group is None and target_card.drafter_model_ids:
        chosen = _select_drafter_id(
            list(target_card.drafter_model_ids), _drafter_preference()
        )
        if chosen is not None:
            candidate_bytes = max(candidate_bytes, _drafter_weight_size_bytes(chosen))
    return Memory.from_bytes(candidate_bytes)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: mx.distributed.Group | None,
) -> Generator[
    ModelLoadingResponse,
    None,
    tuple[
        Model,
        TokenizerWrapper,
        "VisionProcessor | None",
        Model | None,
        ModelId | None,
        CoupledDrafter | None,
    ],
]:
    target_card = bound_instance.bound_shard.model_card
    target_size = get_weights_size(bound_instance.bound_shard)

    # Pre-include drafter size in the wired-memory limit so the OS doesn't
    # page out drafter weights between requests. We have to make this decision
    # *before* loading the target because `set_wired_limit_for_model` configures
    # the limit once. Skip the bump for asymmetric placements: the drafter
    # weights live on a different node so they don't draw from this rank's
    # wired pool.
    combined_size = target_size + _collocated_drafter_wired_bytes(
        target_card=target_card,
        group=group,
        drafter_placement=bound_instance.instance.drafter_placement,
    )

    set_wired_limit_for_model(combined_size)

    drafter_model: Model | None = None
    drafter_id: ModelId | None = None
    coupled_drafter: CoupledDrafter | None = None

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(target_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Eval layers one by one for progress reporting
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                yield ModelLoadingResponse(layers_loaded=i, total=total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

        # Skip the local in-process drafter when an asymmetric drafter
        # rank exists for this instance: ``DrafterPlacement`` means the
        # drafter is a separate ``DrafterRunner`` reachable via
        # ``RemoteTransport`` over the parent group, and loading a
        # second copy locally would just duplicate the weights and
        # confuse the spec-decode loop. See
        # :func:`_try_load_collocated_drafter` for the coupled-vs-
        # standard precedence and fallback rules; both single-device
        # and tensor-parallel placements use the same helper.
        if bound_instance.instance.drafter_placement is None:
            coupled_drafter, drafter_id, drafter_model = _try_load_collocated_drafter(
                target_card, model, allow_standard_drafter_fallback=True
            )
        else:
            # Codex P2 (PR #20 round-(N+10), utils_mlx.py:578):
            # single-rank asymmetric target also has a remote drafter
            # but pre-fix this branch never surfaced the drafter id,
            # so ``GenerationStats.drafter_model_id`` stayed ``None``
            # and dashboards / telemetry gated on a non-null id
            # silently dropped attribution for every such request.
            # Mirror the multi-rank branch below: copy the placement's
            # drafter id even when no local weights are loaded.
            drafter_id = bound_instance.instance.drafter_placement.drafter_model_id

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = yield from shard_and_load(
            bound_instance.bound_shard,
            group=group,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

        # Asymmetric multi-rank placement: the drafter weights live on
        # a separate ``DrafterRunner``, so this rank doesn't load them
        # locally (no ``drafter_model``). The model id, however, is
        # known from the placement and is the only piece downstream
        # telemetry needs to surface "this request used the X drafter".
        # Without this, ``GenerationStats.drafter_model_id`` stays
        # ``None`` for every multi-target asymmetric request even
        # though the drafter is materially serving traffic.
        #
        # Symmetric multi-rank (tensor-parallel) placements have
        # ``drafter_placement is None`` and reach the same coupled-
        # drafter loader as the single-device branch: each TP rank
        # replicates the (small) coupled drafter and consumes the
        # post-all-reduce hidden state locally. Standard external
        # drafters still can't ride tensor parallel today
        # (``_maybe_load_drafter`` returns weights paired with a
        # standard generation step that ``mlx_generate`` only routes
        # under ``group is None``), so the loader will produce a
        # standard drafter for TP placements too -- the generator
        # caps that path off downstream with a ``"none"`` draft mode
        # while the coupled path stays active.
        drafter_placement = bound_instance.instance.drafter_placement
        if drafter_placement is not None:
            drafter_id = drafter_placement.drafter_model_id
        else:
            coupled_drafter, drafter_id, drafter_model = _try_load_collocated_drafter(
                target_card, model, allow_standard_drafter_fallback=False
            )

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from exo.worker.engines.mlx.vision import VisionProcessor

        vision_start_time = time.perf_counter()
        try:
            vision_processor: VisionProcessor | None = VisionProcessor(
                vision_config, bound_instance.bound_shard.model_card.model_id
            )
            vision_processor.load()
            logger.info(
                f"Time taken to load vision weights: {(time.perf_counter() - vision_start_time):.2f}s"
            )
        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to load vision weights — disabling vision for this runner"
            )
            vision_processor = None
    else:
        vision_processor = None

    return (
        cast(Model, model),
        tokenizer,
        vision_processor,
        drafter_model,
        drafter_id,
        coupled_drafter,
    )


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, tuple[nn.Module, TokenizerWrapper]]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = yield from tensor_auto_parallel(model, group)
        case AsymmetricTensorShardMetadata():
            rank_zero_ratio = shard_metadata.ratio
            ratios_list = [rank_zero_ratio, 1.0 - rank_zero_ratio]
            logger.info(
                f"loading model from {model_path} with asymmetric tensor parallelism "
                f"(ratios={[f'{r:.0%}' for r in ratios_list]})"
            )
            model = yield from asymmetric_tensor_auto_parallel(
                model, group, ratios_list
            )
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = yield from pipeline_auto_parallel(model, group, shard_metadata)
            mx.eval(model.parameters())
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # For GLM-5
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm-4.7" in model_id_lower:
        # For GLM-4.7
        # 151336: <|user|>, 151329: <|endoftext|>, 151338: <|observation|>
        return [151336, 151329, 151338]
    elif "glm" in model_id_lower:
        # For GLM-4.5 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # For Qwen3.5 / Qwen3.6: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    # We should really consider going back to mlx lm load to get tokenizer
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def _needs_v4_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """Remove `<think>…</think>` blocks and any stray `<think>`/`</think>` tags
    from prior-turn assistant content.

    The V4 encoder drops `reasoning_content` for older turns when
    `drop_thinking=True`"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    System messages almost exclusively must go at the start of a message
    and there must only be a single one.

    Also, Codex sends "developer" messages which are just system prompts.
    """
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = cast(str, msg.get("content", ""))
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)
    formatted_messages = non_system
    if system_parts:
        formatted_messages.insert(
            0, {"role": "system", "content": "\n".join(system_parts)}
        )
    return formatted_messages


def render_chat_template(
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, Any]],
    task_params: TextGenerationTaskParams,
) -> str:
    """
    Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages = consolidate_system_messages(messages)

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # Only use chat mode if enable thinking is explicitly Fakse.
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    if _needs_v4_encoding(task_params):
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
            encode_messages as encode_messages_v4,
        )

        v4_messages = [dict(m) for m in formatted_messages]
        for msg in v4_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _strip_v4_thinking_markers(content)
        if task_params.tools:
            for msg in v4_messages:
                if msg.get("role") in ("system", "developer"):
                    msg["tools"] = task_params.tools
                    break
            else:
                v4_messages.insert(
                    0, {"role": "system", "content": "", "tools": task_params.tools}
                )

        prompt = encode_messages_v4(
            messages=v4_messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=_v4_reasoning_effort(task_params),
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    # Put reasoning content in thinking block for GPT OSS
    if "gpt-oss" in task_params.model.lower():
        for msg in formatted_messages:
            if msg.get("role") == "assistant" and "thinking" not in msg:
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    msg["thinking"] = rc

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    return prompt


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    messages: list[dict[str, ChatTemplateValue]] = []
    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        messages = task_params.chat_template_messages
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            messages.append({"role": "system", "content": task_params.instructions})

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            messages.append({"role": msg.role, "content": msg.content})

    prompt = render_chat_template(tokenizer, messages, task_params)
    logger.debug(prompt)

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """Approximate token count of the system prompt portion of the input."""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    assert tokenizer.think_start_tokens
    assert tokenizer.think_end_tokens
    think_start_tokens: list[int] = tokenizer.think_start_tokens
    think_end_tokens: list[int] = tokenizer.think_end_tokens
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []

    depth = 0
    accumulated_think_start_length = 0
    accumulated_think_end_length = 0

    for token in token_list:
        if token == think_start_tokens[accumulated_think_start_length]:
            accumulated_think_start_length += 1
            if accumulated_think_start_length == len(think_start_tokens):
                depth += 1
                accumulated_think_start_length = 0

        elif token == think_end_tokens[accumulated_think_end_length]:
            accumulated_think_end_length += 1
            if accumulated_think_end_length == len(think_end_tokens):
                if depth == 0:
                    result.extend(think_start_tokens)
                else:
                    depth -= 1
                accumulated_think_end_length = 0

        else:
            accumulated_think_start_length = 0
            accumulated_think_end_length = 0

        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array | None, mx.array | None]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: mx.distributed.Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(mx.array(bool_), group=group)
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: mx.distributed.Group | None):
    if group is None:
        return
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))


# ``int32`` lower / upper bounds. Values broadcast through
# :func:`mx_broadcast_int_list` must be non-negative (the wire protocol
# uses unsigned token IDs and length prefixes) AND fit in int32 with
# room for the all-sum to land back in range. Since exactly one rank
# contributes the values and the rest contribute zero, the sum is the
# root's values per element regardless of group size, so the per-element
# bound is plain int32 max. We tighten to ``2**31 - 1`` (positive int32
# max) and reject negatives explicitly so a caller passing a Python
# ``-1`` doesn't silently wrap into a 4-billion-ish "valid" int32.
_MX_BROADCAST_MAX_VALUE: Final[int] = (1 << 31) - 1
# Toggle to dump every broadcast call's send/recv buffers. Set via
# ``EXO_PROBE_BROADCAST=1`` for ad-hoc diagnostics; leave off in
# steady state because the per-token logging spam quickly dominates.
_BROADCAST_PROBE: Final[bool] = bool(os.environ.get("EXO_PROBE_BROADCAST"))


# Distributed backend literal -- matches the strings we pass to
# ``mx.distributed.init(backend=...)`` in :func:`mlx_distributed_init`.
DistributedBackend = Literal["ring", "jaccl"]


def _detect_distributed_backend() -> DistributedBackend:
    """Resolve the active MLX distributed backend from the env vars
    set by :func:`mlx_distributed_init`.

    Why env-var sniffing instead of asking the group: ``mx.distributed.Group``
    only exposes ``rank()`` / ``size()`` / ``split()`` and gives no
    public hook for the backend name. We control the init path
    (:func:`mlx_distributed_init`) and set ``MLX_HOSTFILE`` for ring
    and ``MLX_IBV_DEVICES`` (plus ``MLX_JACCL_COORDINATOR``) for
    jaccl, so checking those env vars is a deterministic, in-process
    signal that doesn't require threading a backend literal through
    every call site.

    Backend selection matters because the ring backend is built around
    collective primitives (``all_sum`` / ``all_gather``) and does not
    support arbitrary point-to-point ``send`` / ``recv`` between
    non-neighbor ranks; multi-rank ring deployments would fail or
    hang the moment :func:`mx_broadcast_int_list` issued a
    ``send(dst=N)`` for a non-neighbor ``N``. JACCL, on the other
    hand, supports arbitrary ``send`` / ``recv`` and we deliberately
    use that to keep int32 broadcasts off the same all-reduce wire as
    TP float32 collectives (see the docstring on
    :func:`mx_broadcast_int_list` for the historical wire-conflation
    bug).

    Returns:
      ``"ring"`` when ``MLX_HOSTFILE`` is set, else ``"jaccl"``.
      Defaults to ``"ring"`` when neither marker is present so the
      ring-safe code path runs in ambiguous setups (e.g. tests that
      construct a fake group without going through
      :func:`mlx_distributed_init`).

    Raises:
      None. Detection is best-effort by design: the caller already
      gated multi-rank entry on ``group is not None``, and a
      misdetected backend at most picks the slower-but-correct
      collective path.
    """
    if os.environ.get("MLX_HOSTFILE"):
        return "ring"
    if os.environ.get("MLX_IBV_DEVICES") or os.environ.get("MLX_JACCL_COORDINATOR"):
        return "jaccl"
    return "ring"


def mx_broadcast_int_list(
    values: list[int] | None,
    length: int,
    group: mx.distributed.Group | None,
    *,
    is_root: bool,
) -> list[int]:
    """Broadcast a fixed-length int list from one rank to all peers.

    Backend-aware implementation:

      * ``ring``: use ``all_sum`` of an int32 buffer where non-root
        ranks contribute zeros and root contributes ``values``. Sum
        across the group recovers ``values`` element-wise (root's
        contribution is the only nonzero summand). MLX's ring backend
        is built around collective primitives and does not support
        arbitrary point-to-point ``send`` / ``recv`` between
        non-neighbor ranks, so this is the only ring-safe option.
      * ``jaccl``: rank-0 fanout via :func:`mx.distributed.send` /
        :func:`mx.distributed.recv`. Root issues one send to every
        peer; each peer issues a single matching recv from rank 0.

    Why split by backend: under JACCL the model's TP layers issue
    ``all_sum`` on the same target group on float32 buffers, every
    layer, every forward. A previous revision used ``all_sum`` for
    this broadcast on JACCL too and observed silent corruption on
    the spec-decode hot path: with >100 in-flight ``all_sum``
    collectives per round all on the same group, JACCL's pairing
    logic occasionally matched our int32 "broadcast" on rank A
    against the model's float32 TP all-reduce on rank B, scrambling
    the int32 buffer (symptom: token ids ~10^9 emitted by the spec
    loop, ``IndexError`` deep in the SPM detokenizer). Switching to
    ``send`` / ``recv`` on JACCL makes this broadcast a different
    primitive than the TP all-reduce so JACCL has no opportunity to
    merge them. Ring lacks both the JACCL pairing pitfall and the
    arbitrary-``send`` capability, so it stays on ``all_sum``.

    Caller note: the spec-decode hot path no longer routes through
    this function -- it uses :func:`target_peer_broadcast_int_list`
    over a dedicated TCP fanout (see :class:`TargetPeerFanout`). The
    only remaining caller is :func:`mx_all_gather_tasks` at admit
    boundaries, which fires far below TP all-reduce frequency, so
    even on JACCL the wire-conflation risk is low; the
    ``send`` / ``recv`` path is kept for defense-in-depth.

    The fixed-length contract means the caller pads to ``length`` on
    root and both ranks agree on ``length`` ahead of time, which keeps
    the recv shape (or all_sum buffer shape) known statically.

    Args:
      values: On root, a list of exactly ``length`` ints to broadcast.
        Each value must be in ``[0, 2**31 - 1]``. Negative values are
        rejected explicitly so a stray ``-1`` doesn't silently wrap
        on the int32 cast and corrupt the broadcast. Ignored on
        non-root.
      length: Buffer size, agreed by all ranks. Must be ``>= 1``.
      group: Distributed group; ``None`` is a single-rank short-circuit
        that simply returns ``values`` (root-only).
      is_root: ``True`` on the rank holding the source values; ``False``
        elsewhere. Exactly one rank in ``group`` must pass ``True``.

    Returns:
      A list of ``length`` ints identical on every rank in ``group``,
      equal to root's ``values``.

    Raises:
      ValueError: ``length`` is non-positive, the root's ``values`` are
        ``None`` or wrong length, or any root value is out of int32
        range. These are caller bugs, not runtime conditions.
    """
    if length < 1:
        raise ValueError(f"mx_broadcast_int_list length must be >= 1, got {length}")

    if group is None:
        if not is_root:
            raise ValueError(
                "mx_broadcast_int_list: single-rank short-circuit requires "
                "is_root=True (only the root has source values)"
            )
        if values is None or len(values) != length:
            raise ValueError(
                "mx_broadcast_int_list: single-rank call requires "
                f"values of length {length}, got "
                f"{None if values is None else len(values)}"
            )
        _validate_broadcast_values(values)
        return list(values)

    group_size = group.size()

    if is_root and (values is None or len(values) != length):
        raise ValueError(
            "mx_broadcast_int_list root rank requires values of "
            f"length {length}, got {None if values is None else len(values)}"
        )
    if is_root:
        # ``cast`` for the type-checker: validated above.
        _validate_broadcast_values(cast(list[int], values))

    backend = _detect_distributed_backend()

    if backend == "ring":
        # Ring backend: collective ``all_sum``. Root contributes the
        # values, every other rank contributes a zero buffer of the
        # same shape, so the element-wise sum is ``values``. This is
        # the only ring-safe broadcast primitive (ring rejects
        # arbitrary point-to-point ``send`` / ``recv`` between
        # non-neighbor ranks).
        if is_root:
            local = mx.array(cast(list[int], values), dtype=mx.int32)
        else:
            local = mx.zeros(shape=(length,), dtype=mx.int32)
        summed = mx.distributed.all_sum(local, group=group)
        mx.eval(summed)
        out = [int(v) for v in cast(list[int], summed.tolist())]
        if _BROADCAST_PROBE:
            role = "ROOT" if is_root else "PEER"
            logger.warning(
                f"mx_broadcast_int_list[ring] {role} recovered {out} (len={length})"
            )
        return out

    # JACCL backend: send/recv fanout from rank 0.
    if is_root:
        send_buffer = mx.array(cast(list[int], values), dtype=mx.int32)
        for dst in range(1, group_size):
            sent = mx.distributed.send(send_buffer, dst=dst, group=group)
            mx.eval(sent)
        if _BROADCAST_PROBE:
            logger.warning(
                f"mx_broadcast_int_list[jaccl] ROOT sent {values} (len={length})"
            )
        return list(cast(list[int], values))

    received = mx.distributed.recv(shape=(length,), dtype=mx.int32, src=0, group=group)
    mx.eval(received)
    out = [int(v) for v in cast(list[int], received.tolist())]
    if _BROADCAST_PROBE:
        logger.warning(
            f"mx_broadcast_int_list[jaccl] PEER recvd {out} (expected len={length})"
        )
    return out


def target_peer_broadcast_int_list(
    values: list[int] | None,
    length: int,
    fanout: TargetPeerFanout,
    *,
    is_root: bool,
) -> list[int]:
    """Broadcast a fixed-length signed int list over the TCP fanout.

    Drop-in replacement for :func:`mx_broadcast_int_list` on the
    spec-decode hot path. Same shape contract (``length`` agreed by
    every rank up front; root passes ``values``, peers pass
    ``None``); the only difference is that this version rides direct
    TCP sockets instead of ``mx.distributed.send`` / ``recv``,
    sidestepping the JACCL int/float wire-conflation bug entirely.

    Wire format (every frame): ``length`` little-endian signed int32
    values, no header. The peer side knows ``length`` from the same
    shape contract the caller agreed to.

    Args:
      values: On root, exactly ``length`` int32-range values to
        broadcast. Ignored on peers.
      length: Buffer size, agreed by all ranks. Must be ``>= 1``.
      fanout: Pre-built fanout from :func:`_maybe_setup_target_peer_fanout`.
        Carries the per-rank role (rank 0 vs peer) and the connected
        sockets. Mismatched ``is_root`` vs ``fanout.rank`` is a caller
        bug and raises :class:`ValueError`.
      is_root: ``True`` on rank 0, ``False`` elsewhere. Asserted
        against ``fanout.rank``.

    Returns:
      A list of ``length`` ints identical on every rank, equal to
      root's ``values``.

    Raises:
      ValueError: caller-bug conditions (length, values shape,
        is_root vs rank mismatch).
      ConnectionError: a peer closed the socket mid-frame; surfaces
        as a runner failure for the supervisor to rebuild.
    """
    import socket as _socket

    from exo.worker.engines.mlx.generator.target_peer_socket import (
        recv_int32_frame,
        send_int32_frame,
    )

    if length < 1:
        raise ValueError(
            f"target_peer_broadcast_int_list length must be >= 1, got {length}"
        )
    if is_root != (fanout.rank == 0):
        raise ValueError(
            f"target_peer_broadcast_int_list is_root={is_root} disagrees "
            f"with fanout.rank={fanout.rank}; exactly one rank in the "
            "fanout must pass is_root=True"
        )
    if is_root:
        if values is None or len(values) != length:
            raise ValueError(
                "target_peer_broadcast_int_list root rank requires values "
                f"of length {length}, got "
                f"{None if values is None else len(values)}"
            )
        for sock in fanout.peer_sockets.values():
            assert isinstance(sock, _socket.socket)  # narrow object -> socket
            send_int32_frame(sock, values)
        return list(values)
    sock = fanout.rank_zero_socket
    if sock is None:
        raise RuntimeError(
            "target_peer_broadcast_int_list called on peer rank but "
            "fanout.rank_zero_socket is None; bootstrap must populate it"
        )
    assert isinstance(sock, _socket.socket)
    return recv_int32_frame(sock, length)


def mx_all_sum_int_list(
    values: list[int],
    length: int,
    group: mx.distributed.Group | None,
) -> list[int]:
    """Element-wise ``all_sum`` of an ``int32`` list across all ranks.

    Unlike :func:`mx_broadcast_int_list` (one-rank-contributes), every
    rank contributes its own ``values`` and every rank sees the
    element-wise sum. Used by the two-collective intersection
    protocol in :func:`mx_all_gather_tasks` to vote on which tasks
    every rank has locally: each rank emits a ``[0, 1]`` indicator
    vector and the sum equals the group's vote count per slot.

    Same wire reliability story as :func:`mx_broadcast_int_list`:
    rides MLX's well-exercised ``all_sum`` primitive, validates
    int32 bounds explicitly so a stray Python ``-1`` doesn't wrap
    silently.

    Args:
      values: This rank's contribution. Length must equal ``length``;
        each value must be in ``[0, 2**31 - 1]``. After all-sum the
        per-element bound is ``group_size * max(value)`` -- callers
        sizing for ``[0, 1]`` indicators sit far below int32 max for
        any plausible ``group_size``.
      length: Buffer size, agreed by all ranks.
      group: Distributed group; ``None`` short-circuits to a copy of
        ``values`` (single-rank vote sums to itself).

    Returns:
      A list of length ``length`` with the element-wise sum of every
      rank's ``values``, identical on every rank.

    Raises:
      ValueError: ``length`` is non-positive, ``values`` length
        mismatches, or any value is out of int32 range.
    """
    if length < 1:
        raise ValueError(f"mx_all_sum_int_list length must be >= 1, got {length}")
    if len(values) != length:
        raise ValueError(
            f"mx_all_sum_int_list values must have length {length}, got {len(values)}"
        )
    _validate_broadcast_values(values)
    if group is None:
        return list(values)
    buffer = mx.array(values, dtype=mx.int32)
    # ``all_sum`` is acceptable here because :func:`mx_all_sum_int_list`
    # is only called from the task agreement protocol, which fires at
    # admit boundaries -- not on the per-token spec-decode hot path.
    # The thrash that broke the broadcast helper (interleaving with
    # the model's TP all-reduce 100+ times per round) does not apply
    # at this call frequency.
    summed = mx.distributed.all_sum(buffer, group=group)
    mx.eval(summed)
    return [int(v) for v in cast(list[int], summed.tolist())]


def _validate_broadcast_values(values: list[int]) -> None:
    """Range-check root-side broadcast values.

    Centralised so both the single-rank short-circuit and the multi-
    rank all-sum path enforce identical contracts. Linear scan; for
    ``length`` values this is microseconds and runs once per round on
    the spec-decode hot path -- amortised free against an MLX
    collective.
    """
    for index, value in enumerate(values):
        if value < 0 or value > _MX_BROADCAST_MAX_VALUE:
            raise ValueError(
                f"mx_broadcast_int_list values must be in "
                f"[0, {_MX_BROADCAST_MAX_VALUE}]; "
                f"index {index} = {value} is out of range "
                f"(negatives wrap silently in int32 all-sum; values "
                f">= 2**31 overflow)"
            )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]


# Maximum number of tasks the agreement protocol can carry per round.
# Sized to ``EXO_MAX_CONCURRENT_REQUESTS`` (default 8) plus headroom for
# transient ``_maybe_queue`` build-up; tasks beyond this slot count get
# deferred to the next agreement round, never lost. Matches the sizing
# the supervisor already enforces via ``max_concurrent_tasks`` at the
# generator layer, so steady-state oversubscription is not a real
# concern.
_MX_AGREE_MAX_TASKS: Final[int] = 16
# UUID4 string length (``len("01234567-...-...-...-............") == 36``).
# The agreement protocol broadcasts task IDs as fixed-width ASCII so
# every rank can decode the same canonical payload. Hashes are not
# enough on their own because root needs to specify *which* tasks are
# in the agreed set without leaving the consumer guessing on collision.
_MX_TASK_ID_BYTES: Final[int] = 36
# Buffer layout: ``[count, task_id_bytes_0, task_id_bytes_1, ...]`` where
# each task_id slot is ``_MX_TASK_ID_BYTES`` ints (one ASCII char per
# int32 slot). A char fits trivially in int32, and using one slot per
# char avoids endian / packing concerns at the cost of ~4x bandwidth --
# acceptable since this only runs at admit boundaries, not per-token.
_MX_AGREE_BUFFER_LEN: Final[int] = 1 + _MX_AGREE_MAX_TASKS * _MX_TASK_ID_BYTES


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    """Two-phase intersection-based task agreement across target ranks.

    Returns ``(agreed, leftover)`` where:

      * ``agreed``: tasks every rank in the group has locally, in the
        canonical order set by the root rank. Identical on every
        rank by construction (the consensus is computed inside the
        function, not after the return).
      * ``leftover``: this rank's local tasks that didn't make it
        into ``agreed`` (either root hasn't seen them yet or another
        peer is still waiting on libp2p delivery). Every rank stashes
        its leftover for the next agreement cycle.

    Wire protocol:
      Phase 1 (broadcast root's IDs):
        Root encodes ``[count, id_0_chars, ..., id_(count-1)_chars]``
        into a fixed ``_MX_AGREE_BUFFER_LEN`` int32 buffer
        (zero-padded slots) and broadcasts via
        :func:`mx_broadcast_int_list`. Non-root ranks decode it as
        their canonical view of "candidate tasks".
      Phase 2 (vote on intersection):
        Every rank emits a ``[0, 1]`` vote vector indexed by phase-1
        slot: 1 means "I have this task locally", 0 means "I don't".
        :func:`mx_all_sum_int_list` element-wise-sums the votes
        across the group. A slot whose sum equals ``group_size`` is
        agreed -- every rank had it. Slots below ``group_size`` are
        deferred (they re-enter the next round once delivery
        completes).

    Why intersection instead of root-authoritative:
      Root-authoritative agreement (root admits all its tasks; non-
      root admits only the subset it has locally) breaks the
      collective-count contract. If root admits a task the non-root
      doesn't have, non-root's ``_active_tasks`` stays empty, its
      next ``step()`` calls ``agree_on_tasks`` again while root is
      mid-``next(gen)`` issuing spec-loop ``all_sum`` collectives.
      The two collective streams interleave on the wire and corrupt
      each other's payloads (manifests as ``IndexError: list index
      out of range`` in the detokenizer because broadcast tokens
      arrive scrambled). Intersection keeps both ranks at the same
      collective count: every rank that admits a task admits it on
      the same step.

    Why ``group is None`` short-circuits without touching MLX:
      ``mx.distributed.all_gather(group=None)`` delegates to MLX's
      default group, which on an asymmetric runner is the parent
      (target+drafter) group. The drafter rank is busy in
      ``drafter_serve_loop`` doing its own ``recv`` on that same
      default group, so an unguarded all-gather here cross-talks
      with the drafter's wire protocol. When ``group is None`` we
      are by construction the only participating rank, so every
      task is trivially "agreed".

    Cost:
      Two collectives per call (one broadcast + one all-sum), each
      on small int32 buffers (~600 bytes). On Apple Silicon JACCL
      this is sub-millisecond and runs only at admit boundaries,
      not per token.
    """
    if group is None:
        return list(tasks), []

    is_root = group.rank() == 0
    group_size = group.size()

    # ----- Phase 1: root broadcasts canonical task ID list -----
    if is_root:
        admitted = tasks[:_MX_AGREE_MAX_TASKS]
        payload: list[int] = [len(admitted)]
        for task in admitted:
            payload.extend(_encode_task_id(task.task_id))
        payload.extend([0] * (_MX_AGREE_BUFFER_LEN - len(payload)))
        broadcast = mx_broadcast_int_list(
            payload, _MX_AGREE_BUFFER_LEN, group, is_root=True
        )
    else:
        broadcast = mx_broadcast_int_list(
            None, _MX_AGREE_BUFFER_LEN, group, is_root=False
        )

    count = broadcast[0]
    if count < 0 or count > _MX_AGREE_MAX_TASKS:
        # Programming error: root encoded a count outside the agreed
        # bounds. Hard failure -- buffer corrupt, can't decode safely.
        raise RuntimeError(
            f"mx_all_gather_tasks: broadcast count {count} outside "
            f"[0, {_MX_AGREE_MAX_TASKS}]; broadcast buffer corrupt"
        )

    canonical_ids: list[str] = []
    for i in range(count):
        start = 1 + i * _MX_TASK_ID_BYTES
        end = start + _MX_TASK_ID_BYTES
        canonical_ids.append(_decode_task_id(broadcast[start:end]))

    # ----- Phase 2: every rank votes on which canonical IDs it has -----
    local_by_id: dict[str, TextGeneration] = {t.task_id: t for t in tasks}
    vote = [1 if cid in local_by_id else 0 for cid in canonical_ids]
    vote.extend([0] * (_MX_AGREE_MAX_TASKS - count))
    summed_vote = mx_all_sum_int_list(vote, _MX_AGREE_MAX_TASKS, group)

    # ----- Phase 3: build agreed (intersection) and leftover -----
    agreed: list[TextGeneration] = []
    for i, cid in enumerate(canonical_ids):
        if summed_vote[i] != group_size:
            continue
        local = local_by_id.pop(cid, None)
        if local is None:
            # Root contributed this ID but isn't a vote-counter on
            # itself -- only possible if we're not root and we don't
            # have the task. The vote sum requirement above handles
            # this case (we'd have voted 0 and it wouldn't reach
            # ``group_size``); reaching here means buffer corruption.
            raise RuntimeError(
                f"mx_all_gather_tasks: canonical id {cid} agreed by "
                "vote but missing locally; vote/broadcast desync"
            )
        agreed.append(local)

    # Codex P1 (PR #21 round 3): preserve admission progress when the
    # first page of ``tasks`` contains IDs that aren't yet present on
    # every peer.
    #
    # Pre-fix behavior: ``leftover`` was just ``local_by_id.values()``,
    # which preserves dict insertion order. So if ``tasks[:k]`` were
    # all stuck (e.g., a fresh peer whose first 16 deliveries were
    # delayed), root would re-broadcast the same first page every
    # round and tasks at positions ``k..N`` could starve indefinitely
    # because they never entered the canonical broadcast.
    #
    # Fix: split the leftover into two regions and concatenate them so
    # next round's ``tasks[:_MX_AGREE_MAX_TASKS]`` is biased toward
    # candidates that haven't been broadcast yet.
    #   front_of_leftover: tasks that were never in the canonical
    #     broadcast (positions >= count) -- these have never had a
    #     chance to be admitted, prioritize them.
    #   back_of_leftover: canonical tasks that didn't reach
    #     intersection -- demote them so they don't keep blocking
    #     root's first page. They still get retried, just rotated
    #     behind everything that hasn't been tried yet.
    canonical_id_set: set[str] = set(canonical_ids)
    front_of_leftover: list[TextGeneration] = []
    back_of_leftover: list[TextGeneration] = []
    for task in tasks:
        if task.task_id not in local_by_id:
            # Already admitted into ``agreed``.
            continue
        if task.task_id in canonical_id_set:
            back_of_leftover.append(task)
        else:
            front_of_leftover.append(task)
    leftover = front_of_leftover + back_of_leftover
    return agreed, leftover


def _encode_task_id(task_id: str) -> list[int]:
    """ASCII-encode ``task_id`` into ``_MX_TASK_ID_BYTES`` int32 slots.

    Right-pads with zeros if ``task_id`` is shorter than the slot
    count; raises if it's longer or contains non-ASCII (UUIDs are pure
    ASCII by construction, so any rejection here points at upstream
    bugs).
    """
    encoded = task_id.encode("ascii")
    if len(encoded) > _MX_TASK_ID_BYTES:
        raise ValueError(
            f"task_id {task_id!r} exceeds {_MX_TASK_ID_BYTES} bytes; "
            "agreement buffer slot is sized for UUID4 strings only"
        )
    out = [int(b) for b in encoded]
    out.extend([0] * (_MX_TASK_ID_BYTES - len(out)))
    return out


def _decode_task_id(slots: list[int]) -> str:
    """Inverse of :func:`_encode_task_id`: int32 slots -> ASCII string.

    Stops at the first zero byte (the encode pad), so the result is
    bounded by ``_MX_TASK_ID_BYTES``. Any non-ASCII byte is rejected
    locally rather than silently coerced; the broadcast contract
    requires ASCII-only IDs.
    """
    chars: list[str] = []
    for value in slots:
        if value == 0:
            break
        if value < 0 or value > 127:
            raise ValueError(
                f"task_id slot {value} outside ASCII range; broadcast payload corrupt"
            )
        chars.append(chr(value))
    return "".join(chars)
