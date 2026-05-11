import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from os import environ
from typing import Literal, Sequence

from loguru import logger

from exo.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    find_ip_prioritised,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    DrafterPlacementDegradationReason,
    DrafterPlacementDegraded,
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.topology import SocketConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    DrafterPlacement,
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import Sharding
from exo.utils.ports import random_ephemeral_port, random_ephemeral_port_excluding

ASYMMETRIC_TENSOR_AUTO_UPGRADE_ENV = "EXO_ENABLE_ASYMMETRIC_TP_AUTO_UPGRADE"


def _supports_asymmetric_tensor_parallel(model_card: ModelCard) -> bool:
    model_id = model_card.model_id.lower()
    base_model = model_card.base_model.lower()
    return (
        base_model.startswith("qwen3.5")
        or "qwen3.5" in model_id
        or "qwen-3.5" in model_id
    )


def _asymmetric_tensor_auto_upgrade_enabled() -> bool:
    return environ.get(ASYMMETRIC_TENSOR_AUTO_UPGRADE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0-1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    required_nodes: set[NodeId] | None = None,
    allowed_nodes: set[NodeId] | None = None,
    allow_single_node_total_memory: bool = False,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] | None = None,
    on_drafter_placement_degraded: (
        Callable[[DrafterPlacementDegraded], None] | None
    ) = None,
) -> dict[InstanceId, Instance]:
    sharding = command.sharding
    instance_meta = command.instance_meta
    cycles = topology.get_cycles()
    candidate_cycles = list(filter(lambda it: len(it) >= command.min_nodes, cycles))

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
    if allowed_nodes is not None:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if set(cycle.node_ids).issubset(allowed_nodes)
        ]

    # Reserve drafter-eligible nodes for the drafter rank when possible, so
    # the placement layer doesn't accidentally pull a drafter-eligible node
    # into the target cycle and then degrade because no eligible host
    # remains. If filtering them out leaves zero cycles, fall back to the
    # unfiltered set -- the user gets target placement at the cost of the
    # asymmetric drafter, and `_select_drafter_placement` emits a
    # ``AllEligibleNodesInTargetCycle`` degradation downstream.
    #
    # Codex P1.3 (PR #20): the reservation filter must also respect
    # memory feasibility. Pre-fix, ``cycles_excluding_drafters`` was
    # adopted as long as it was non-empty -- which would drop the only
    # memory-feasible cycle when every spare-target candidate was too
    # small for the model. ``filter_cycles_by_memory`` would then
    # return ``[]`` and the placement aborted with "No cycles found
    # with sufficient memory" even though the unfiltered set had at
    # least one feasible cycle (it just happened to include a
    # drafter-eligible node). We instead probe ``cycles_excluding_drafters``
    # against memory first; if that yields zero feasible cycles we
    # fall back to the unfiltered set so the instance still lands.
    # ``_select_drafter_placement`` emits ``AllEligibleNodesInTargetCycle``
    # downstream so the operator sees the asymmetric drafter degradation.
    eligible_drafter_set = set(command.model_card.drafter_eligible_nodes)
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles,
        node_memory,
        command.model_card.storage_size,
        allow_single_node_total_memory=allow_single_node_total_memory,
    )
    if eligible_drafter_set and command.model_card.drafter_model_ids:
        cycles_excluding_drafters = [
            cycle
            for cycle in candidate_cycles
            if not (set(cycle.node_ids) & eligible_drafter_set)
        ]
        if cycles_excluding_drafters:
            feasible_excluding_drafters = filter_cycles_by_memory(
                cycles_excluding_drafters,
                node_memory,
                command.model_card.storage_size,
                allow_single_node_total_memory=allow_single_node_total_memory,
            )
            if feasible_excluding_drafters:
                candidate_cycles = cycles_excluding_drafters
                cycles_with_sufficient_memory = feasible_excluding_drafters
    if len(cycles_with_sufficient_memory) == 0:
        raise ValueError("No cycles found with sufficient memory")

    if (
        sharding == Sharding.AsymmetricTensor
        and not _supports_asymmetric_tensor_parallel(command.model_card)
    ):
        raise ValueError(
            f"Asymmetric tensor parallelism is not yet supported for "
            f"model '{command.model_card.model_id}'. Supported: Qwen3.5."
        )

    if sharding in (Sharding.Tensor, Sharding.AsymmetricTensor):
        if not command.model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        if sharding == Sharding.Tensor:
            # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
            # DeepSeek V4 is MQA (num_key_value_heads=1) but its sharding strategy
            # head-parallelises wq_b/wo_a and shards MoE experts instead of splitting
            # KV heads, so the kv-head divisibility check doesn't apply.
            is_deepseek_v4 = command.model_card.base_model.startswith("DeepSeek V4")
            kv_heads = command.model_card.num_key_value_heads
            cycles_with_sufficient_memory = [
                cycle
                for cycle in cycles_with_sufficient_memory
                if command.model_card.hidden_size % len(cycle) == 0
                and (is_deepseek_v4 or kv_heads is None or kv_heads % len(cycle) == 0)
            ]
            if not cycles_with_sufficient_memory:
                raise ValueError(
                    f"No tensor sharding found for model with "
                    f"hidden_size={command.model_card.hidden_size}"
                    f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                    f" across candidate cycles"
                )

            # Auto-upgrade to AsymmetricTensor when equal TP won't fit on
            # the smallest node but asymmetric split would.
            if (
                _asymmetric_tensor_auto_upgrade_enabled()
                and _supports_asymmetric_tensor_parallel(command.model_card)
            ):
                for cycle in cycles_with_sufficient_memory:
                    if len(cycle) != 2:
                        continue
                    equal_share = command.model_card.storage_size.in_bytes / len(cycle)
                    min_node_mem = min(
                        node_memory[nid].ram_available.in_bytes for nid in cycle
                    )
                    if equal_share > min_node_mem * 0.9:
                        # Equal split too tight; try asymmetric.
                        total_mem = sum(
                            node_memory[nid].ram_available.in_bytes for nid in cycle
                        )
                        if command.model_card.storage_size.in_bytes < total_mem * 0.85:
                            logger.info(
                                "Equal tensor split won't fit on smallest node "
                                f"({min_node_mem / 1e9:.0f}GB available, "
                                f"needs {equal_share / 1e9:.0f}GB). "
                                "Auto-upgrading to AsymmetricTensor."
                            )
                            sharding = Sharding.AsymmetricTensor
                        break
    if sharding == Sharding.AsymmetricTensor:
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 2
        ]
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if _asymmetric_tensor_rank_zero_is_socket_reachable(
                cycle=cycle,
                node_memory=node_memory,
                topology=topology,
            )
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Asymmetric tensor parallelism currently requires exactly 2 nodes "
                "with the largest-memory rank-0 node socket-reachable"
            )

    if sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )
    if sharding == Sharding.Pipeline and command.model_card.base_model.startswith(
        "Gemma 4"
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
            )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)
    rdma_ctl_status = node_rdma_ctl or {}

    def _all_rdma_ctl_enabled(cycle: Cycle) -> bool:
        return all(
            ((status := rdma_ctl_status.get(node_id)) is not None and status.enabled)
            for node_id in cycle
        )

    smallest_rdma_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.is_rdma_cycle(cycle) and _all_rdma_ctl_enabled(cycle)
    ]

    if instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        # Filter to cycles whose every node advertises a valid Thunderbolt
        # IPv4 peer path BEFORE the scoring/selection pass. Previously the
        # preflight only ran on the already-chosen cycle, so a single
        # unrepaired node could fail placement even when another RDMA cycle
        # of the same size was perfectly valid (e.g. mixed clusters where
        # only one node is still on 169.254-only paths). When no candidate
        # is eligible we deliberately fall back to the full RDMA pool so
        # the post-selection ``_validate_jaccl_thunderbolt_ipv4_paths``
        # check still surfaces the actionable, node-specific error message
        # (which lists the missing nodes) instead of a generic
        # "no candidates" failure here.
        #
        # Codex P2 (PR #11 round 4): the JACCL prefilter must NOT run on
        # singleton cycles. A ``MlxJaccl`` request with ``min_nodes=1``
        # gets downgraded to ``MlxRing`` further down (single-node
        # JACCL is meaningless because target ranks have no peers to
        # talk to over Thunderbolt RDMA), and that downgraded ring
        # placement does not require a TB-IPv4 path. Pre-fix, requiring
        # TB-IPv4 on length-1 candidates pushed the selector toward
        # nodes that happened to have TB metadata (lower memory /
        # download score in mixed clusters) instead of letting the
        # ring downgrade pick the actual best singleton.
        jaccl_eligible_rdma_cycles = [
            cycle
            for cycle in smallest_rdma_cycles
            if len(cycle) == 1
            or all(
                _node_has_or_lacks_known_jaccl_path(node_network, node_id)
                != "known_no_path"
                for node_id in cycle.node_ids
            )
        ]
        smallest_cycles = jaccl_eligible_rdma_cycles or smallest_rdma_cycles

    resolved_download_status = download_status or {}

    selected_cycle = max(
        smallest_cycles,
        key=lambda cycle: (
            _cycle_download_score(
                cycle, command.model_card.model_id, resolved_download_status
            ),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
            any(topology.node_is_leaf(node_id) for node_id in cycle),
        ),
    )
    selected_cycle = _prefer_socket_reachable_rank_zero(selected_cycle, topology)
    if sharding == Sharding.AsymmetricTensor:
        selected_cycle = _order_asymmetric_tensor_cycle(
            cycle=selected_cycle,
            node_memory=node_memory,
            topology=topology,
        )

    # Single-node target cycle requires Pipeline sharding (PP=1). Under
    # the V3+ asymmetric-drafter wire, the drafter rank does NOT join
    # the target's ``mx.distributed`` group; it talks to target rank 0
    # over a direct TCP socket (see ``DrafterPlacement``). A single-
    # rank target therefore never needs ``mx.distributed`` at all and
    # ring stays sufficient regardless of drafter eligibility.
    #
    # Codex P1.4 (PR #20, placement.py:396): pre-fix, the asymmetric-
    # drafter peek auto-upgraded ``MlxRing -> MlxJaccl`` whenever the
    # card declared drafter-eligible nodes -- which then forced
    # ``_validate_jaccl_thunderbolt_ipv4_paths`` to fire and fail on
    # any Wi-Fi/Ethernet-only single-node deploy. Single-rank targets
    # don't need a distributed group, so the upgrade was both
    # unnecessary and actively harmful. Keep ring locked in for
    # single-rank cycles; the drafter socket wire is independent.
    if len(selected_cycle) == 1:
        sharding = Sharding.Pipeline
        instance_meta = InstanceMeta.MlxRing

    # Three independent post-selection adjustments. They land in this
    # order so the JACCL preflight fails fast (raising a node-specific
    # error message) before we go through the work of computing the
    # singleton total-memory expansion or the drafter-multi-node warning.
    # The first two checks are mutually exclusive in practice -- the JACCL
    # preflight only fires when ``instance_meta == MlxJaccl`` (multi-node)
    # and the ``allow_single_node_total_memory`` expansion only fires for
    # singleton cycles, which were already downgraded to ``MlxRing`` by
    # the block above -- but we keep both unconditional so the invariant
    # is encoded in the code itself rather than in a comment about
    # ordering. The drafter-multi-node warning (item 10) is purely an
    # operator hint emitted when a drafter-aware model card ends up on
    # more than one node, since speculative decoding is single-device
    # only in mlx_lm and the drafter would otherwise be silently dropped.
    if instance_meta == InstanceMeta.MlxJaccl:
        _validate_jaccl_thunderbolt_ipv4_paths(selected_cycle, node_network)

    if len(selected_cycle) > 1 and command.model_card.drafter_model_ids:
        logger.warning(
            f"Model {command.model_card.model_id} declares drafters "
            f"{list(command.model_card.drafter_model_ids)} but is being "
            f"placed across {len(selected_cycle)} nodes. Speculative "
            "decoding is single-device only and will be disabled for this "
            "instance. To get the drafter speedup, place a smaller quant "
            "(e.g. 4-bit) on the largest single node instead."
        )

    placement_node_memory = (
        _node_memory_with_total_capacity(selected_cycle, node_memory)
        if allow_single_node_total_memory and len(selected_cycle) == 1
        else node_memory
    )
    shard_assignments = get_shard_assignments(
        command.model_card, selected_cycle, sharding, placement_node_memory
    )

    instance_id = InstanceId()
    # Codex P2 (PR #21 round 3): the drafter / target-peer ports must
    # also avoid colliding with the per-meta listener port that the
    # ``match instance_meta`` block below allocates on rank 0
    # (``coordinator_port`` for MlxJaccl or ``ephemeral_port`` for
    # MlxRing). Pre-allocate that port here and pass it as a
    # ``reserved_ports`` set so ``_select_drafter_placement``'s draws
    # exclude it; otherwise rank 0 occasionally hit ``EADDRINUSE``
    # during runner bootstrap when the random draws happened to
    # coincide.
    pre_allocated_listener_port = random_ephemeral_port()
    drafter_placement = _select_drafter_placement(
        command=command,
        selected_cycle=selected_cycle,
        instance_meta=instance_meta,
        topology=topology,
        node_memory=node_memory,
        node_network=node_network,
        instance_id=instance_id,
        reserved_ports=frozenset({pre_allocated_listener_port}),
        on_drafter_placement_degraded=on_drafter_placement_degraded,
        download_status=download_status or {},
    )

    # Codex P1.4: under the V3+ wire, single-rank target cycles always
    # use ``MlxRing`` (no auto-upgrade to ``MlxJaccl`` even when an
    # asymmetric drafter is reachable). The drafter wire is a TCP
    # socket independent of ``mx.distributed``, so there's no need
    # for jaccl's ``Group.split``. The pre-fix revert path (jaccl ->
    # ring on missing drafter placement) is therefore dead under the
    # new policy and removed; ring is locked in upstream.

    # Asymmetric placement (``drafter_placement is not None``) keeps the
    # drafter rank OUT of the parent ``mx.distributed`` group: the
    # drafter talks to target rank 0 over a direct TCP socket
    # (``DrafterPlacement.drafter_socket_host``/``port``). Subgraph +
    # connectivity tables (``hosts_by_node`` / ``jaccl_devices``)
    # therefore cover only target nodes -- this lets target ranks of
    # any size run TP/PP collectives without requiring
    # ``Group.split`` (jaccl/ring backends do not implement split on
    # Apple Silicon).
    nodes_for_group = list(selected_cycle.node_ids)
    cycle_digraph: Topology = topology.get_subgraph_from_nodes(nodes_for_group)

    target_instances = dict(deepcopy(current_instances))

    match instance_meta:
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                nodes_for_group,
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=pre_allocated_listener_port,
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
                drafter_placement=drafter_placement,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = pre_allocated_listener_port
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=Cycle(node_ids=nodes_for_group),
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
                drafter_placement=drafter_placement,
            )

    # Multi-node placement WITHOUT an asymmetric drafter rank still loses
    # speculative decoding (mlx_lm doesn't run draft_model on TP/PP target
    # ranks today). Degrade-loud so operators see it without crawling logs;
    # the user's request still completes.
    if (
        len(selected_cycle) > 1
        and command.model_card.drafter_model_ids
        and drafter_placement is None
    ):
        logger.warning(
            f"Model {command.model_card.model_id} declares drafters "
            f"{list(command.model_card.drafter_model_ids)} but is being "
            f"placed across {len(selected_cycle)} nodes WITHOUT an asymmetric "
            "drafter rank. Speculative decoding is single-device only and "
            "will be disabled for this instance. To get the drafter speedup, "
            "either place a smaller quant on a single node OR list a separate "
            "drafter-eligible node in the model card's `drafter_eligible_nodes`."
        )

    return target_instances


def _select_drafter_placement(
    *,
    command: PlaceInstance,
    selected_cycle: Cycle,
    instance_meta: InstanceMeta,
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    instance_id: InstanceId,
    reserved_ports: frozenset[int],
    on_drafter_placement_degraded: (Callable[[DrafterPlacementDegraded], None] | None),
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> DrafterPlacement | None:
    """Pick a drafter-eligible node for asymmetric drafter placement.

    A drafter rank is appended to the parent ``mx.distributed`` group when
    *all* of the following hold:

      * The model card lists ``drafter_eligible_nodes``.
      * The card lists ``drafter_model_ids`` (otherwise there's nothing to
        run on the drafter rank).
      * At least one eligible node is alive in topology, NOT already a
        target rank, AND reachable from target rank 0 over the right
        transport (RDMA for ``MlxJaccl``; socket for ``MlxRing``).

    The fallback is loud-but-graceful: when none of the eligible nodes
    satisfies the constraints, the function emits a
    :class:`DrafterPlacementDegraded` event via
    ``on_drafter_placement_degraded`` and returns ``None``. The caller
    proceeds with the legacy symmetric topology, the user's request still
    completes, and the operator sees the degradation event surfaced in
    the dashboard / API stats so they know to fix the cluster (bring an
    eligible node online, free RAM, repair the network edge).

    The drafter is always assigned the **last rank** in the parent group
    (``len(selected_cycle)``). Target ranks split off into a subgroup at
    runtime via ``mx.distributed.Group.split``.
    """
    eligible_nodes = list(command.model_card.drafter_eligible_nodes)
    drafter_candidates = list(command.model_card.drafter_model_ids)
    if not eligible_nodes or not drafter_candidates:
        return None

    target_node_ids = list(selected_cycle.node_ids)
    fallback = _drafter_fallback(target_node_ids)

    alive_in_topology = set(topology.list_nodes())
    alive_eligible = [n for n in eligible_nodes if n in alive_in_topology]
    if not alive_eligible:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.NoEligibleNodeAvailable,
            fallback=fallback,
            detail=(
                f"None of {eligible_nodes} are present in topology "
                f"(known nodes: {sorted(alive_in_topology)})"
            ),
        )
        return None

    not_in_target = [n for n in alive_eligible if n not in target_node_ids]
    if not not_in_target:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.AllEligibleNodesInTargetCycle,
            fallback=fallback,
            detail=(
                f"All eligible nodes {alive_eligible} are already target "
                f"ranks ({target_node_ids}); no spare host available"
            ),
        )
        return None

    requires_rdma = instance_meta == InstanceMeta.MlxJaccl
    reachable: list[NodeId] = []
    for candidate in not_in_target:
        if _drafter_node_is_reachable(
            target_node_ids=target_node_ids,
            drafter_node=candidate,
            topology=topology,
            requires_rdma=requires_rdma,
        ):
            reachable.append(candidate)

    if not reachable:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero,
            fallback=fallback,
            detail=(
                f"No {'RDMA' if requires_rdma else 'socket'} path from target "
                f"ranks {target_node_ids} to any of {not_in_target}"
            ),
        )
        return None

    # Scan all reachable candidates and pick the first one with enough
    # advertised memory. Without this loop a single memory-constrained
    # node at ``reachable[0]`` would suppress asymmetric drafting even
    # when later candidates are viable; the topology scan above already
    # established directional reachability, so any of them is a legal
    # placement target. We also need to be defensive about the
    # degradation detail string here: ``_node_has_drafter_memory``
    # returns ``False`` both for "memory entry present and below floor"
    # and "memory entry absent" (e.g. a freshly-online node that hasn't
    # reported memory stats yet), so dereferencing
    # ``node_memory[drafter_node_id]`` for the detail string raises
    # ``KeyError`` and aborts placement instead of emitting the graceful
    # ``DrafterPlacementDegraded`` event we promised below.
    #
    # Codex P1 (PR #20 round-(N+10), placement.py:599): two-pass
    # selection. First, prefer a memory-eligible node that already
    # has *some* drafter candidate fully downloaded. Drafter
    # auto-download is explicitly skipped during planning and
    # ``DrafterRunner._handle_load`` raises ``FileNotFoundError``
    # when the chosen weights are absent, so picking a memory-
    # eligible-but-cold node ahead of a memory-eligible-and-warm
    # node fails the instance instead of using the available
    # weights. Second pass falls back to the first memory-eligible
    # node so a fully-cold cluster still gets a graceful runner-
    # level failure rather than a placement-time abort.
    eligible_candidates: list[NodeId] = []
    skipped_reasons: list[str] = []
    for candidate in reachable:
        if _node_has_drafter_memory(
            drafter_node=candidate,
            node_memory=node_memory,
            target_card=command.model_card,
        ):
            eligible_candidates.append(candidate)
        else:
            skipped_reasons.append(
                _describe_drafter_memory_skip(candidate, node_memory)
            )

    drafter_node_id: NodeId | None = None
    for candidate in eligible_candidates:
        if _node_has_any_drafter_on_disk(
            drafter_candidates=drafter_candidates,
            drafter_node_id=candidate,
            download_status=download_status,
        ):
            drafter_node_id = candidate
            break
    if drafter_node_id is None and eligible_candidates:
        # No memory-eligible node has the drafter weights on disk;
        # fall back to the first eligible node. The runner will
        # surface a load error and degrade gracefully -- the
        # placement-time pre-fix behavior, just preserved as a
        # second-pass fallback so warm clusters never lose to cold
        # ones.
        drafter_node_id = eligible_candidates[0]

    if drafter_node_id is None:
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.InsufficientDrafterMemory,
            fallback=fallback,
            detail=(
                f"No reachable drafter node satisfied the conservative "
                f"{_DRAFTER_MEMORY_FLOOR.in_gb:.1f}GB drafter estimate "
                f"({'; '.join(skipped_reasons)})"
            ),
        )
        return None

    # Codex P1 (PR #20 round-(N+3), placement.py:617): prefer a drafter
    # candidate that is already on the chosen drafter node's disk.
    # ``DrafterRunner._handle_load`` raises if the chosen weights are
    # absent and drafter auto-download is explicitly skipped during
    # planning, so cards that list ``[fast, fallback]`` previously
    # failed startup whenever ``fast`` was missing on the drafter node
    # despite ``fallback`` being present locally. Only the first
    # candidate that is fully ``DownloadCompleted`` for this drafter
    # node is preferred; if none are on-disk we fall back to
    # ``drafter_candidates[0]`` so the load failure (loud, with a
    # graceful degradation event from the runner) is no worse than the
    # pre-fix behavior.
    drafter_model_id = _select_available_drafter_model_id(
        drafter_candidates=drafter_candidates,
        drafter_node_id=drafter_node_id,
        download_status=download_status,
    )
    drafter_runner_id = RunnerId()
    drafter_rank = len(selected_cycle)

    # Resolve target rank 0's IP from the drafter's perspective. Target
    # rank 0 == selected_cycle.node_ids[0] by construction (every shard
    # assigner enumerates the cycle in order; ``device_rank`` is the
    # enumeration index). We pick the same priority order ``ring`` uses
    # (Thunderbolt-bridge first, then ethernet, then wifi) because the
    # drafter wire is small fixed-size frames where TCP latency over a
    # direct cable beats RDMA setup latency every time.
    #
    # ``find_ip_prioritised`` returns the SINK end of connections going
    # ``node_id -> other_node_id``: i.e. the address ``other_node_id``
    # advertises for that direction. We want the address target rank 0
    # advertises *to the drafter*, so ``other_node_id`` is the target
    # and ``node_id`` is the drafter.
    target_rank_zero = selected_cycle.node_ids[0]
    drafter_socket_host = find_ip_prioritised(
        drafter_node_id,
        target_rank_zero,
        topology,
        node_network,
        ring=True,
    )
    if drafter_socket_host is None:
        # ``_drafter_node_is_reachable`` already checked the directional
        # edge; if topology says reachable but no IP is exposed, the
        # node is misconfigured. Bail out loudly via degradation rather
        # than picking ``0.0.0.0`` (which the drafter cannot dial).
        _emit_drafter_degraded(
            on_drafter_placement_degraded,
            command=command,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero,
            fallback=fallback,
            detail=(
                f"Target rank 0 ({target_rank_zero}) has no IP address "
                f"reachable from drafter node {drafter_node_id} in topology"
            ),
        )
        return None
    # Codex P1 (PR #20, placement.py:711): pick a kernel-vetted-free
    # port for the drafter listener. ``random_ephemeral_port`` (since
    # PR #20 round-(N+12)) asks the master's kernel for a free port
    # via ``bind(("", 0))`` rather than picking uniformly random and
    # hoping. Same-host deploys (master == target rank 0's host) are
    # therefore collision-free; cross-host deploys still rely on the
    # remote target's kernel having that port free at
    # ``bind_target_listener`` time, but that path now raises a self-
    # describing ``OSError`` so the failure surfaces as a clear
    # cross-host port collision rather than a generic "Address
    # already in use". A two-phase "target binds, advertises back"
    # protocol would close the cross-host gap entirely; that requires
    # changing ``DrafterPlacement``'s wire schema and is tracked for
    # a follow-up PR.
    #
    # Codex P2 (PR #21 round 3): both rank-0 listener ports must avoid
    # each other AND the caller-supplied ``reserved_ports`` set, which
    # carries the per-meta listener port (jaccl coordinator port or
    # ring ephemeral port) that the placement entry point pre-allocates.
    # Pre-fix the collision-avoidance loop only checked
    # ``target_peer_socket_port != drafter_socket_port`` and missed
    # those sibling listeners, so rank 0 occasionally hit
    # ``EADDRINUSE`` during runner bootstrap (drafter accept loop in
    # ``_maybe_accept_drafter_socket`` versus target peer fanout in
    # ``_maybe_setup_target_peer_fanout``).
    drafter_socket_port = random_ephemeral_port_excluding(reserved_ports)
    # Inter-target-peer wire: target rank 0 binds a separate ephemeral
    # port for the spec-decode int-broadcast fanout (drafts in / sampled
    # tokens out). Decoupled from the drafter port because both bind on
    # rank 0 and a single port can only accept one connection class
    # cleanly. Each non-zero target rank dials the IP rank 0 advertises
    # *to that peer* -- different peers may reach rank 0 over different
    # interfaces (e.g. a Thunderbolt /30 mesh exposes a unique IP per
    # node pair). The map below resolves those per-peer IPs once at
    # placement time so workers don't re-do the topology dance at
    # bootstrap.
    target_peer_socket_port = random_ephemeral_port_excluding(
        reserved_ports | {drafter_socket_port}
    )
    # Keys stored as strings so the dict round-trips through the
    # event-router JSON wire (JSON has no int dict keys, and pydantic
    # strict mode rejects str keys for a ``dict[int, _]`` field at
    # re-validation). Consumers stringify the rank before lookup.
    target_peer_hosts_by_rank: dict[str, str] = {}
    for peer_rank, peer_node_id in enumerate(selected_cycle.node_ids):
        if peer_rank == 0:
            continue
        peer_view_of_rank_zero = find_ip_prioritised(
            peer_node_id,
            target_rank_zero,
            topology,
            node_network,
            ring=True,
        )
        if peer_view_of_rank_zero is None:
            # Same fail-loud rationale as the drafter IP: target rank 0
            # is unreachable from a peer in topology, so the spec-decode
            # int-broadcast wire cannot be brought up. Falling back to
            # the legacy ``mx.distributed`` broadcast would re-introduce
            # the JACCL int/float wire-conflation bug. Degrade to no
            # drafter so the user still gets generation, just at
            # standard (non-speculative) speed.
            _emit_drafter_degraded(
                on_drafter_placement_degraded,
                command=command,
                instance_id=instance_id,
                target_node_ids=target_node_ids,
                eligible_nodes=eligible_nodes,
                reason=DrafterPlacementDegradationReason.NoReachablePathFromTargetRankZero,
                fallback=fallback,
                detail=(
                    f"Target rank 0 ({target_rank_zero}) has no IP address "
                    f"reachable from peer target rank {peer_rank} "
                    f"(node {peer_node_id}) in topology"
                ),
            )
            return None
        target_peer_hosts_by_rank[str(peer_rank)] = peer_view_of_rank_zero
    return DrafterPlacement(
        drafter_node_id=drafter_node_id,
        drafter_runner_id=drafter_runner_id,
        drafter_model_id=drafter_model_id,
        drafter_rank=drafter_rank,
        drafter_socket_host=drafter_socket_host,
        drafter_socket_port=drafter_socket_port,
        target_peer_socket_port=target_peer_socket_port,
        target_peer_hosts_by_rank=target_peer_hosts_by_rank,
    )


def _select_available_drafter_model_id(
    *,
    drafter_candidates: Sequence[ModelId],
    drafter_node_id: NodeId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> ModelId:
    """Pick a drafter model id, preferring an on-disk candidate.

    Iterates the card's ``drafter_model_ids`` in order and returns the
    first one that is fully downloaded on ``drafter_node_id``. If none
    are on disk, returns ``drafter_candidates[0]`` so the failure mode
    is unchanged from the pre-fix behavior (the runner will surface a
    load error and graceful degradation).

    The caller has already verified ``drafter_candidates`` is non-empty.
    """
    assert drafter_candidates, (
        "_select_available_drafter_model_id requires drafter_candidates"
    )
    node_progresses = download_status.get(drafter_node_id, ())
    completed_on_drafter = {
        progress.shard_metadata.model_card.model_id
        for progress in node_progresses
        if isinstance(progress, DownloadCompleted)
    }
    for candidate in drafter_candidates:
        if candidate in completed_on_drafter:
            return candidate
    return drafter_candidates[0]


def _node_has_any_drafter_on_disk(
    *,
    drafter_candidates: Sequence[ModelId],
    drafter_node_id: NodeId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    """Return ``True`` if any drafter candidate is fully downloaded on the node.

    Codex P1 (PR #20 round-(N+10), placement.py:599): used as the
    primary tiebreaker in drafter-node selection so a memory-eligible
    "warm" node (one with at least one drafter on disk) wins over a
    memory-eligible "cold" node, preventing
    ``DrafterRunner._handle_load`` from failing the instance with
    ``FileNotFoundError`` when a viable warm node existed but a cold
    node was picked first.
    """
    node_progresses = download_status.get(drafter_node_id, ())
    completed_on_drafter = {
        progress.shard_metadata.model_card.model_id
        for progress in node_progresses
        if isinstance(progress, DownloadCompleted)
    }
    return any(candidate in completed_on_drafter for candidate in drafter_candidates)


def _drafter_fallback(target_node_ids: list[NodeId]) -> str:
    """``single_device_drafter`` when target is single-node, else ``no_drafter``.

    Multi-node target with no asymmetric drafter rank can't host the
    drafter at all (mlx_lm spec decode is single-device); single-node
    target falls back to in-process drafter as before.
    """
    return "single_device_drafter" if len(target_node_ids) == 1 else "no_drafter"


def _emit_drafter_degraded(
    callback: Callable[[DrafterPlacementDegraded], None] | None,
    *,
    command: PlaceInstance,
    instance_id: InstanceId,
    target_node_ids: list[NodeId],
    eligible_nodes: list[NodeId],
    reason: DrafterPlacementDegradationReason,
    fallback: str,
    detail: str,
) -> None:
    logger.error(
        f"Drafter placement degraded for {command.model_card.model_id} "
        f"({reason.value}): {detail}; falling back to {fallback}"
    )
    if callback is None:
        return
    assert fallback in ("single_device_drafter", "no_drafter")
    callback(
        DrafterPlacementDegraded(
            model_id=command.model_card.model_id,
            instance_id=instance_id,
            target_node_ids=target_node_ids,
            eligible_nodes=eligible_nodes,
            reason=reason,
            fallback=fallback,
            detail=detail,
        )
    )


def _drafter_node_is_reachable(
    *,
    target_node_ids: list[NodeId],
    drafter_node: NodeId,
    topology: Topology,
    requires_rdma: bool,  # retained for ABI parity; unused under v3+ wire
) -> bool:
    """Drafter must be socket-reachable from target rank 0 only.

    Under the v3+ asymmetric wire (this module's :class:`DrafterPlacement`
    + ``RemoteTransport``) the drafter is NOT a member of the target
    ranks' ``mx.distributed.Group``. The only edge the wire actually
    needs is a TCP socket from the drafter node DIALING target rank 0.
    Every other "all target ranks must reach drafter" requirement from
    the v2 wire (where drafter was an mx.distributed peer) is gone.

    ``requires_rdma`` is accepted but ignored: the drafter wire is plain
    TCP regardless of whether the target ranks talk to each other over
    JACCL/RDMA or ring/TCP. The argument is retained so callers don't
    need to rev simultaneously with this module.

    Codex P2 (PR #20 round-(N+3), placement.py:746): pre-fix this check
    required socket edges in BOTH directions
    (``target_rank_zero -> drafter`` and ``drafter -> target_rank_zero``),
    but the v3 wire only needs the drafter -> target rank 0 dial. In
    topologies that record only one directed edge (the side that
    initiated discovery), placement falsely emitted
    ``NoReachablePathFromTargetRankZero`` and disabled asymmetric
    drafting even though the actual TCP dial would work.

    Codex P1 (PR #20 round-(N+7), placement.py): the round-(N+3) fix
    relaxed reachability to "either direction", but the runtime wire
    is unidirectional: the drafter ALWAYS dials target rank 0
    (target rank 0 listens, drafter connects). ``Topology
    .get_all_connections_between(source, sink)`` is itself
    directional, so a topology that only records
    ``target -> drafter`` edges (target reached drafter during
    discovery, but drafter never directly dialed target) is NOT a
    valid drafter-to-target dial path. Pre-fix the relaxed check
    admitted such topologies; placement then proceeded,
    ``find_ip_prioritised(drafter, target, ...)`` may have returned
    an address anyway, and bootstrap failed later during the
    drafter's actual ``connect()`` instead of emitting the intended
    graceful ``DrafterPlacementDegraded`` fallback. Validate ONLY
    the drafter -> target rank 0 direction so the placement-time
    decision matches the runtime dial direction.
    """
    del requires_rdma  # documented above; the v3 wire is socket-only
    if not target_node_ids:
        return False
    target_rank_zero = target_node_ids[0]
    socket_check: Callable[[object], bool] = lambda c: isinstance(  # noqa: E731
        c, SocketConnection
    )
    # Validate the drafter -> target rank 0 direction only: this
    # matches the runtime wire's actual dial direction (drafter
    # initiates, target rank 0 listens). The reverse direction is
    # not interchangeable because ``Topology
    # .get_all_connections_between`` is directional.
    drafter_to_target = topology.get_all_connections_between(
        drafter_node, target_rank_zero
    )
    return any(socket_check(c) for c in drafter_to_target)


# Conservative floor for the drafter's wired-memory bump. The drafter
# weights are usually 1-5GB (e.g. gemma-4-e2b @ 8-bit ~ 2GB), but during
# load the runner may briefly hold the safetensors mmap + decompression
# buffers; bake in headroom so placement doesn't pick a node that will
# OOM at warmup. If the drafter on disk is larger than this floor the
# runner's own ``set_wired_limit_for_model`` will catch it; this is just
# a placement-time sanity check.
_DRAFTER_MEMORY_FLOOR = Memory.from_gb(6.0)


def _node_has_drafter_memory(
    *,
    drafter_node: NodeId,
    node_memory: Mapping[NodeId, MemoryUsage],
    target_card: ModelCard,
) -> bool:
    del target_card  # reserved for future per-drafter sizing
    if drafter_node not in node_memory:
        return False
    return node_memory[drafter_node].ram_available >= _DRAFTER_MEMORY_FLOOR


def _describe_drafter_memory_skip(
    drafter_node: NodeId,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> str:
    """One-line explanation of why ``drafter_node`` was rejected by
    :func:`_node_has_drafter_memory`.

    Used to compose the degradation event detail when the entire
    reachable candidate list fails the memory floor. Operators reading
    ``DrafterPlacementDegraded`` events need to know whether a node
    was skipped because it hadn't reported memory yet (transient,
    safe to retry once stats arrive) versus reported and below floor
    (persistent, needs a different placement). Emitting both states
    distinctly keeps that signal in the event stream.
    """
    if drafter_node not in node_memory:
        return f"node {drafter_node} has not reported memory stats yet"
    available = node_memory[drafter_node].ram_available
    return (
        f"node {drafter_node} has {available.in_gb:.1f}GB available "
        f"(< {_DRAFTER_MEMORY_FLOOR.in_gb:.1f}GB floor)"
    )


def _prefer_socket_reachable_rank_zero(cycle: Cycle, topology: Topology) -> Cycle:
    """Rotate multi-node placements so rank 0 is easiest for peers to reach.

    MLX ring and JACCL both make rank 0 the listener/coordinator. Discovery can
    produce RDMA-only edges in one direction and socket control-plane edges in
    another, so putting a node with advertised inbound socket edges at rank 0
    avoids assigning the listener role to a machine peers cannot dial.
    """
    if len(cycle) <= 1:
        return cycle

    inbound_socket_edges: dict[NodeId, int] = {node_id: 0 for node_id in cycle}
    for connection in topology.list_connections():
        if connection.sink not in inbound_socket_edges:
            continue
        if isinstance(connection.edge, SocketConnection):
            inbound_socket_edges[connection.sink] += 1

    best_index = max(
        range(len(cycle.node_ids)),
        key=lambda index: (inbound_socket_edges[cycle.node_ids[index]], -index),
    )
    if best_index == 0:
        return cycle
    return Cycle(node_ids=cycle.node_ids[best_index:] + cycle.node_ids[:best_index])


def _node_memory_with_total_capacity(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Mapping[NodeId, MemoryUsage]:
    return {
        node_id: (
            memory_usage.model_copy(update={"ram_available": memory_usage.ram_total})
            if node_id in cycle.node_ids
            else memory_usage
        )
        for node_id, memory_usage in node_memory.items()
    }


def _validate_jaccl_thunderbolt_ipv4_paths(
    cycle: Cycle,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> None:
    """Reject the placement only when we have *positive evidence* that
    a node lacks a TB-IPv4 peer path.

    Codex P1 (PR #11 round 5): ``State.node_network`` is populated by
    a best-effort async watcher and starts empty on cold-boot, so
    ``node_network.get(node_id)`` returning ``None`` is not the same
    thing as ``the node has no Thunderbolt interface``. The original
    guard collapsed both into "missing" and rejected ``MlxJaccl``
    placements whenever the gatherer hadn't run yet (or failed
    transiently for a node), even on clusters with healthy RDMA
    topology. We now distinguish the two:

    * ``known_no_path`` -- the node has gathered network info and
      none of its interfaces satisfy the Thunderbolt IPv4 predicate.
      That is genuine misconfiguration; raise with the actionable
      ``bb rdma repair`` guidance.
    * ``unknown`` -- the node has no entry in ``node_network`` (yet).
      We let placement proceed because the topology-derived RDMA
      edge already attests that some real connection exists; the
      JACCL backend will surface a clearer per-link error if the
      address turns out to be unusable at bind time.
    """
    missing_nodes = [
        node_id
        for node_id in cycle.node_ids
        if _node_has_or_lacks_known_jaccl_path(node_network, node_id) == "known_no_path"
    ]
    if missing_nodes:
        raise ValueError(
            "Requested RDMA (MlxJaccl), but the selected nodes do not advertise "
            "MLX/JACCL Thunderbolt IPv4 peer paths. Run `bb rdma repair all` and "
            "`bb rdma jaccl-status all`, then retry. Missing nodes: "
            + ", ".join(str(node_id) for node_id in missing_nodes)
        )


def _node_has_or_lacks_known_jaccl_path(
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_id: NodeId,
) -> Literal["has_path", "known_no_path", "unknown"]:
    """Three-valued JACCL preflight verdict for a single node.

    Returns ``"unknown"`` when:

    * ``node_id`` has no entry in ``node_network`` at all (the
      best-effort gatherer hasn't reported yet on this node), OR
    * the entry exists but **interface typing is missing** for every
      interface (e.g. the ``networksetup -listallhardwareports``
      parse failed on the gatherer side, so we have IP addresses
      but no ``interface_type`` field to classify them as
      thunderbolt vs ethernet vs wifi).

    Returns ``"has_path"`` when at least one Thunderbolt-style
    interface advertises a routable IPv4. Returns ``"known_no_path"``
    when typing IS available (at least one interface has a non-None,
    non-``"unknown"`` ``interface_type``) but no qualifying interface
    exists -- that's positive evidence of misconfiguration and we
    surface the actionable ``bb rdma repair`` error.

    Codex P1 (PR #11 round-(N+2)): pre-fix this helper collapsed
    "interfaces present but typing unavailable" into ``known_no_path``
    and rejected placement, even though we had no positive evidence
    that the node actually lacked a Thunderbolt path. With this
    refinement, the gatherer's partial-success/parse-failure case is
    treated as ``unknown`` and placement proceeds; the JACCL backend
    will surface a clearer per-link error if the IP turns out to be
    unusable at bind time.
    """
    info = node_network.get(node_id)
    if info is None:
        return "unknown"
    if _has_jaccl_thunderbolt_ipv4(info):
        return "has_path"
    if _interface_typing_is_missing(info):
        return "unknown"
    return "known_no_path"


# Match the exact set of macOS interface names that can plausibly be
# a Thunderbolt link or bridge:
#
# * ``en2`` ... ``en9`` and ``en10`` ... ``en9999`` -- ``en0`` and
#   ``en1`` are reserved for Wi-Fi/primary NIC by Apple convention
#   (also encoded in
#   :func:`exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup`,
#   which classifies any other ``en\\d+`` as ``"maybe_ethernet"``
#   because Apple Silicon Thunderbolt bridges always live on
#   ``en2``/``en3``/``en4``). Excluding ``en0``/``en1`` prevents the
#   permissive fallback from firing on a Wi-Fi-only node whose
#   primary ``en0`` happened to land in ``"unknown"`` typing
#   (e.g. due to a transient ``networksetup`` parse failure).
# * ``bridge0`` ... ``bridge99`` -- ``bridge0`` is the canonical
#   macOS Thunderbolt Bridge service device, but
#   :func:`exo.utils.info_gatherer.info_gatherer._get_bridge_services`
#   and :func:`_find_thunderbolt_bridge` enumerate **arbitrary**
#   ``bridge\\d+`` devices and intersect their member set with the
#   Thunderbolt hardware-port device list -- a user with multiple
#   bridges (or a system that already had ``bridge0`` claimed by
#   another service) can have a real Thunderbolt Bridge exposed as
#   ``bridge1``/``bridge2``/etc. Codex P1 (PR #11 round-(N+15),
#   placement.py:567) called out that hard-coding ``bridge0`` here
#   rejects those legitimate configurations. We accept
#   ``bridge[0-9]{1,2}`` (i.e. ``bridge0``..``bridge99``); macOS
#   Internet Sharing reserves ``bridge100``+ for NAT/Parallels/
#   VirtualBox VM stacks (see ``man 8 bridge``), so excluding the
#   3-digit range still keeps VM-stack bridges out of the
#   permissive fallback.
_THUNDERBOLT_CANDIDATE_INTERFACE_NAME = re.compile(
    r"^(en[2-9]|en[1-9]\d+|bridge[0-9]{1,2})$"
)


def _is_plausible_thunderbolt_candidate(
    interface: NetworkInterfaceInfo,
) -> bool:
    """Return whether an ``"unknown"``-typed interface could plausibly
    be a Thunderbolt bridge whose hardware-port line wasn't classified.

    The heuristic limits the permissive ``unknown``-typing fallback to
    interfaces whose names exactly match the Apple/macOS Thunderbolt
    naming convention (see :data:`_THUNDERBOLT_CANDIDATE_INTERFACE_NAME`)
    AND that advertise a routable IPv4
    (:func:`_is_routable_jaccl_ipv4` filters loopback / link-local /
    unset addresses).

    Tunnel/VPN adapters (``utun*``, ``tun*``, ``tap*``, ``wg*``,
    ``gif*``, ``stf*``, ``ipsec*``), Apple Wireless Direct Link
    (``awdl*`` / ``llw*``), packet-capture (``pktap*``), loopback
    (``lo*``), Internet-Sharing/VM-stack bridges
    (``bridge100``, ``bridge101``, ...), and the Wi-Fi/primary
    leaves (``en0``, ``en1``) all fail the name check, so a
    Wi-Fi-only node that happens to have a Tailscale ``utun3``
    link or a Parallels ``bridge100`` with a routable IPv4 no
    longer slips through the JACCL preflight.

    Codex history:

    Round-(N+13) introduced the helper with regex ``^en\\d+$`` --
    too narrow because ``info_gatherer`` explicitly models the
    macOS Thunderbolt Bridge as ``bridge0`` and that device does
    not appear in ``networksetup -listallhardwareports``.

    Round-(N+14) widened to ``^(en|bridge)\\d+$`` to admit
    ``bridge0``. Codex flagged (P1, PR #11 round-(N+14),
    placement.py:548) that this re-admitted ``bridge100``
    (Parallels Desktop), ``bridge101`` (Parallels), arbitrary
    ``bridge\\d+`` from VirtualBox/VMware, AND ``en0``/``en1``
    (Wi-Fi/primary), so the Wi-Fi-only-on-VPN attack surface
    re-opened with VM-stack bridges as the new bypass vector.

    Round-(N+15) narrowed to ``^(en[2-9]|en[1-9]\\d+|bridge0)$``
    (excludes ``en0``/``en1`` and rejects every non-``bridge0``
    bridge). Codex flagged (P1, PR #11 round-(N+15),
    placement.py:567) that the gatherer's
    :func:`exo.utils.info_gatherer.info_gatherer._find_thunderbolt_bridge`
    operates on **arbitrary** ``bridgeX`` devices -- a user with
    multiple bridge services (or one whose ``bridge0`` is already
    claimed by another stack) can have a real Thunderbolt Bridge
    exposed as ``bridge1``/``bridge2``/etc., so hard-coding
    ``bridge0`` rejected legitimate TB-only configurations.

    Round-(N+16) (this commit) widens the bridge half to
    ``bridge[0-9]{1,2}`` (i.e. ``bridge0``..``bridge99``) so the
    real-Thunderbolt indices below the macOS Internet-Sharing
    reservation (``bridge100``+) are accepted, while the VM-stack
    bridges in the 3-digit range remain excluded.
    """
    if not _THUNDERBOLT_CANDIDATE_INTERFACE_NAME.match(interface.name):
        return False
    return _is_routable_jaccl_ipv4(interface.ip_address)


def _interface_typing_is_missing(network_info: NodeNetworkInfo) -> bool:
    """Heuristic for "the gatherer couldn't classify this node's
    interfaces" vs "the gatherer reports a node with no TB interfaces".

    Returns ``True`` when:

    * ``network_info`` has no interfaces at all (gatherer reported
      nothing), OR
    * **every** interface has ``interface_type == "unknown"`` (the
      gatherer's parse of ``networksetup -listallhardwareports``
      failed across the board), OR
    * **some** interface has ``interface_type == "unknown"`` AND
      passes :func:`_is_plausible_thunderbolt_candidate` (interface
      name matches ``en\\d+`` AND has a routable IPv4) -- this
      narrows the permissive fallback to genuine TB-bridge
      candidates rather than VPN/tunnel adapters with routable IPs.

    Returns ``False`` when typing IS available for every routable
    candidate -- the node has positive evidence of bad config and
    placement should reject with the actionable
    ``bb rdma repair`` error.

    Codex history:

    Round-(N+2) introduced the helper using ``all(...)`` --
    correctly handles total parse failure but rejects mixed-typing
    nodes (Wi-Fi typed plus unparsed TB bridge).

    Round-(N+11) widened to ``any(interface.interface_type ==
    "unknown" ...)`` to admit the partial-typing case. That was
    too permissive: ``get_network_interfaces`` assigns ``"unknown"``
    to interfaces not present in ``networksetup`` output (loopback,
    tunnel, etc.) so virtually every node had at least one
    unknown interface and the JACCL preflight reverted to
    permissive behavior on misconfigured clusters too -- the user
    only saw the runtime JACCL failure later.

    Round-(N+12) coupled the unknown check with routable-IPv4
    candidacy. That filtered out loopback and link-local interfaces
    but VPN/tunnel adapters (``utun*`` from Tailscale/Wireguard)
    are typed as ``"unknown"`` AND have routable ``10.x``/``100.x``
    IPv4s, so the permissive branch still fired on Wi-Fi-only nodes
    with VPNs and bypassed the preflight (Codex P1 PR #11
    round-(N+12) follow-up at placement.py:597).

    Round-(N+13) (this commit) further restricts the permissive
    fallback to the Apple ``en\\d+`` naming convention via
    :func:`_is_plausible_thunderbolt_candidate`. ``utun*`` /
    ``wg*`` / ``tun*`` / ``awdl*`` / ``lo*`` no longer satisfy the
    plausibility check, so a Wi-Fi-only node with a Tailscale tunnel
    correctly resolves to ``known_no_path`` (and the actionable
    ``bb rdma repair`` error). The legitimate Thunderbolt-bridge
    case -- ``en3`` with a routable IPv4 whose hardware-port line
    failed to parse -- still defers to ``unknown``.
    """
    if not network_info.interfaces:
        return True
    if all(
        interface.interface_type == "unknown" for interface in network_info.interfaces
    ):
        return True
    return any(
        interface.interface_type == "unknown"
        and _is_plausible_thunderbolt_candidate(interface)
        for interface in network_info.interfaces
    )


def _has_jaccl_thunderbolt_ipv4(network_info: NodeNetworkInfo | None) -> bool:
    """Return whether the node advertises at least one Thunderbolt-style
    routable IPv4 interface usable as a JACCL peer path.

    Why ``maybe_ethernet`` is accepted alongside ``thunderbolt``:
    :func:`exo.utils.info_gatherer.system_info._get_interface_types_from_networksetup`
    reclassifies any ``en*`` adapter that isn't ``en0`` / ``en1`` to
    ``"maybe_ethernet"`` regardless of what ``networksetup
    -listallhardwareports`` reports the hardware port as. On every
    cluster machine we ship, the Thunderbolt bridge sits on ``en2`` /
    ``en3`` / ``en4``, so its interface_type ends up as
    ``"maybe_ethernet"`` even though the underlying hardware is in
    fact a Thunderbolt link. Restricting the preflight to
    ``interface_type == "thunderbolt"`` rejected those (correctly
    repaired) bridges as missing, breaking placement on real
    deployments. The upstream guard ``instance_meta ==
    InstanceMeta.MlxJaccl`` already requires an RDMA-connected cycle
    (libp2p only forms RDMA edges over Thunderbolt on Apple Silicon),
    so accepting ``maybe_ethernet`` here cannot let a true LAN
    ethernet sneak past -- nodes without TB hardware would have been
    filtered upstream by the missing RDMA edge.
    """
    if network_info is None:
        return False
    return any(
        interface.interface_type in ("thunderbolt", "maybe_ethernet")
        and _is_routable_jaccl_ipv4(interface.ip_address)
        for interface in network_info.interfaces
    )


def _is_routable_jaccl_ipv4(ip_address: str) -> bool:
    """Return True iff ``ip_address`` is a syntactically-valid, unicast
    IPv4 address that's plausibly usable as a JACCL peer path.

    A valid IPv4 here is *exactly* four numeric octets in 0..255
    separated by dots, and the first octet must fall in the unicast
    range (1..223). We deliberately do not use ``ipaddress.IPv4Address``
    because that class accepts a few legacy alternate forms (e.g.
    integer-only ``"3232235521"``) that we don't want to allow as
    Thunderbolt peer paths -- the upstream gatherer always reports
    dotted-quad form, so anything else is malformed interface data
    we'd rather reject fast than parse leniently.

    Octet validation matters because malformed strings like
    ``"999.1.1.1"`` or ``"1..2.3"`` would otherwise satisfy the
    preflight (they have four split components on the dot delimiter)
    and let an ``MlxJaccl`` placement reach the runtime layer, where
    it'd fail with a far less actionable error when the JACCL backend
    tries to resolve unusable peer addresses.

    Non-unicast ranges rejected (in addition to the loopback /
    link-local / all-zero prefixes already filtered):

    - ``224.0.0.0/4`` (multicast 224..239) -- a peer path can never
      be a multicast group;
    - ``240.0.0.0/4`` (reserved / experimental 240..254) -- not
      assigned for general use, including the misconfiguration
      target ranges some Thunderbolt utilities default to;
    - ``255.255.255.255`` (limited broadcast) -- specifically
      called out by the codex review because the previous rule
      accepted it as a "valid IPv4" even though it cannot be a
      peer path.

    The unicast cap at 223 covers all three above (Class D starts at
    224, Class E at 240, broadcast falls inside Class E).
    """
    if ":" in ip_address:
        return False
    if ip_address.startswith(("0.", "127.", "169.254.")):
        return False
    octets = ip_address.split(".")
    if len(octets) != 4:
        return False
    parsed: list[int] = []
    for octet in octets:
        # Reject empty fields ("1..2.3"), non-digit characters, leading
        # whitespace, signs, etc. We don't allow leading zeros either
        # ("01.2.3.4"), since networksetup never emits them and they
        # historically trigger octal-style parsing in some libc tools.
        #
        # Codex P3 (PR #11 round 4): ``str.isdigit()`` returns True for
        # Unicode digit characters (e.g. superscript digits like
        # ``"\u00b2"``) that ``int()`` then rejects with
        # ``ValueError``. The earlier guard let those through to
        # ``int(octet)``, so a malformed network string from a
        # corrupted info-gatherer payload would raise instead of
        # cleanly returning False, aborting placement instead of
        # surfacing the routine "no eligible cycle" path. Restrict to
        # the ASCII 0-9 range explicitly.
        if not octet.isascii() or not octet.isdigit():
            return False
        if len(octet) > 1 and octet.startswith("0"):
            return False
        # Codex P2 (PR #11 round-(N+8), placement.py): even after the
        # ASCII-digit guard, ``int(octet)`` can still raise
        # ``ValueError`` because CPython enforces a string-conversion
        # digit limit (``sys.set_int_max_str_digits``, default 4300).
        # A pathological ``node_network`` payload such as
        # ``"9" * 4301 + ".1.1.1"`` would reach this line and abort
        # the placement preflight instead of returning False. The
        # contract for this helper is "never raise on malformed
        # network payloads", so cap octet length at 3 (any IPv4 octet
        # in the range 0..255 fits in three ASCII digits) before
        # attempting conversion.
        if len(octet) > 3:
            return False
        value = int(octet)
        if value < 0 or value > 255:
            return False
        parsed.append(value)
    # First octet in unicast range only (1..223). 0.* is already
    # caught above by the prefix block, but we re-check the full
    # range here for clarity and because the unicast bound rejects
    # multicast (224..239), reserved/experimental (240..254), and
    # broadcast (255). The directed-broadcast case (e.g.
    # ``192.168.10.255``) on a /24 is not generally distinguishable
    # without subnet info -- we accept it as syntactically unicast
    # and let the JACCL backend reject it on actual bind.
    return 1 <= parsed[0] <= 223


def _order_asymmetric_tensor_cycle(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    topology: Topology,
) -> Cycle:
    """Order an asymmetric TP cycle with the largest reachable node as rank 0."""
    ordered_cycle = Cycle(
        node_ids=sorted(
            cycle.node_ids,
            key=lambda node_id: node_memory[node_id].ram_available.in_bytes,
            reverse=True,
        )
    )
    preferred_cycle = _prefer_socket_reachable_rank_zero(ordered_cycle, topology)
    if preferred_cycle.node_ids[0] != ordered_cycle.node_ids[0]:
        raise ValueError(
            "Asymmetric tensor parallelism requires the largest-memory rank-0 "
            "node to be socket-reachable"
        )
    return ordered_cycle


def _asymmetric_tensor_rank_zero_is_socket_reachable(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    topology: Topology,
) -> bool:
    try:
        _order_asymmetric_tensor_cycle(
            cycle=cycle,
            node_memory=node_memory,
            topology=topology,
        )
    except ValueError:
        return False
    return True


def auto_place_prefill_siblings(
    *,
    decode_instance_id: InstanceId,
    decode_instance: Instance,
    model_card: ModelCard,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
) -> tuple[dict[InstanceId, Instance], list[InstanceId]]:
    """Place single-rank prefill-only siblings on each viable eligible node.

    Returns a tuple of ``(new_instances, new_prefill_instance_ids)`` where
    ``new_instances`` maps newly-created prefill ``InstanceId`` to its
    placement and ``new_prefill_instance_ids`` preserves placement order.
    Both are empty when ``model_card.prefill_eligible_nodes`` is empty,
    when no candidate is alive in topology, or when every candidate fails
    feasibility (insufficient RAM, no socket reachability, etc.) -- the
    decode instance still comes up; the caller emits no
    ``InstanceLinkCreated`` and the user's traffic prefills locally on
    the target rank.

    The recursive ``place_instance`` call is invoked with a sanitised
    model card (drafter and prefill eligibility cleared) and
    ``allowed_nodes={candidate}`` to force a single-node Pipeline / PP=1
    placement. We do NOT inherit drafter placement onto prefill siblings:
    the prefill role is a pure remote-prefill server (TCP-only via
    :class:`~exo.worker.disaggregated.server.PrefillServer`), so it
    needs the target weights but not the drafter pair.
    """
    eligible = list(dict.fromkeys(model_card.prefill_eligible_nodes))
    if not eligible:
        return {}, []

    decode_nodes: set[NodeId] = set(
        decode_instance.shard_assignments.node_to_runner.keys()
    )
    if decode_instance.drafter_placement is not None:
        decode_nodes.add(decode_instance.drafter_placement.drafter_node_id)

    alive = set(topology.list_nodes())

    candidates = [
        node_id
        for node_id in eligible
        if node_id in alive and node_id not in decode_nodes
    ]
    if not candidates:
        logger.warning(
            f"Auto-prefill placement skipped for decode {decode_instance_id}: "
            f"no eligible node alive AND outside the decode cycle. "
            f"eligible={eligible} decode_nodes={sorted(decode_nodes)} "
            f"alive={sorted(alive)}"
        )
        return {}, []

    # Sanitise the recursive card so the prefill-only sibling does not
    # itself recursively spawn drafters or further prefill siblings.
    prefill_card = model_card.model_copy(
        update={
            "drafter_eligible_nodes": [],
            "drafter_model_ids": [],
            "prefill_eligible_nodes": [],
        }
    )

    placed: dict[InstanceId, Instance] = {}
    placed_ids: list[InstanceId] = []
    accumulating_instances: dict[InstanceId, Instance] = dict(current_instances)

    for candidate_node in candidates:
        sub_command = PlaceInstance(
            model_card=prefill_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        )
        try:
            sub_placement = place_instance(
                sub_command,
                topology,
                accumulating_instances,
                node_memory,
                node_network,
                allowed_nodes={candidate_node},
                download_status=download_status,
            )
        except ValueError as err:
            logger.warning(
                f"Auto-prefill skip {candidate_node} for decode "
                f"{decode_instance_id}: {err}"
            )
            continue

        new_ids_this_round = [
            iid for iid in sub_placement if iid not in accumulating_instances
        ]
        if not new_ids_this_round:
            logger.warning(
                f"Auto-prefill on {candidate_node} returned no new "
                f"instance for decode {decode_instance_id}; skipping"
            )
            continue
        for iid in new_ids_this_round:
            placed[iid] = sub_placement[iid]
            placed_ids.append(iid)
            accumulating_instances[iid] = sub_placement[iid]

    return placed, placed_ids


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
