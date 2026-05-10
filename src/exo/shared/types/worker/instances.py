from enum import Enum
from typing import final

from pydantic import Field, model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, ModelId, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


@final
class DrafterPlacement(FrozenModel):
    """Locator for an asymmetric drafter rank inside an :class:`Instance`.

    The drafter runs on a separate node from the target ranks. It is
    intentionally NOT a member of the target ranks'
    ``mx.distributed.Group``: the target group is target-only, and
    drafter <-> target IPC flows over a direct TCP socket established
    at instance bootstrap. Decoupling the drafter from
    ``mx.distributed`` lets target ranks of any size use TP/PP
    collectives without requiring ``Group.split`` (which jaccl/ring
    backends do not implement on Apple Silicon).

    Convention: ``drafter_rank`` is preserved as a logical placement
    index (always equal to ``len(target_ranks)``) for telemetry and
    tests, but no longer corresponds to a rank inside an
    ``mx.distributed.Group``. The drafter dials
    ``drafter_socket_host:drafter_socket_port`` to reach target rank 0;
    target rank 0 binds and listens on that endpoint at instance
    bootstrap.

    Fields:
        drafter_node_id:    Where the drafter runner lives.
        drafter_runner_id:  Identifies the drafter runner; the bootstrap
                            checks ``bound_runner_id == drafter_runner_id``
                            to switch into drafter-only loading mode and
                            enter the drafter serve loop instead of the
                            normal generation engine.
        drafter_model_id:   Which drafter weights to load. Must be one
                            of the entries in the target's
                            ``ModelCard.drafter_model_ids`` list
                            (placement enforces this invariant).
        drafter_rank:       Logical placement index of the drafter
                            inside the conceptual parent group
                            (target_world_size). Retained for
                            placement bookkeeping; not a real
                            ``mx.distributed`` rank in the v3+ wire.
        drafter_socket_host: Host (LAN/Thunderbolt-bridge IP or
                             hostname) target rank 0 advertises for
                             the drafter wire. The drafter dials this
                             host to reach target rank 0.
        drafter_socket_port: TCP port target rank 0 binds on for
                             drafter wire ops. Allocated at placement
                             time; the runner bootstrap binds that
                             specific port (failure is a hard error).
        target_peer_socket_port: TCP port target rank 0 binds on for
                             *inter-target-rank* spec-decode int
                             broadcasts. Distinct from
                             ``drafter_socket_port`` because the drafter
                             dials in over a different IP than the
                             other target ranks; sharing a port would
                             collide. ``None`` for single-target
                             instances (no peer to broadcast to) and
                             for legacy/historical wire payloads
                             produced before the field existed.

                             Codex P1 (PR #21 round-(N+9),
                             instances.py:97): this MUST stay optional
                             with a safe default so older
                             ``DrafterPlacement`` JSON (rolling-upgrade
                             peers, replayed historical events) still
                             round-trips through pubsub
                             ``model_validate_json`` -- making it
                             required broke instance/state replay any
                             time a mixed-version cluster or a stored
                             event stream lacks the field. The fanout
                             helper (`_maybe_setup_target_peer_fanout`)
                             treats ``None`` as "no peer wire", which
                             matches the legacy single-rank-target
                             behavior.
        target_peer_hosts_by_rank: For each non-zero target rank,
                             the IP that rank uses to dial target rank
                             0 over the inter-target socket wire.
                             Resolved at placement time via
                             :func:`find_ip_prioritised`; differs
                             per peer because Thunderbolt /30 meshes
                             expose a unique IP per node pair. Keys
                             are device ranks **stored as strings**
                             so the type round-trips cleanly through
                             JSON (the wire format used by
                             :mod:`event_router`); ``dict[int, str]``
                             would fail strict re-validation because
                             JSON has no int dict keys. Convert to
                             int at the consumer (see
                             :func:`_maybe_setup_target_peer_fanout`).
    """

    drafter_node_id: NodeId
    drafter_runner_id: RunnerId
    drafter_model_id: ModelId
    drafter_rank: int = Field(ge=0)
    drafter_socket_host: str
    drafter_socket_port: int = Field(ge=1, le=65535)
    target_peer_socket_port: int | None = Field(default=None, ge=1, le=65535)
    target_peer_hosts_by_rank: dict[str, str] = Field(default_factory=dict)


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments
    # When set, this instance places the drafter on a separate node from
    # the target ranks and routes drafter/verify IPC over a direct TCP
    # socket (see :class:`DrafterPlacement`). ``None`` (the default)
    # preserves legacy symmetric placement: every rank in
    # ``shard_assignments`` runs a target shard, and any drafter
    # declared on the model card is loaded in-process alongside the
    # target on the single-device cycle.
    drafter_placement: DrafterPlacement | None = None

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)

    @property
    def parent_group_size(self) -> int:
        """Size of the target ranks' ``mx.distributed`` group.

        Always equals ``len(shard_assignments.runner_to_shard)``: in
        the v3+ asymmetric wire the drafter rank does NOT join the
        target ``mx.distributed.Group`` (it talks to target rank 0 via
        a direct TCP socket). Symmetric and asymmetric placement
        therefore both report the same size here, equal to the number
        of target shards.
        """
        return len(self.shard_assignments.runner_to_shard)

    def is_drafter_runner(self, runner_id: RunnerId) -> bool:
        return (
            self.drafter_placement is not None
            and self.drafter_placement.drafter_runner_id == runner_id
        )

    @property
    def all_runner_ids(self) -> list[RunnerId]:
        """Every runner id participating in this instance, target + drafter.

        Lifecycle barriers (ConnectToGroup, LoadModel, StartWarmup,
        Ready) wait on the *whole* parent group, so plan-time readiness
        checks iterate this list. Generation tasks themselves are
        target-only and iterate ``shard_assignments.runner_to_shard``
        directly.
        """
        runners = list(self.shard_assignments.runner_to_shard.keys())
        if self.drafter_placement is not None:
            runners.append(self.drafter_placement.drafter_runner_id)
        return runners

    @property
    def all_node_to_runner(self) -> dict[NodeId, RunnerId]:
        """Per-node runner id including the drafter rank when asymmetric.

        Worker plan iterates this when deciding which node should spawn
        which runner. Symmetric placement returns the legacy
        ``shard_assignments.node_to_runner`` mapping unchanged.
        """
        result = dict(self.shard_assignments.node_to_runner)
        if self.drafter_placement is not None:
            result[self.drafter_placement.drafter_node_id] = (
                self.drafter_placement.drafter_runner_id
            )
        return result


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(FrozenModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def is_drafter_rank(self) -> bool:
        """``True`` when this runner serves the drafter, not a target shard.

        Callers that read ``bound_shard``, ``is_image_model``, or any
        target-shard-derived property MUST branch on this first; those
        properties raise on a drafter-rank bound instance because the
        drafter has no target shard.
        """
        return self.instance.is_drafter_runner(self.bound_runner_id)

    @property
    def parent_rank(self) -> int:
        """This runner's rank inside the parent ``mx.distributed`` group.

        Target ranks read it from their bound shard's ``device_rank``;
        the drafter rank reads it from
        ``DrafterPlacement.drafter_rank``. Plan-time connect/warmup
        ordering checks use this so the same predicate works for both
        symmetric (drafter rank doesn't exist) and asymmetric (drafter
        is rank ``parent_group_size - 1``) placement.
        """
        if self.is_drafter_rank:
            placement = self.instance.drafter_placement
            assert placement is not None  # type narrowed by is_drafter_rank
            return placement.drafter_rank
        return self.bound_shard.device_rank

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None, (
            "bound_shard is only defined for target ranks; "
            "check `is_drafter_rank` before reading it"
        )
        return shard

    @property
    def is_image_model(self) -> bool:
        if self.is_drafter_rank:
            return False
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_runner_known(self) -> "BoundInstance":
        if self.bound_runner_id in self.instance.shard_assignments.runner_to_shard:
            return self
        if self.instance.is_drafter_runner(self.bound_runner_id):
            placement = self.instance.drafter_placement
            assert placement is not None  # type narrowed by is_drafter_runner
            assert self.bound_node_id == placement.drafter_node_id, (
                f"Drafter runner {self.bound_runner_id} bound to node "
                f"{self.bound_node_id}, but DrafterPlacement points to "
                f"{placement.drafter_node_id}"
            )
            return self
        raise AssertionError(
            f"bound_runner_id {self.bound_runner_id} is neither a target rank "
            f"in shard_assignments nor the drafter rank declared by "
            f"instance.drafter_placement"
        )
