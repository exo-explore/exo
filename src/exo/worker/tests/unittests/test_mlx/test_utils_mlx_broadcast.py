"""Unit tests for the MLX utility primitives used by the V2 multi-target spec loop.

These exercise the contracts that the asymmetric pipelined drafter
relies on for cross-rank determinism without spinning up MLX or
``mx.distributed``:

  * :func:`mx_broadcast_int_list` -- length / range / root contract.
    The single-rank short-circuit can be exercised directly; the
    multi-rank ``all_sum`` path is covered indirectly because it
    delegates value validation to the same helper.
  * :func:`_validate_broadcast_values` -- the int32 bounds are tighter
    than Python's ``int`` range, so out-of-range values from a callsite
    bug must raise rather than wrap silently.
  * :func:`_encode_task_id` / :func:`_decode_task_id` -- ASCII codec
    used by ``mx_all_gather_tasks`` to broadcast canonical task IDs.
    Round-trip and bounds are verifiable without MLX.
  * :func:`mx_all_gather_tasks` -- the single-rank short-circuit. The
    multi-rank root-authoritative agreement path needs an actual
    ``mx.distributed`` group, so we cover the structural contract here
    and the cluster bench exercises the real collective.

Kept MLX-free so it runs in milliseconds on CI alongside the rest of
the unittest suite.
"""

from __future__ import annotations

import pytest

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.utils_mlx import (
    _MX_BROADCAST_MAX_VALUE,  # pyright: ignore[reportPrivateUsage]
    _MX_TASK_ID_BYTES,  # pyright: ignore[reportPrivateUsage]
    _decode_task_id,  # pyright: ignore[reportPrivateUsage]
    _detect_distributed_backend,  # pyright: ignore[reportPrivateUsage]
    _encode_task_id,  # pyright: ignore[reportPrivateUsage]
    _validate_broadcast_values,  # pyright: ignore[reportPrivateUsage]
    mx_all_gather_tasks,
    mx_broadcast_int_list,
)

# ---------------------------------------------------------------------------
# Validation helper (unit, no MLX needed)
# ---------------------------------------------------------------------------


class TestValidateBroadcastValues:
    """``_validate_broadcast_values`` rejects values that would corrupt
    the int32 ``all_sum`` buffer: negatives wrap on cast, and values
    >= 2**31 overflow on sum."""

    def test_accepts_zero(self) -> None:
        _validate_broadcast_values([0, 0, 0])

    def test_accepts_typical_token_ids(self) -> None:
        # Gemma-4 vocab is ~256k; well inside int32 positive range.
        _validate_broadcast_values([0, 1, 256_000, 999_999])

    def test_accepts_max_value(self) -> None:
        _validate_broadcast_values([_MX_BROADCAST_MAX_VALUE])

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _validate_broadcast_values([0, -1, 0])

    def test_rejects_overflow(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _validate_broadcast_values([_MX_BROADCAST_MAX_VALUE + 1])

    def test_error_includes_offending_index(self) -> None:
        with pytest.raises(ValueError, match=r"index 2 = -7"):
            _validate_broadcast_values([0, 1, -7, 3])


# ---------------------------------------------------------------------------
# mx_broadcast_int_list (single-rank short-circuit + contract)
# ---------------------------------------------------------------------------


class TestMxBroadcastIntListSingleRank:
    """The ``group is None`` short-circuit covers single-rank deployments
    (the V1 single-target path and the non-distributed test fakes).
    Multi-rank cluster behaviour is exercised by the cluster bench
    because it needs a real ``mx.distributed`` group."""

    def test_returns_values_when_root(self) -> None:
        result = mx_broadcast_int_list([1, 2, 3], length=3, group=None, is_root=True)
        assert result == [1, 2, 3]
        # Returned list must be a new object so mutating it doesn't
        # corrupt the caller's source list.
        result[0] = 99
        # No assertion on the source -- just exercising that the call
        # didn't share storage. ``list(values)`` semantics.

    def test_rejects_zero_length(self) -> None:
        with pytest.raises(ValueError, match="length must be >= 1"):
            mx_broadcast_int_list([], length=0, group=None, is_root=True)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            mx_broadcast_int_list([1, 2], length=3, group=None, is_root=True)

    def test_rejects_none_values_on_root(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            mx_broadcast_int_list(None, length=3, group=None, is_root=True)

    def test_rejects_consumer_in_single_rank(self) -> None:
        # Only the root has source values; ``is_root=False`` with no
        # group means there's no peer to broadcast from -- caller bug.
        with pytest.raises(ValueError, match="single-rank short-circuit"):
            mx_broadcast_int_list([1, 2, 3], length=3, group=None, is_root=False)

    def test_validates_values_on_root(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            mx_broadcast_int_list([0, -1], length=2, group=None, is_root=True)


# ---------------------------------------------------------------------------
# Backend detection (controls which distributed primitive we use)
# ---------------------------------------------------------------------------


class TestDetectDistributedBackend:
    """``_detect_distributed_backend`` resolves ring vs jaccl from the
    env vars set by :func:`mlx_distributed_init`. Backend selection
    matters because MLX's ring backend does not support arbitrary
    point-to-point ``send`` / ``recv`` between non-neighbor ranks --
    multi-rank ring deployments would fail or hang the moment
    :func:`mx_broadcast_int_list` issued a ``send(dst=N)`` for a
    non-neighbor ``N``. Confirm the helper picks the ring-safe path
    whenever the ring marker (``MLX_HOSTFILE``) is present and the
    JACCL path only when the JACCL markers are present in isolation."""

    def test_ring_backend_when_hostfile_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLX_HOSTFILE", "/tmp/hosts.json")
        monkeypatch.delenv("MLX_IBV_DEVICES", raising=False)
        monkeypatch.delenv("MLX_JACCL_COORDINATOR", raising=False)
        assert _detect_distributed_backend() == "ring"

    def test_jaccl_backend_when_ibv_devices_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLX_HOSTFILE", raising=False)
        monkeypatch.setenv("MLX_IBV_DEVICES", "/tmp/devices.json")
        monkeypatch.delenv("MLX_JACCL_COORDINATOR", raising=False)
        assert _detect_distributed_backend() == "jaccl"

    def test_jaccl_backend_when_only_coordinator_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``MLX_JACCL_COORDINATOR`` is set alongside ``MLX_IBV_DEVICES``
        # by :func:`mlx_distributed_init`, but treat either one as a
        # sufficient marker so a partially-populated env doesn't
        # silently route through the slower ring path.
        monkeypatch.delenv("MLX_HOSTFILE", raising=False)
        monkeypatch.delenv("MLX_IBV_DEVICES", raising=False)
        monkeypatch.setenv("MLX_JACCL_COORDINATOR", "tcp://10.0.0.1:1234")
        assert _detect_distributed_backend() == "jaccl"

    def test_ring_wins_when_both_markers_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pathological env where both markers are set (e.g. an old
        # JACCL run leaked vars into a fresh ring session). Defaulting
        # to ring is the conservative choice: the all-sum primitive
        # works on JACCL too (it just gives up the wire-conflation
        # protection that ``send`` / ``recv`` provided), whereas
        # routing through ``send`` / ``recv`` on ring is a hard
        # failure.
        monkeypatch.setenv("MLX_HOSTFILE", "/tmp/hosts.json")
        monkeypatch.setenv("MLX_IBV_DEVICES", "/tmp/devices.json")
        assert _detect_distributed_backend() == "ring"

    def test_defaults_to_ring_when_no_markers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Test fakes that build a fake ``Group`` without going through
        # :func:`mlx_distributed_init` shouldn't crash on a missing
        # backend marker -- pick the ring-safe path so the helper
        # stays callable from unit tests.
        monkeypatch.delenv("MLX_HOSTFILE", raising=False)
        monkeypatch.delenv("MLX_IBV_DEVICES", raising=False)
        monkeypatch.delenv("MLX_JACCL_COORDINATOR", raising=False)
        assert _detect_distributed_backend() == "ring"


# ---------------------------------------------------------------------------
# Task-list hashing (drift detection)
# ---------------------------------------------------------------------------


def _make_task(task_id: str) -> TextGeneration:
    """Build a minimal :class:`TextGeneration` for hash-based drift tests.

    The hash function only inspects ``task_id``; the rest of the fields
    are filled with the smallest valid values that satisfy Pydantic's
    strict-mode validation. Keep the construction here so the cluster-
    facing types' field churn doesn't ripple through every assertion
    body.
    """
    return TextGeneration(
        task_id=TaskId(task_id),
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[
                InputMessage(role="user", content=InputMessageContent("hello")),
            ],
            max_output_tokens=1,
        ),
    )


class TestTaskIdCodec:
    """``_encode_task_id`` / ``_decode_task_id`` are the wire codec for
    the root-authoritative agreement protocol. Round-trip must be
    exact and bounds must be enforced; otherwise a corrupt payload
    silently misagrees on which task to admit."""

    def test_round_trip_uuid4(self) -> None:
        ident = "01234567-89ab-cdef-0123-456789abcdef"
        encoded = _encode_task_id(ident)
        assert len(encoded) == _MX_TASK_ID_BYTES
        assert _decode_task_id(encoded) == ident

    def test_short_id_is_zero_padded(self) -> None:
        encoded = _encode_task_id("alpha")
        # Trailing slots stay zero so the decoder's null terminator
        # logic stops at the right place.
        assert encoded[5:] == [0] * (_MX_TASK_ID_BYTES - 5)
        assert _decode_task_id(encoded) == "alpha"

    def test_rejects_oversize_id(self) -> None:
        too_long = "a" * (_MX_TASK_ID_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds"):
            _encode_task_id(too_long)

    def test_rejects_non_ascii_byte_on_decode(self) -> None:
        bogus = [200] + [0] * (_MX_TASK_ID_BYTES - 1)
        with pytest.raises(ValueError, match="outside ASCII range"):
            _decode_task_id(bogus)

    def test_decoder_stops_at_null(self) -> None:
        # Two real chars, then a null, then garbage: decoder must
        # stop at the null and ignore the trailing data.
        slots = [ord("a"), ord("b"), 0, ord("z")] + [0] * (_MX_TASK_ID_BYTES - 4)
        assert _decode_task_id(slots) == "ab"


# ---------------------------------------------------------------------------
# mx_all_gather_tasks single-rank short-circuit
# ---------------------------------------------------------------------------


class TestMxAllGatherTasksSingleRank:
    """Single-rank short-circuit: returns the local task list as-is and
    never invokes a collective. The multi-rank root-authoritative path
    needs an actual ``mx.distributed`` group and is exercised by the
    cluster bench."""

    def test_empty_input(self) -> None:
        agreed, different = mx_all_gather_tasks([], group=None)
        assert agreed == []
        assert different == []

    def test_passes_through_tasks(self) -> None:
        tasks = [_make_task("task-1"), _make_task("task-2")]
        agreed, different = mx_all_gather_tasks(tasks, group=None)
        assert agreed == tasks
        assert different == []

    def test_returns_a_copy(self) -> None:
        # The caller mutates ``self._maybe_queue`` after the gather;
        # the returned list must be a different object so post-gather
        # mutation doesn't corrupt the agreement view.
        tasks = [_make_task("task-1")]
        agreed, _different = mx_all_gather_tasks(tasks, group=None)
        assert agreed is not tasks


# ---------------------------------------------------------------------------
# Two-phase intersection agreement: end-to-end via in-process simulation
# ---------------------------------------------------------------------------


def _agree_intersection(
    rank_views: list[list[TextGeneration]],
) -> list[tuple[list[TextGeneration], list[TextGeneration]]]:
    """Run the two-phase intersection protocol entirely in-process.

    Mirrors :func:`mx_all_gather_tasks` for ``len(rank_views)`` ranks
    without spinning up MLX. Phase 1 is root's broadcast (the first
    entry in ``rank_views`` is treated as root); phase 2 is the
    cross-rank vote (sum of indicator vectors). Returns each rank's
    ``(agreed, leftover)`` pair so tests can assert that all ranks
    land on the SAME ``agreed`` set, which is the whole point of the
    protocol -- without it, divergent admit decisions leave one rank
    in the spec loop while the other re-enters ``agree_on_tasks``,
    causing collective-stream cross-talk and downstream
    ``IndexError`` in the detokenizer when broadcast token slots
    arrive scrambled.
    """
    from exo.worker.engines.mlx.utils_mlx import (
        _MX_AGREE_BUFFER_LEN,  # pyright: ignore[reportPrivateUsage]
        _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        _MX_TASK_ID_BYTES,  # pyright: ignore[reportPrivateUsage]
        _decode_task_id,  # pyright: ignore[reportPrivateUsage]
        _encode_task_id,  # pyright: ignore[reportPrivateUsage]
    )

    if not rank_views:
        return []
    group_size = len(rank_views)
    root_tasks = rank_views[0]

    admitted = root_tasks[:_MX_AGREE_MAX_TASKS]
    payload: list[int] = [len(admitted)]
    for task in admitted:
        payload.extend(_encode_task_id(task.task_id))
    payload.extend([0] * (_MX_AGREE_BUFFER_LEN - len(payload)))

    count = payload[0]
    canonical_ids: list[str] = []
    for i in range(count):
        start = 1 + i * _MX_TASK_ID_BYTES
        end = start + _MX_TASK_ID_BYTES
        canonical_ids.append(_decode_task_id(payload[start:end]))

    rank_locals: list[dict[TaskId, TextGeneration]] = [
        {t.task_id: t for t in tasks} for tasks in rank_views
    ]
    votes_per_rank = [
        [1 if cid in local else 0 for cid in canonical_ids] for local in rank_locals
    ]
    summed = [sum(votes[i] for votes in votes_per_rank) for i in range(count)]

    canonical_id_set = {TaskId(cid) for cid in canonical_ids}
    results: list[tuple[list[TextGeneration], list[TextGeneration]]] = []
    for rank_idx, local in enumerate(rank_locals):
        agreed: list[TextGeneration] = []
        local_remaining = dict(local)
        for i, cid in enumerate(canonical_ids):
            if summed[i] != group_size:
                continue
            task = local_remaining.pop(TaskId(cid), None)
            if task is not None:
                agreed.append(task)
        # Mirror ``mx_all_gather_tasks``'s starvation-avoiding leftover
        # ordering (Codex P1, PR #21 round 3): tasks that never made it
        # into the canonical broadcast (never given a chance) go to
        # the front, canonical-but-not-agreed tasks go to the back.
        front_of_leftover: list[TextGeneration] = []
        back_of_leftover: list[TextGeneration] = []
        for task in rank_views[rank_idx]:
            if task.task_id not in local_remaining:
                continue
            if task.task_id in canonical_id_set:
                back_of_leftover.append(task)
            else:
                front_of_leftover.append(task)
        leftover = front_of_leftover + back_of_leftover
        results.append((agreed, leftover))
    return results


class TestIntersectionAgreement:
    """Cross-rank intersection semantics. The protocol's correctness
    contract is that every rank that returns from
    :func:`mx_all_gather_tasks` lands on the SAME ``agreed`` set, so
    the next ``_admit_queued_tasks`` admits identical tasks on every
    rank -- preventing the divergence that historically led to
    cross-talk between admit collectives and spec-loop collectives."""

    def test_unanimous_admission(self) -> None:
        a_root = _make_task("alpha")
        a_peer = _make_task("alpha")
        results = _agree_intersection([[a_root], [a_peer]])
        assert len(results) == 2
        for agreed, leftover in results:
            assert [t.task_id for t in agreed] == ["alpha"]
            assert leftover == []

    def test_root_only_task_deferred_on_both_ranks(self) -> None:
        # Root has task that peer hasn't received yet: NEITHER rank
        # admits it. This is the whole reason for intersection
        # rather than root-authoritative.
        results = _agree_intersection([[_make_task("alpha")], []])
        for agreed, _ in results:
            assert agreed == []
        assert [t.task_id for t in results[0][1]] == ["alpha"]
        assert results[1][1] == []

    def test_peer_only_task_deferred_on_both_ranks(self) -> None:
        results = _agree_intersection([[], [_make_task("future")]])
        for agreed, _ in results:
            assert agreed == []
        assert results[0][1] == []
        assert [t.task_id for t in results[1][1]] == ["future"]

    def test_partial_overlap_only_intersection_admitted(self) -> None:
        a_root = _make_task("alpha")
        a_peer = _make_task("alpha")
        beta = _make_task("beta")
        gamma = _make_task("gamma")
        results = _agree_intersection([[a_root, beta], [a_peer, gamma]])
        for agreed, _ in results:
            assert [t.task_id for t in agreed] == ["alpha"]
        assert [t.task_id for t in results[0][1]] == ["beta"]
        assert [t.task_id for t in results[1][1]] == ["gamma"]

    def test_three_rank_intersection(self) -> None:
        # 3-rank target: agreed is what *every* rank has. Anything
        # short of unanimous stays out.
        results = _agree_intersection(
            [
                [_make_task("alpha"), _make_task("beta")],
                [_make_task("alpha"), _make_task("beta")],
                [_make_task("alpha")],
            ]
        )
        for agreed, _ in results:
            assert [t.task_id for t in agreed] == ["alpha"]

    def test_canonical_order_is_root_order(self) -> None:
        ids_root = ["c", "a", "b"]
        ids_peer = ["b", "a", "c"]
        results = _agree_intersection(
            [
                [_make_task(i) for i in ids_root],
                [_make_task(i) for i in ids_peer],
            ]
        )
        for agreed, _ in results:
            assert [t.task_id for t in agreed] == ids_root

    def test_root_caps_at_max_tasks(self) -> None:
        from exo.worker.engines.mlx.utils_mlx import (
            _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        )

        many = [_make_task(f"t{i:02d}") for i in range(_MX_AGREE_MAX_TASKS + 4)]
        peer_copy = [_make_task(t.task_id) for t in many]
        results = _agree_intersection([many, peer_copy])
        for agreed, _ in results:
            assert len(agreed) == _MX_AGREE_MAX_TASKS


class TestNoStarvationWhenFirstPageStuck:
    """Codex P1 (PR #21 round 3): a queue larger than
    ``_MX_AGREE_MAX_TASKS`` whose entire first page is missing on a
    peer would have starved tasks at positions ``>= _MX_AGREE_MAX_TASKS``
    indefinitely, because every round root re-broadcast the same
    ``tasks[:_MX_AGREE_MAX_TASKS]`` and tasks past the cap never
    entered the canonical set.

    The fix demotes canonical-but-not-agreed IDs to the back of
    leftover, so the next agreement round's first page is biased
    toward fresh candidates that haven't been broadcast yet.
    """

    def test_first_page_all_missing_advances_via_demotion(self) -> None:
        """Round 1 stalls (all canonical IDs missing on peer); round 2
        must broadcast tasks that were beyond the cap, not the same
        stuck first page.
        """
        from exo.worker.engines.mlx.utils_mlx import (
            _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        )

        max_tasks: int = _MX_AGREE_MAX_TASKS
        # Build a deeply-backlogged root queue. The first ``max_tasks``
        # IDs are stuck (peer hasn't received them yet); the remaining
        # ``max_tasks + 4`` IDs are present on every rank.
        stuck_ids = [f"stuck{i:02d}" for i in range(max_tasks)]
        deliverable_ids = [f"ok{i:02d}" for i in range(max_tasks + 4)]

        root_view = [_make_task(tid) for tid in stuck_ids + deliverable_ids]
        peer_view = [_make_task(tid) for tid in deliverable_ids]

        # ----- Round 1: every canonical ID misses on the peer -----
        round_1 = _agree_intersection([root_view, peer_view])
        root_admitted_r1, root_leftover_r1 = round_1[0]
        peer_admitted_r1, peer_leftover_r1 = round_1[1]
        assert root_admitted_r1 == []
        assert peer_admitted_r1 == []

        # Pre-fix, root's leftover would have been the original order
        # (stuck first), so round 2 would have had the same first page
        # and made no progress. The fix demotes stuck canonical IDs
        # to the back so deliverable IDs are now at the front.
        root_leftover_ids_r1 = [t.task_id for t in root_leftover_r1]
        # First ``max_tasks + 4`` slots should be deliverable tasks
        # (the ones that were beyond the cap in round 1).
        assert root_leftover_ids_r1[: len(deliverable_ids)] == deliverable_ids, (
            "deliverable tasks must rotate to the front of root's "
            "leftover so the next agreement round can broadcast them; "
            f"got {root_leftover_ids_r1!r}"
        )
        # Stuck IDs are demoted to the tail.
        assert root_leftover_ids_r1[len(deliverable_ids) :] == stuck_ids, (
            "stuck canonical tasks must be demoted to the back "
            f"got {root_leftover_ids_r1!r}"
        )

        # ----- Round 2: deliverable tasks finally get broadcast -----
        round_2 = _agree_intersection([root_leftover_r1, peer_leftover_r1])
        root_admitted_r2, root_leftover_r2 = round_2[0]
        peer_admitted_r2, _peer_leftover_r2 = round_2[1]

        # The first ``max_tasks`` deliverable IDs land in the canonical
        # broadcast and are admitted by both ranks.
        admitted_ids = {t.task_id for t in root_admitted_r2}
        assert admitted_ids == set(deliverable_ids[:max_tasks]), (
            f"first {max_tasks} deliverable IDs must be admitted in "
            f"round 2 once the demotion lifts the starvation; got "
            f"admitted={sorted(admitted_ids)}"
        )
        # Both ranks land on the same agreed set (the protocol's core
        # contract: divergence breaks collective-stream sync).
        assert {t.task_id for t in peer_admitted_r2} == admitted_ids

        # Root's leftover after round 2 should still contain the stuck
        # IDs and the four deliverable IDs that didn't fit in round 2's
        # canonical broadcast.
        root_leftover_ids_r2 = {t.task_id for t in root_leftover_r2}
        assert root_leftover_ids_r2 == set(stuck_ids) | set(
            deliverable_ids[max_tasks:]
        ), (
            "round-2 leftover must keep the still-stuck IDs and the "
            f"deliverable IDs beyond the cap; got {root_leftover_ids_r2!r}"
        )

    def test_partial_first_page_miss_does_not_block_beyond_cap_tasks(self) -> None:
        """A subtler variant: root queue is exactly larger than the
        cap, only one of the first-page IDs is missing. The pre-fix
        leftover would still have rotated the missing ID into round
        2's first page, but the deliverable tasks beyond position
        ``max_tasks`` would still see the missing ID at the head of
        root's broadcast each round, slowing throughput. The fix
        keeps the head of next round's queue clear of stuck IDs.
        """
        from exo.worker.engines.mlx.utils_mlx import (
            _MX_AGREE_MAX_TASKS,  # pyright: ignore[reportPrivateUsage]
        )

        max_tasks: int = _MX_AGREE_MAX_TASKS
        # Single stuck ID at root's head, plus ``max_tasks`` deliverable IDs.
        stuck_id = "stuck00"
        deliverable_ids = [f"ok{i:02d}" for i in range(max_tasks)]

        root_view = [_make_task(tid) for tid in [stuck_id] + deliverable_ids]
        peer_view = [_make_task(tid) for tid in deliverable_ids]

        # Round 1: canonical = stuck + first 15 deliverable IDs.
        # Peer admits 15 deliverable, stuck is deferred.
        round_1 = _agree_intersection([root_view, peer_view])
        root_admitted_r1, root_leftover_r1 = round_1[0]
        admitted_ids_r1 = {t.task_id for t in root_admitted_r1}
        assert admitted_ids_r1 == set(deliverable_ids[: max_tasks - 1])

        # Stuck ID is at the BACK of root's leftover, the last
        # deliverable ID is at the FRONT (it was beyond the cap).
        root_leftover_ids_r1 = [t.task_id for t in root_leftover_r1]
        assert root_leftover_ids_r1 == [
            deliverable_ids[max_tasks - 1],
            stuck_id,
        ], (
            "demotion must keep beyond-cap deliverable tasks at the "
            "front and push the stuck canonical task to the back; "
            f"got {root_leftover_ids_r1!r}"
        )
