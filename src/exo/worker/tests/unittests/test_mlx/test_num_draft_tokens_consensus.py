"""Regression test for ``_broadcast_clamped_num_draft_tokens``.

Pins the contract that EVERY target rank in a multi-target
asymmetric placement uses the same ``num_draft_tokens`` (== K) when
constructing its ``PipelinedModelDrafter``. Pre-fix only rank 0 ran
the transport clamp, so a per-request override above the wire-
protocol budget desynchronized the ``_broadcast_drafts`` /
``_broadcast_target_tokens`` collectives whenever rank 1 used the
unclamped value (Codex P1 on PR #20 round 3).

These tests stay MLX-free by patching ``mx_broadcast_int_list`` with
a deterministic stand-in that emulates a rank-0 broadcast: the
captured value from rank 0's call is what every later non-root
caller receives.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Final

import pytest

from exo.worker.engines.mlx.generator import generate as gen


class _FakeMxGroup:
    """Minimal stand-in for ``mx.distributed.Group`` covering only
    the methods ``_broadcast_clamped_num_draft_tokens`` actually
    calls (``rank``, ``size``).
    """

    def __init__(self, *, rank: int, size: int) -> None:
        self._rank: Final[int] = rank
        self._size: Final[int] = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture
def shared_broadcast_state() -> dict[str, list[int]]:
    """Mailbox the fake broadcast uses to persist rank-0's value
    across the (sequential) calls from rank 0 and rank 1 in the same
    test.
    """
    return {"value": []}


def _make_fake_broadcaster(
    state: dict[str, list[int]],
) -> object:
    def fake_broadcast(
        values: list[int] | None,
        length: int,
        group: object,
        *,
        is_root: bool,
    ) -> list[int]:
        assert length == 1, (
            "_broadcast_clamped_num_draft_tokens is contracted to a "
            f"single-int broadcast; got length={length}"
        )
        if is_root:
            assert values is not None and len(values) == 1
            state["value"] = list(values)
            return list(values)
        # Non-root: return the previously-captured rank-0 value.
        # In a real ``mx.distributed`` broadcast the non-root rank
        # never sees rank 0's input -- it pulls from the wire. In
        # this test fixture the rank-0 caller runs first to seed
        # ``state["value"]``; the assertion catches missed orderings.
        assert state["value"], (
            "fake broadcaster: non-root call but no rank-0 value has "
            "been recorded; tests must call the rank-0 path first"
        )
        return list(state["value"])

    return fake_broadcast


def test_root_rank_broadcasts_clamped_value(
    monkeypatch: pytest.MonkeyPatch,
    shared_broadcast_state: dict[str, list[int]],
) -> None:
    """Rank 0 calls the helper after clamping. The helper returns
    rank 0's input verbatim AND records it in the broadcast mailbox
    so non-root ranks pick it up.
    """
    monkeypatch.setattr(
        gen,
        "mx_broadcast_int_list",
        _make_fake_broadcaster(shared_broadcast_state),
    )

    group = _FakeMxGroup(rank=0, size=2)
    consensus = gen._broadcast_clamped_num_draft_tokens(
        effective_num_draft_tokens=4,
        group=group,  # pyright: ignore[reportArgumentType]
    )

    assert consensus == 4
    assert shared_broadcast_state["value"] == [4]


def test_non_root_rank_adopts_root_clamped_value(
    monkeypatch: pytest.MonkeyPatch,
    shared_broadcast_state: dict[str, list[int]],
) -> None:
    """The bug Codex flagged: rank 0 clamps from 8 to 4, rank 1
    keeps 8 unless we broadcast. After the fix, rank 1's local
    ``effective_num_draft_tokens`` is overwritten with rank 0's 4.
    """
    monkeypatch.setattr(
        gen,
        "mx_broadcast_int_list",
        _make_fake_broadcaster(shared_broadcast_state),
    )

    # Step 1 -- rank 0 broadcasts the clamped value.
    rank_zero_consensus = gen._broadcast_clamped_num_draft_tokens(
        effective_num_draft_tokens=4,
        group=_FakeMxGroup(rank=0, size=2),  # pyright: ignore[reportArgumentType]
    )
    assert rank_zero_consensus == 4

    # Step 2 -- rank 1 enters with its UNCLAMPED value (8). Pre-fix
    # rank 1 would have constructed PipelinedModelDrafter with K=8
    # and sized ``_broadcast_drafts`` slots accordingly; rank 0 sized
    # them to K=4. The fix's broadcast forces rank 1 to adopt 4.
    rank_one_consensus = gen._broadcast_clamped_num_draft_tokens(
        effective_num_draft_tokens=8,  # local-only stale value
        group=_FakeMxGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
    )

    assert rank_one_consensus == 4, (
        "non-root target rank must adopt rank 0's clamped "
        "num_draft_tokens; pre-fix it used its own unclamped value "
        "and desynchronized _broadcast_drafts collectives"
    )


def test_consensus_no_op_when_request_within_budget(
    monkeypatch: pytest.MonkeyPatch,
    shared_broadcast_state: dict[str, list[int]],
) -> None:
    """When the per-request K is already at or below the transport
    budget, rank 0 doesn't clamp. Both ranks enter with the same K
    and the broadcast is effectively a no-op (same value flows
    through). This test confirms the fix doesn't change behavior on
    the common path.
    """
    monkeypatch.setattr(
        gen,
        "mx_broadcast_int_list",
        _make_fake_broadcaster(shared_broadcast_state),
    )

    rank_zero = gen._broadcast_clamped_num_draft_tokens(
        effective_num_draft_tokens=3,
        group=_FakeMxGroup(rank=0, size=2),  # pyright: ignore[reportArgumentType]
    )
    rank_one = gen._broadcast_clamped_num_draft_tokens(
        effective_num_draft_tokens=3,
        group=_FakeMxGroup(rank=1, size=2),  # pyright: ignore[reportArgumentType]
    )

    assert rank_zero == rank_one == 3
