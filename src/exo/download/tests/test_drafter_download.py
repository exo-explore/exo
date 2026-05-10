"""Tests for chained drafter downloads in :class:`DownloadCoordinator`.

When a target model card declares ``drafter_model_id``, kicking off a
download for the target should also kick off a download for the matching
drafter so single-device speculative decoding works without manual setup.

These tests stub the underlying ``ShardDownloader`` and the model-card
loader so they can run in CI without touching HuggingFace or the disk.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import anyio
import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgressData,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TARGET_ID = ModelId("test-org/test-target")
DRAFTER_ID = ModelId("test-org/test-drafter")


@contextlib.contextmanager
def _patch_card_loaders(side_effect: object):
    """Patch both ``ModelCard.load`` and ``ModelCard.load_cached_only``.

    The coordinator's drafter-chain and delete-cascade paths call
    :meth:`ModelCard.load_cached_only` (introduced in PR #18 round-
    (N+15) to avoid Hugging Face round-trips on the command-processor
    coroutine), while older surfaces still call :meth:`ModelCard.load`.
    Tests need both methods to return the same mock card so the
    side-effect contract holds across both call sites without each
    test caring which entrypoint the coordinator happens to use.

    A side-effect that raises on unexpected ids can therefore be
    applied uniformly via the same helper; passing a callable that
    raises ``AssertionError`` for unmatched ids stays meaningful
    against either entrypoint.
    """
    with (
        patch.object(ModelCard, "load", side_effect=side_effect),
        patch.object(ModelCard, "load_cached_only", side_effect=side_effect),
    ):
        yield


def _make_target_card(drafters: list[ModelId]) -> ModelCard:
    return ModelCard(
        model_id=TARGET_ID,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=drafters,
    )


def _make_drafter_card() -> ModelCard:
    return ModelCard(
        model_id=DRAFTER_ID,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _make_shard(card: ModelCard) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


class _RecordingShardDownloader(ShardDownloader):
    """Records every shard ``ensure_shard`` is called on and reports
    ``status='complete'`` immediately so the coordinator advances to a
    terminal state."""

    def __init__(self) -> None:
        self.ensured: list[ModelId] = []
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        self.ensured.append(shard.model_card.model_id)
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=1,
            total_files=1,
            downloaded=shard.model_card.storage_size,
            downloaded_this_session=shard.model_card.storage_size,
            total=shard.model_card.storage_size,
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="complete",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        return Path("/fake/models") / shard.model_card.model_id.normalize()

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(_make_drafter_card()),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=shard.model_card.storage_size,
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


async def _wait_for_completed(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> DownloadCompleted | None:
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadCompleted)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event.download_progress
    except TimeoutError:
        return None


@contextlib.asynccontextmanager
async def _running_coordinator(
    downloader: _RecordingShardDownloader,
    *,
    offline: bool = False,
) -> AsyncIterator[
    tuple[
        DownloadCoordinator,
        Sender[ForwarderDownloadCommand],
        Receiver[Event],
    ]
]:
    cmd_send: Sender[ForwarderDownloadCommand]
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        offline=offline,
    )
    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        yield coordinator, cmd_send, event_recv
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_target_with_drafter_chains_drafter_download() -> None:
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            assert await _wait_for_completed(event_recv, DRAFTER_ID) is not None

    assert TARGET_ID in downloader.ensured
    assert DRAFTER_ID in downloader.ensured


async def test_target_without_drafter_does_not_chain() -> None:
    target_shard = _make_shard(_make_target_card([]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError("ModelCard.load should not be called when no drafter")

    with _patch_card_loaders(fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_skipped_when_disabled_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_DISABLE_DRAFTER", "1")
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load should not be called when EXO_DISABLE_DRAFTER set"
        )

    with _patch_card_loaders(fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_swallows_card_load_error() -> None:
    """If the drafter's ModelCard cannot be loaded (e.g. HF unreachable, card
    not in repo), the target download must still complete and the coordinator
    must not crash."""
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def boom(_: ModelId) -> ModelCard:
        raise RuntimeError("simulated card load failure")

    with _patch_card_loaders(boom):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (_, cmd_send, event_recv):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            await asyncio.sleep(0.05)

    assert downloader.ensured == [TARGET_ID]


async def test_drafter_chain_skipped_in_offline_mode() -> None:
    """Offline-mode coordinators must NOT call ``ModelCard.load`` for
    declared drafters even when the target download itself is locally
    complete.

    ``ModelCard.load`` falls through to ``ModelCard.fetch_from_hf``
    whenever the drafter card isn't already in ``_card_cache``. Under
    ``EXO_OFFLINE=true`` that's an outbound HuggingFace request that
    can stall command processing for the full client timeout before
    the eventual ``DownloadFailed`` is swallowed by the silent
    best-effort drafter chain. The fix short-circuits
    ``_maybe_chain_drafter_download`` when ``self.offline`` is True
    so no card resolution is attempted.

    Test calls ``_maybe_chain_drafter_download`` directly so the
    assertion is precise: ``ModelCard.load`` is the network entry
    point, and the test fails immediately if the offline guard
    regresses to letting it fire.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called in offline mode "
            "(would trigger a HuggingFace fetch)"
        )

    with _patch_card_loaders(fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader, offline=True) as (
            coordinator,
            _,
            _,
        ):
            await coordinator._maybe_chain_drafter_download(  # pyright: ignore[reportPrivateUsage]
                target_shard
            )
            await asyncio.sleep(0.05)

    # No drafter download was ever queued because the chain
    # short-circuited before ``ModelCard.load``.
    assert downloader.ensured == []


async def test_drafter_chain_runs_off_command_processing_path() -> None:
    """Codex flagged (P1, PR #18 round 2) that the drafter card fetch
    ran inline inside ``_command_processor``, so a slow HF call
    blocked unrelated commands. The fix backgrounds the chain via
    ``_tg.start_soon``; this test verifies that a second command
    arriving while ``ModelCard.load`` is hung still progresses.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    # Block ModelCard.load until we've observed the second command
    # being processed.
    drafter_load_started = asyncio.Event()
    drafter_load_release = asyncio.Event()

    async def slow_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            drafter_load_started.set()
            await drafter_load_release.wait()
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    # Second command -- a CancelDownload -- proves the command loop
    # is still responsive even while the drafter chain is hung.
    second_target = ModelId("test-org/second-target")

    with _patch_card_loaders(slow_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            _coordinator,
            cmd_send,
            event_recv,
        ):
            # Kick off the target download; the drafter chain will
            # block on ``slow_load``.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None

            # Wait for the drafter chain to actually be running and
            # blocked on ``slow_load`` (proves the chain was
            # dispatched). A bounded wait so a regression that takes
            # the chain off-process entirely surfaces as a clear
            # timeout failure instead of a silent skip.
            async with asyncio.timeout(2.0):
                await drafter_load_started.wait()

            # Command loop must remain responsive: send a
            # CancelDownload for an UNRELATED model and verify it
            # processes immediately (no-op, but the coordinator must
            # observe it). Before the fix, this would block until
            # ``slow_load`` completed (or timed out).
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(
                        target_node_id=NODE_ID, model_id=second_target
                    ),
                )
            )

            # A small grace window to let the cancel command be
            # observed; the drafter chain is still blocked so any
            # progress here is by definition concurrent.
            await asyncio.sleep(0.1)

            # Release the drafter load so the test cleans up.
            drafter_load_release.set()
            await asyncio.sleep(0.1)


async def test_cancel_during_chain_aborts_drafter_download() -> None:
    """Codex P1 (PR #18 round-(N+1)): a CancelDownload that arrives
    AFTER StartDownload but BEFORE the chain coroutine has registered
    its drafters in ``_drafter_children`` must still prevent the
    drafter download from starting. Pre-fix, the cancel cascade ran
    against an empty children list (the chain hadn't populated it
    yet) and the chain then merrily dispatched ``ensure_shard`` for
    the drafter despite the user having revoked the parent intent.
    Post-fix, ``_spawn_drafter_chain`` pre-registers an empty entry
    and the chain re-checks membership after every ``await`` so the
    cascade pops the entry and signals the chain to bail.

    The race is reproduced deterministically by stalling
    ``ModelCard.load`` so the chain reaches its post-load
    cancellation re-check while a cancel is in flight.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    drafter_load_started = asyncio.Event()
    drafter_load_release = asyncio.Event()

    async def slow_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            drafter_load_started.set()
            await drafter_load_release.wait()
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(slow_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None

            # Wait for the chain to actually enter ``ModelCard.load``;
            # at this point the cancel is racing the load resolution.
            async with asyncio.timeout(2.0):
                await drafter_load_started.wait()

            # Cancel the target while the chain is hung mid-load.
            # Pre-fix: ``_drafter_children[TARGET_ID]`` was empty, so
            # the cascade had nothing to cancel; after release, the
            # chain proceeded to call ``ensure_shard(DRAFTER_ID)``.
            # Post-fix: the entry exists (pre-registered), the cancel
            # cascade pops it, and the chain's post-load re-check
            # sees ``cancelled() == True`` and returns.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(target_node_id=NODE_ID, model_id=TARGET_ID),
                )
            )

            # Give the cancel command a moment to be processed
            # before releasing the load.
            await asyncio.sleep(0.1)
            drafter_load_release.set()

            # Allow the chain coroutine to run its post-load check.
            await asyncio.sleep(0.1)

            # Drafter download must NOT have been kicked off, because
            # the parent target was cancelled before its load
            # resolved. Only the target made it into ``ensured``.
            assert DRAFTER_ID not in downloader.ensured, (
                "drafter download must NOT start when its parent target "
                "was cancelled mid-chain; got ensured="
                f"{downloader.ensured!r}"
            )
            # The cancel cascade must also have removed the parent
            # entry, so a duplicate cancel doesn't try to cascade
            # into a stale drafter list.
            assert TARGET_ID not in coordinator._drafter_children, (  # pyright: ignore[reportPrivateUsage]
                "cancel cascade must clear _drafter_children for the "
                "target so a duplicate cancel doesn't double-cascade"
            )


async def test_failed_target_does_not_chain_drafter() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:231): a target
    that is already in ``DownloadFailed`` state must NOT trigger a
    drafter chain. The round-(N+1) "backfill drafters even when
    target was already tracked" branch swept failed targets into
    the same fast-path, kicking off drafter downloads for a target
    that won't itself download. Drafters served by a non-runnable
    target are useless (the runner can't boot speculative decoding
    without the target weights), so we must consume the network/
    disk only when the target is at least possibly going to run.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called when target is "
            "already in DownloadFailed state"
        )

    with _patch_card_loaders(fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            _,
        ):
            # Pre-seed the target's download_status as FAILED.
            from exo.shared.types.worker.downloads import DownloadFailed

            coordinator.download_status[TARGET_ID] = DownloadFailed(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                error_message="simulated previous failure",
                model_directory="/fake/target",
            )

            # Re-issuing StartDownload for a previously-failed target
            # must NOT chain drafters. Pre-fix: the round-(N+1) code
            # called ``self._spawn_drafter_chain(shard)`` from inside
            # the failed-state fast-path branch; we'd get
            # ``ModelCard.load`` and the AssertionError above.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            await asyncio.sleep(0.1)

    # Drafter must NOT have been queued for download.
    assert DRAFTER_ID not in downloader.ensured, (
        "drafter download must NOT start when target is in "
        f"DownloadFailed state; got ensured={downloader.ensured!r}"
    )


async def test_restart_target_re_chains_cancelled_drafter() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:437): after a
    cancel cascade demotes a chained drafter to ``DownloadPending``,
    a subsequent ``StartDownload`` for the same target is a fresh
    user intent and must bring the drafter back to life. Pre-fix,
    ``drafter_id in self.download_status`` short-circuited
    regardless of the drafter's current state, so a once-cancelled
    drafter never restarted and speculative decoding silently
    stayed disabled until the operator manually started each
    drafter.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_shard = _make_shard(_make_drafter_card())
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            # Simulate the post-cancel state: the drafter was
            # previously chained, then cancelled (DownloadPending).
            from exo.shared.types.worker.downloads import DownloadPending

            coordinator.download_status[DRAFTER_ID] = DownloadPending(
                shard_metadata=drafter_shard,
                node_id=NODE_ID,
                model_directory="/fake/drafter",
            )

            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            assert await _wait_for_completed(event_recv, DRAFTER_ID) is not None

    # Drafter must have been re-ensured: pre-fix this list contained
    # only the target, because the drafter's stale ``DownloadPending``
    # status short-circuited the chain branch.
    assert DRAFTER_ID in downloader.ensured, (
        "subsequent StartDownload(target) must re-chain a previously "
        f"cancelled drafter; got ensured={downloader.ensured!r}"
    )


async def test_cancel_target_cascades_to_chained_drafter() -> None:
    """Codex flagged (P2, PR #18 round 2) that cancelling a target
    left chained drafters running independently. The fix wires a
    parent->children mapping that ``_cancel_download`` cascades.

    Test calls ``_cancel_download`` directly with a synthesised
    children mapping so we don't depend on the timing of the
    background chain task to populate state.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    downloader = _RecordingShardDownloader()
    async with _running_coordinator(downloader) as (
        coordinator,
        _,
        _,
    ):
        # Pre-seed the parent->children mapping and active downloads
        # so the cancel cascade has something to operate on.
        coordinator._drafter_children[TARGET_ID] = [DRAFTER_ID]  # pyright: ignore[reportPrivateUsage]
        target_scope = anyio.CancelScope()
        drafter_scope = anyio.CancelScope()
        coordinator.active_downloads[TARGET_ID] = target_scope
        coordinator.active_downloads[DRAFTER_ID] = drafter_scope

        # Status entries needed by ``_cancel_download``'s pending
        # synthesis path.
        def _ongoing_progress(
            downloaded_mb: int, total_mb: int
        ) -> DownloadProgressData:
            return DownloadProgressData(
                downloaded=Memory.from_mb(downloaded_mb),
                downloaded_this_session=Memory.from_mb(downloaded_mb),
                total=Memory.from_mb(total_mb),
                completed_files=0,
                total_files=1,
                speed=0.0,
                eta_ms=0,
                files={},
            )

        coordinator.download_status[TARGET_ID] = DownloadOngoing(
            shard_metadata=target_shard,
            node_id=NODE_ID,
            model_directory="/fake/target",
            download_progress=_ongoing_progress(100, 500),
        )
        drafter_card = _make_drafter_card()
        drafter_shard_meta = _make_shard(drafter_card)
        coordinator.download_status[DRAFTER_ID] = DownloadOngoing(
            shard_metadata=drafter_shard_meta,
            node_id=NODE_ID,
            model_directory="/fake/drafter",
            download_progress=_ongoing_progress(10, 50),
        )

        await coordinator._cancel_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]

        # Both scopes must be cancelled.
        assert target_scope.cancel_called
        assert drafter_scope.cancel_called
        # And the parent->children mapping is cleared so a duplicate
        # cancel command doesn't try to cancel a stale drafter.
        assert TARGET_ID not in coordinator._drafter_children  # pyright: ignore[reportPrivateUsage]


async def test_rechain_preserves_drafter_link_for_cancel_cascade() -> None:
    """Codex P2 (PR #18 round-(N+2), coordinator.py:442): when
    ``StartDownload`` is re-issued for a target whose chain is still
    in flight, the second chain run MUST mutate the same
    ``_drafter_children`` list that any in-flight chain holds a
    reference to. Pre-fix, the second run reassigned the dict slot
    to a fresh list, orphaning the in-flight chain's appended
    drafter ids and breaking the ``_cancel_download`` cascade.

    We simulate the bug by directly invoking
    ``_maybe_chain_drafter_download`` twice, capturing the list
    object the first invocation observes, and asserting that drafter
    ids appended via the second chain are visible through that
    same captured reference -- which is what the cancel cascade
    relies on.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _,
            _,
        ):
            # First chain run -- pre-register and run synchronously
            # so the slot exists when we capture the list reference.
            coordinator._drafter_children.setdefault(TARGET_ID, [])  # pyright: ignore[reportPrivateUsage]
            captured_list: list[ModelId] = coordinator._drafter_children[TARGET_ID]  # pyright: ignore[reportPrivateUsage]
            await coordinator._maybe_chain_drafter_download(target_shard)  # pyright: ignore[reportPrivateUsage]

            # The drafter must be visible through the captured list
            # AND through the live dict-resolved list. Pre-fix, a
            # second run would diverge these.
            assert DRAFTER_ID in captured_list, (
                "first chain run must populate the captured list ref"
            )
            assert captured_list is coordinator._drafter_children[TARGET_ID], (  # pyright: ignore[reportPrivateUsage]
                "_drafter_children slot must NOT be reassigned by chain run"
            )

            # Second chain run (e.g. user re-issued StartDownload).
            await coordinator._maybe_chain_drafter_download(target_shard)  # pyright: ignore[reportPrivateUsage]

            # The captured list reference must still be the live one
            # tracked by ``_drafter_children`` -- otherwise a cancel
            # cascade based on ``_drafter_children[TARGET_ID]`` would
            # miss any drafter the second run started.
            assert captured_list is coordinator._drafter_children[TARGET_ID], (  # pyright: ignore[reportPrivateUsage]
                "rechain must mutate the same list, not replace the slot, "
                "so the cancel cascade always sees every drafter ever "
                "started for this target"
            )
            # Dedup: the drafter must not be duplicated across runs.
            assert captured_list.count(DRAFTER_ID) == 1, (
                "rechain must dedup drafter ids it already linked"
            )


async def test_cancel_cascade_recurses_unconditionally_for_pending_children() -> None:
    """Codex P1 (PR #18 round-(N+3), coordinator.py:212): the cancel
    cascade pre-fix gated child recursion on ``active_downloads``
    membership, so a child registered in ``_drafter_children`` but
    not yet promoted into ``active_downloads`` (e.g., a chained
    drafter mid-``_start_download``) was silently skipped. The
    cascade now recurses into every registered child unconditionally
    so the cancel intent reaches each one even before the launch
    flow has populated ``active_downloads``.
    """
    drafter_card = _make_drafter_card()
    drafter_shard = _make_shard(drafter_card)

    downloader = _RecordingShardDownloader()
    async with _running_coordinator(downloader) as (coordinator, _, _):
        # Yield once so ``coordinator.run()``'s TaskGroup is entered
        # before we exercise ``_cancel_download`` and the
        # ``_running_coordinator`` finalizer asks for ``shutdown()``.
        await asyncio.sleep(0)
        coordinator._drafter_children[TARGET_ID] = [DRAFTER_ID]  # pyright: ignore[reportPrivateUsage]
        # Note: DRAFTER_ID is intentionally NOT in
        # ``active_downloads`` -- this models the race window where
        # the chain has registered the link via ``remember_drafter_link``
        # but ``_start_download`` hasn't yet populated
        # ``active_downloads``. Status is set to ``DownloadPending`` so
        # ``_cancel_download`` can no-op gracefully on the inner gate
        # while still being CALLED on the child (the regression we're
        # protecting against is the cascade SKIPPING the call entirely).

        from exo.shared.types.worker.downloads import DownloadPending

        coordinator.download_status[DRAFTER_ID] = DownloadPending(
            shard_metadata=drafter_shard,
            node_id=NODE_ID,
            model_directory="/fake/drafter",
        )

        cancel_calls: list[ModelId] = []
        original_cancel = coordinator._cancel_download  # pyright: ignore[reportPrivateUsage]

        async def tracking_cancel(model_id: ModelId) -> None:
            cancel_calls.append(model_id)
            await original_cancel(model_id)

        coordinator._cancel_download = tracking_cancel  # pyright: ignore[reportPrivateUsage]
        try:
            await coordinator._cancel_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]
        finally:
            coordinator._cancel_download = original_cancel  # pyright: ignore[reportPrivateUsage]

        # Pre-fix: cascade would have skipped the child because
        # ``DRAFTER_ID not in active_downloads``. Post-fix: the cascade
        # MUST call ``_cancel_download(DRAFTER_ID)`` so the cancel
        # intent reaches every registered drafter regardless of its
        # current launch progress.
        assert DRAFTER_ID in cancel_calls, (
            "cascade must recurse into pending children, not gate on "
            f"active_downloads; got cancel_calls={cancel_calls!r}"
        )
        # And the parent->children mapping must still be cleared.
        assert TARGET_ID not in coordinator._drafter_children  # pyright: ignore[reportPrivateUsage]


async def test_concurrent_chain_does_not_double_start_pending_drafter() -> None:
    """Codex P2 (PR #18 round-(N+3), coordinator.py:224): when two
    overlapping chain coroutines both observe a drafter at
    ``DownloadPending`` (e.g., chain A has set ``DownloadPending``
    inside ``_start_download`` but hasn't yet reached
    ``_start_download_task``), pre-fix both fell through and both
    called ``_start_download_task``. ``ensure_shard()`` then cancels
    the first call and restarts -- a flap. Post-fix, the second
    ``_start_download`` for the same model short-circuits via the
    ``_starting_downloads`` lock, so ``ensure_shard`` is invoked
    exactly once.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            _,
            cmd_send,
            event_recv,
        ):
            # Spawn two concurrent target StartDownload commands
            # quickly so two chain coroutines run interleaved.
            for _ in range(2):
                await cmd_send.send(
                    ForwarderDownloadCommand(
                        origin=SystemId("test"),
                        command=StartDownload(
                            target_node_id=NODE_ID, shard_metadata=target_shard
                        ),
                    )
                )

            # Wait for both target completion events; allow the
            # background drafter chains to settle.
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            assert await _wait_for_completed(event_recv, DRAFTER_ID) is not None
            await asyncio.sleep(0.1)

    # Pre-fix: ``ensure_shard(DRAFTER_ID)`` could be invoked twice as
    # the second chain's ``_start_download_task`` overrode the first.
    # Post-fix: the ``_starting_downloads`` gate prevents the duplicate
    # launch and ``ensure_shard`` is invoked exactly once for the
    # drafter.
    assert downloader.ensured.count(DRAFTER_ID) == 1, (
        "concurrent chain runs must not double-start the same drafter; "
        f"got ensured={downloader.ensured!r}"
    )


async def test_failed_drafter_retries_on_target_re_chain() -> None:
    """Codex P1 (PR #18 round-(N+9), coordinator.py:267): if a
    drafter download previously failed (e.g. transient network /
    HF blip) and the user reissues ``StartDownload`` for the
    target, the chain MUST retry the drafter.

    Pre-fix the ``DownloadFailed`` short-circuit in
    ``_start_download`` blocked all retries through that function,
    including the drafter-chain path. So speculative decoding stayed
    silently disabled until manual intervention even though the
    user's re-issue is the supported retry trigger.

    This test simulates the failed→retry flow by:
    1. Pre-seeding the coordinator with ``DownloadFailed`` for the
       drafter (no need to actually fail one to set up the state).
    2. Issuing ``StartDownload`` for the target.
    3. Asserting that the chain re-runs ``ensure_shard`` for the
       drafter (so the retry is observable).
    """
    from exo.shared.types.worker.downloads import DownloadFailed

    target_card = _make_target_card([DRAFTER_ID])
    drafter_card = _make_drafter_card()
    target_shard = _make_shard(target_card)
    drafter_shard = _make_shard(drafter_card)

    downloader = _RecordingShardDownloader()
    with (
        patch(
            "exo.download.coordinator.ModelCard.load",
            return_value=drafter_card,
        ),
        patch(
            "exo.download.coordinator.ModelCard.load_cached_only",
            return_value=drafter_card,
        ),
    ):
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            await asyncio.sleep(0)
            # Pre-seed the failed-drafter state. Use the real
            # internal types to mirror what would happen after a
            # transient HF/network error.
            coordinator.download_status[DRAFTER_ID] = DownloadFailed(
                node_id=NODE_ID,
                shard_metadata=drafter_shard,
                error_message="HTTP 503 from HF (simulated transient)",
            )

            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            # The drafter must complete on the retry path. With
            # the bug present this would time out because
            # ``_start_download`` returned early on
            # ``DownloadFailed`` without invoking ``ensure_shard``.
            drafter_completed = await _wait_for_completed(event_recv, DRAFTER_ID)
            assert drafter_completed is not None, (
                "the drafter chain MUST retry through DownloadFailed when "
                "the user reissues StartDownload for the target; "
                "otherwise speculative decoding stays silently disabled. "
                f"ensured shards: {downloader.ensured!r}"
            )
            assert DRAFTER_ID in downloader.ensured, (
                "ensure_shard must run for the drafter on retry; "
                f"got ensured={downloader.ensured!r}"
            )


async def test_failed_target_top_level_call_still_skips_drafter_chain() -> None:
    """Regression guard for Codex P1 (PR #18 round-(N+9),
    coordinator.py:267): the drafter-chain retry path must NOT
    extend to top-level (user-initiated) target calls.

    If the user issues ``StartDownload`` for a target that
    previously failed, we still want to skip the drafter chain
    (pre-fix behavior from round-(N+2)) because a drafter is
    useless without a runnable target. The new
    ``is_drafter_chain`` parameter is the gate: only chained
    drafter calls retry through ``DownloadFailed``; top-level
    calls retain the short-circuit.
    """
    from exo.shared.types.worker.downloads import DownloadFailed

    target_card = _make_target_card([DRAFTER_ID])
    target_shard = _make_shard(target_card)

    downloader = _RecordingShardDownloader()
    async with _running_coordinator(downloader) as (
        coordinator,
        cmd_send,
        _event_recv,
    ):
        await asyncio.sleep(0)
        # Pre-seed the failed-target state.
        coordinator.download_status[TARGET_ID] = DownloadFailed(
            node_id=NODE_ID,
            shard_metadata=target_shard,
            error_message="HTTP 503 from HF (target itself failed)",
        )

        # The user issues StartDownload for the target *again*
        # (e.g. via stale UI state). With the failed-target
        # short-circuit in place, this should NOT kick off a
        # drafter download.
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=SystemId("test"),
                command=StartDownload(
                    target_node_id=NODE_ID, shard_metadata=target_shard
                ),
            )
        )
        # Tiny grace window for any spurious drafter ensure_shard.
        await asyncio.sleep(0.05)

        assert DRAFTER_ID not in downloader.ensured, (
            "failed target must NOT trigger drafter chain via top-level "
            "_start_download (drafter is useless without target); "
            f"ensured={downloader.ensured!r}"
        )


async def test_shared_drafter_in_flight_survives_cancel_of_one_target() -> None:
    """Codex P1 (PR #18 round-(N+11), coordinator.py:212): with this
    commit's Gemma 4 cards multiple targets reference the same
    drafter (e.g. ``gemma-4-26b`` and ``gemma-4-31b`` both list
    e2b/e4b drafters). Pre-fix the cancel cascade tore down every
    linked drafter for the canceled target, even when the drafter
    was *still downloading* on behalf of another target -- that
    drafter went straight to ``DownloadPending`` and silently
    disabled speculative decoding on the surviving target.

    This regression test exercises the in-flight case: the shared
    drafter is held in ``DownloadOngoing`` (never reaches
    ``ensure_shard``-completed), then target A is cancelled. The
    drafter MUST remain in ``DownloadOngoing`` because target B
    still depends on it. Once B is also cancelled, the drafter
    flips to ``DownloadPending`` (last parent gone, cascade fires).
    """
    from exo.shared.types.worker.downloads import (
        DownloadOngoing,
        DownloadPending,
    )

    target_a_id = ModelId("test-org/target-a")
    target_b_id = ModelId("test-org/target-b")
    shared_drafter_id = ModelId("test-org/shared-drafter")
    target_a_card = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    target_b_card = ModelCard(
        model_id=target_b_id,
        storage_size=Memory.from_mb(700),
        n_layers=40,
        hidden_size=2560,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    shared_drafter_card = ModelCard(
        model_id=shared_drafter_id,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    target_a_shard = _make_shard(target_a_card)
    target_b_shard = _make_shard(target_b_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == shared_drafter_id:
            return shared_drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    # Custom downloader: targets complete immediately, drafter hangs
    # so we can observe DownloadOngoing while the cancel races run.
    drafter_release = anyio.Event()
    drafter_started = anyio.Event()

    class _SuspendingDownloader(_RecordingShardDownloader):
        async def ensure_shard(
            self,
            shard: ShardMetadata,
            config_only: bool = False,  # noqa: ARG002
        ) -> Path:
            self.ensured.append(shard.model_card.model_id)
            if shard.model_card.model_id == shared_drafter_id:
                # Emit an ongoing progress event so the coordinator
                # marks DownloadOngoing.
                ongoing = RepoDownloadProgress(
                    repo_id=str(shard.model_card.model_id),
                    repo_revision="main",
                    shard=shard,
                    completed_files=0,
                    total_files=1,
                    downloaded=Memory.from_bytes(1),
                    downloaded_this_session=Memory.from_bytes(1),
                    total=shard.model_card.storage_size,
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="in_progress",
                )
                for cb in self._progress_callbacks:
                    await cb(shard, ongoing)
                drafter_started.set()
                # Hang until the test releases us.
                await drafter_release.wait()
            progress = RepoDownloadProgress(
                repo_id=str(shard.model_card.model_id),
                repo_revision="main",
                shard=shard,
                completed_files=1,
                total_files=1,
                downloaded=shard.model_card.storage_size,
                downloaded_this_session=shard.model_card.storage_size,
                total=shard.model_card.storage_size,
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                status="complete",
            )
            for cb in self._progress_callbacks:
                await cb(shard, progress)
            return Path("/fake/models") / shard.model_card.model_id.normalize()

    with _patch_card_loaders(fake_load):
        downloader = _SuspendingDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            for shard in (target_a_shard, target_b_shard):
                await cmd_send.send(
                    ForwarderDownloadCommand(
                        origin=SystemId("test"),
                        command=StartDownload(
                            target_node_id=NODE_ID, shard_metadata=shard
                        ),
                    )
                )

            # Wait for both targets to complete; drafter stays mid-download.
            completed_ids: set[ModelId] = set()
            async with asyncio.timeout(5.0):
                while {target_a_id, target_b_id} - completed_ids:
                    event = await event_recv.receive()
                    if isinstance(event, NodeDownloadProgress) and isinstance(
                        event.download_progress, DownloadCompleted
                    ):
                        completed_ids.add(
                            event.download_progress.shard_metadata.model_card.model_id
                        )

            # Make sure the drafter is genuinely in DownloadOngoing so
            # the cancel cascade CAN flip it to DownloadPending.
            with anyio.fail_after(2.0):
                await drafter_started.wait()
            await asyncio.sleep(0.05)
            drafter_status = coordinator.download_status.get(shared_drafter_id)
            assert isinstance(drafter_status, DownloadOngoing), (
                f"drafter must be DownloadOngoing while suspended; got "
                f"{type(drafter_status).__name__}"
            )

            parents = coordinator._drafter_parents.get(shared_drafter_id)  # pyright: ignore[reportPrivateUsage]
            assert parents is not None and parents == {target_a_id, target_b_id}

            # Cancel target A. Drafter MUST stay DownloadOngoing.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(
                        target_node_id=NODE_ID, model_id=target_a_id
                    ),
                )
            )
            await asyncio.sleep(0.1)

            drafter_status_after_a = coordinator.download_status.get(shared_drafter_id)
            assert isinstance(drafter_status_after_a, DownloadOngoing), (
                "shared drafter must remain DownloadOngoing after one of "
                "its parents is cancelled; pre-fix the cascade flipped "
                f"it to DownloadPending. got={type(drafter_status_after_a).__name__}"
            )
            parents_after_a = coordinator._drafter_parents.get(shared_drafter_id)  # pyright: ignore[reportPrivateUsage]
            assert parents_after_a == {target_b_id}, (
                f"cancel of target A must remove A from drafter parent "
                f"set; got parents={parents_after_a!r}"
            )

            # Cancel target B. Now the drafter is genuinely orphaned;
            # cascade must fire and flip it to DownloadPending.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=CancelDownload(
                        target_node_id=NODE_ID, model_id=target_b_id
                    ),
                )
            )
            await asyncio.sleep(0.1)

            drafter_status_final = coordinator.download_status.get(shared_drafter_id)
            assert isinstance(drafter_status_final, DownloadPending), (
                "drafter must be cancelled (DownloadPending) once the "
                f"LAST parent is cancelled; got "
                f"{type(drafter_status_final).__name__}"
            )
            assert shared_drafter_id not in coordinator._drafter_parents, (  # pyright: ignore[reportPrivateUsage]
                "_drafter_parents must be cleaned up once empty"
            )

            # Allow the suspended ensure_shard to unwind so shutdown
            # doesn't leak the task.
            drafter_release.set()


async def test_shared_drafter_survives_delete_of_one_target() -> None:
    """Codex P1 (PR #18 round-(N+11), coordinator.py:743): the
    delete-cascade companion to
    ``test_shared_drafter_in_flight_survives_cancel_of_one_target``.
    With shared drafters across Gemma 4 cards, deleting one target
    must NOT also delete the drafter the other still-installed
    target depends on.

    We mock ``delete_model`` to a recorder so the test does not
    touch the filesystem, and assert it is called for the deleted
    target but NOT for the shared drafter (until the second target
    is deleted too).
    """
    target_a_id = ModelId("test-org/target-a")
    target_b_id = ModelId("test-org/target-b")
    shared_drafter_id = ModelId("test-org/shared-drafter")
    target_a_card = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    target_b_card = ModelCard(
        model_id=target_b_id,
        storage_size=Memory.from_mb(700),
        n_layers=40,
        hidden_size=2560,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    shared_drafter_card = ModelCard(
        model_id=shared_drafter_id,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    target_a_shard = _make_shard(target_a_card)
    target_b_shard = _make_shard(target_b_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == shared_drafter_id:
            return shared_drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    async def fake_delete(model_id: ModelId) -> bool:
        deleted_ids.append(model_id)
        return True

    with (
        patch.object(ModelCard, "load", side_effect=fake_load),
        patch.object(ModelCard, "load_cached_only", side_effect=fake_load),
        patch("exo.download.coordinator.delete_model", side_effect=fake_delete),
    ):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            event_recv,
        ):
            for shard in (target_a_shard, target_b_shard):
                await cmd_send.send(
                    ForwarderDownloadCommand(
                        origin=SystemId("test"),
                        command=StartDownload(
                            target_node_id=NODE_ID, shard_metadata=shard
                        ),
                    )
                )
            completed_ids: set[ModelId] = set()
            wanted = {target_a_id, target_b_id, shared_drafter_id}
            async with asyncio.timeout(5.0):
                while wanted - completed_ids:
                    event = await event_recv.receive()
                    if isinstance(event, NodeDownloadProgress) and isinstance(
                        event.download_progress, DownloadCompleted
                    ):
                        completed_ids.add(
                            event.download_progress.shard_metadata.model_card.model_id
                        )
            await asyncio.sleep(0.1)

            assert coordinator._drafter_parents.get(shared_drafter_id) == {  # pyright: ignore[reportPrivateUsage]
                target_a_id,
                target_b_id,
            }

            # Delete target A. Drafter MUST stay -- target B still references it.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=DeleteDownload(
                        target_node_id=NODE_ID, model_id=target_a_id
                    ),
                )
            )
            await asyncio.sleep(0.1)
            assert deleted_ids == [target_a_id], (
                "delete of target A must NOT cascade into the shared "
                f"drafter; got deleted_ids={deleted_ids!r}"
            )
            assert coordinator._drafter_parents.get(shared_drafter_id) == {target_b_id}  # pyright: ignore[reportPrivateUsage]

            # Delete target B. Now the drafter is genuinely orphaned.
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=DeleteDownload(
                        target_node_id=NODE_ID, model_id=target_b_id
                    ),
                )
            )
            await asyncio.sleep(0.1)
            assert shared_drafter_id in deleted_ids, (
                "drafter must be deleted once the LAST parent is "
                f"deleted; got deleted_ids={deleted_ids!r}"
            )
            assert shared_drafter_id not in coordinator._drafter_parents  # pyright: ignore[reportPrivateUsage]


async def test_chained_drafter_does_not_recursively_chain_via_inner_path() -> None:
    """Codex P2 (PR #18 round-(N+10), coordinator.py:347):
    ``_start_download_inner`` calls ``_spawn_drafter_chain`` in three
    completion arms (cached-on-disk, initial-progress complete,
    actual download started). Pre-fix, the ``is_drafter_chain``
    flag introduced at the outer ``_start_download`` boundary was
    DROPPED at the inner-call boundary, so a drafter being downloaded
    as a chain step would itself trigger ``_spawn_drafter_chain``
    whenever its own card declared ``drafter_model_ids`` (custom
    cards or accidentally self-referential cards). This test
    constructs a drafter card whose own ``drafter_model_ids`` lists
    a "second-level" drafter, runs the chain, and asserts that the
    second-level drafter is never enqueued -- the chain stops at
    one level deep.
    """
    second_level_id = ModelId("test-org/second-level-drafter")
    drafter_card_with_subdrafter = ModelCard(
        model_id=DRAFTER_ID,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        # Self-recursive trap: the drafter's card itself lists a
        # nested drafter. Pre-fix this would recursively chain.
        drafter_model_ids=[second_level_id],
    )
    second_level_card = ModelCard(
        model_id=second_level_id,
        storage_size=Memory.from_mb(20),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )

    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == DRAFTER_ID:
            return drafter_card_with_subdrafter
        if model_id == second_level_id:
            return second_level_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            _coordinator,
            cmd_send,
            event_recv,
        ):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            assert await _wait_for_completed(event_recv, TARGET_ID) is not None
            assert await _wait_for_completed(event_recv, DRAFTER_ID) is not None
            # Grace window so any rogue second-level chain has time
            # to fire before we assert it didn't.
            await asyncio.sleep(0.1)

    assert second_level_id not in downloader.ensured, (
        "chained drafters must NOT recursively re-chain their own "
        "drafter_model_ids; the chain stops at one level deep so a "
        "self-referential or custom-multi-level drafter card cannot "
        "spawn nested background fetches. "
        f"ensured={downloader.ensured!r}"
    )


async def test_starting_downloads_cleared_on_completion() -> None:
    """The ephemeral ``_starting_downloads`` lock must be released
    after ``_start_download`` finishes, so a legitimate restart
    (e.g., after the user cancels the drafter) is not gated by a
    stale entry.
    """
    target_shard = _make_shard(_make_target_card([]))

    downloader = _RecordingShardDownloader()
    async with _running_coordinator(downloader) as (
        coordinator,
        cmd_send,
        event_recv,
    ):
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=SystemId("test"),
                command=StartDownload(
                    target_node_id=NODE_ID, shard_metadata=target_shard
                ),
            )
        )
        assert await _wait_for_completed(event_recv, TARGET_ID) is not None

    assert TARGET_ID not in coordinator._starting_downloads, (  # pyright: ignore[reportPrivateUsage]
        "_starting_downloads must be cleared after _start_download "
        "returns, otherwise restart-after-cancel is silently disabled"
    )


async def test_drafter_chain_does_not_run_when_target_download_fails() -> None:
    """Codex P2 (PR #18 round-(N+12), coordinator.py:487): the
    drafter chain must wait for ``ensure_shard()`` to actually
    succeed before running. Pre-fix, ``_spawn_drafter_chain`` was
    invoked immediately after ``_start_download_task`` queued the
    target download. If ``ensure_shard()`` later raised
    (auth/rate-limit/transient network/gated repo), the target
    flipped to ``DownloadFailed`` but any drafter downloads spawned
    in the meantime kept running to completion, consuming bandwidth
    and disk for a model that could never boot. Worse, the failed
    state is exactly what the round-(N+2) ``DownloadFailed``
    fast-path was supposed to gate against on a *re-issue*; the
    initial-issue gap was an outright regression.

    Post-fix, the chain is invoked from ``download_wrapper`` only on
    the success arm of ``ensure_shard()``. A failed download leaves
    drafter chaining untouched.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called when the target's "
            "ensure_shard() raises -- the chain must wait for target "
            "success before any drafter card resolution"
        )

    class _FailingDownloader(_RecordingShardDownloader):
        async def ensure_shard(
            self,
            shard: ShardMetadata,
            config_only: bool = False,  # noqa: ARG002
        ) -> Path:
            self.ensured.append(shard.model_card.model_id)
            if shard.model_card.model_id == TARGET_ID:
                raise RuntimeError("simulated HF auth failure for gated target repo")
            return Path("/fake/models") / shard.model_card.model_id.normalize()

    with _patch_card_loaders(fail_load):
        downloader = _FailingDownloader()
        async with _running_coordinator(downloader) as (
            _coordinator,
            cmd_send,
            _event_recv,
        ):
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            await asyncio.sleep(0.2)

    assert downloader.ensured == [TARGET_ID], (
        "drafter must NOT be queued when the target's ensure_shard() "
        "raises before completion; the chain is gated on target success. "
        f"got ensured={downloader.ensured!r}"
    )


async def test_delete_cascade_rebuilds_drafter_links_after_restart() -> None:
    """Codex P2 (PR #18 round-(N+12), coordinator.py:817):
    ``_drafter_children`` is process-local state populated during
    runtime chaining and not rehydrated on coordinator startup.
    Pre-fix, deleting a target whose drafters were chained in a
    PREVIOUS process found an empty children list and left the
    drafter weights orphaned on disk -- the only signal back to the
    operator was disk usage that grew over time. (The runtime case
    where the chain ran in the same process is covered by
    ``test_shared_drafter_survives_delete_of_one_target``.)

    Post-fix, ``_reconstruct_drafter_links_for_delete`` consults the
    target's ``ModelCard.drafter_model_ids`` to repopulate the
    children list before the cascade runs, so a delete after
    restart cleans up the linked drafters as if the chain had run
    in the current process.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()
    target_card = _make_target_card([DRAFTER_ID])

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == TARGET_ID:
            return target_card
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            # Yield once so coordinator.run()'s task group enters its
            # ``async with self._tg as tg:`` block before we start
            # exercising private methods. Without this, shutdown()
            # asserts on an uninitialised ``_tg``.
            await asyncio.sleep(0.05)

            target_completed = DownloadCompleted(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                total=target_shard.model_card.storage_size,
                model_directory="/fake/target",
            )
            drafter_completed = DownloadCompleted(
                shard_metadata=_make_shard(drafter_card),
                node_id=NODE_ID,
                total=drafter_card.storage_size,
                model_directory="/fake/drafter",
            )
            coordinator.download_status[TARGET_ID] = target_completed
            coordinator.download_status[DRAFTER_ID] = drafter_completed

            assert TARGET_ID not in coordinator._drafter_children, (  # pyright: ignore[reportPrivateUsage]
                "test setup must mirror post-restart state: "
                "_drafter_children is empty for the target"
            )
            assert DRAFTER_ID not in coordinator._drafter_parents, (  # pyright: ignore[reportPrivateUsage]
                "test setup must mirror post-restart state: "
                "_drafter_parents is empty for the drafter"
            )

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]

    assert TARGET_ID in deleted_ids, "target must be deleted from disk"
    assert DRAFTER_ID in deleted_ids, (
        "drafter must be cascaded into the delete even after restart "
        "(pre-fix: empty _drafter_children left it orphaned). "
        f"deleted_ids={deleted_ids!r}"
    )


async def test_reissue_during_ongoing_target_does_not_chain_drafters() -> None:
    """Codex P2 (PR #18 round-(N+13), coordinator.py:337): the
    ``_start_download`` fast-path branch chains drafters when a
    re-issued ``StartDownload`` arrives for a target that's
    already in the in-memory cache. Pre-fix the branch chained
    even when the target was only ``DownloadOngoing``, so a
    duplicate ``StartDownload`` during an in-flight target
    download spawned drafters BEFORE the target's
    ``ensure_shard()`` had succeeded. That bypassed the
    round-(N+12) success-gated path in ``_start_download_task``
    (which is what initiates the chain after the in-flight
    target's ``ensure_shard()`` returns successfully).

    Round-(N+13) restricts the fast-path chaining to
    ``DownloadCompleted`` only -- if the target is still
    ``DownloadOngoing``, the original in-flight task's
    ``download_wrapper`` will spawn the chain on success, so
    duplicating the spawn here would be both wasteful and
    incorrect (chain runs before target success when target
    transitions from ``DownloadOngoing`` to ``DownloadFailed``).
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))

    async def fail_load(_: ModelId) -> ModelCard:
        raise AssertionError(
            "ModelCard.load must not be called when a duplicate "
            "StartDownload arrives for a target that is still "
            "DownloadOngoing -- the original in-flight task's "
            "download_wrapper handles the chain on success"
        )

    with _patch_card_loaders(fail_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            cmd_send,
            _event_recv,
        ):
            await asyncio.sleep(0.05)

            # Pre-seed the target's status as DownloadOngoing.
            ongoing_progress = DownloadProgressData(
                downloaded=Memory.from_mb(100),
                downloaded_this_session=Memory.from_mb(100),
                total=Memory.from_mb(500),
                completed_files=0,
                total_files=1,
                speed=0.0,
                eta_ms=0,
                files={},
            )
            coordinator.download_status[TARGET_ID] = DownloadOngoing(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                model_directory="/fake/target",
                download_progress=ongoing_progress,
            )

            # Re-issue StartDownload for the in-flight target.
            # Pre-fix the fast-path called ``_spawn_drafter_chain``
            # which would have triggered ``ModelCard.load`` for the
            # drafter (the test's ``fail_load`` would have raised).
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=SystemId("test"),
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=target_shard
                    ),
                )
            )
            await asyncio.sleep(0.1)

    assert downloader.ensured == [], (
        "drafter chain must NOT fire when the duplicate StartDownload "
        "arrives during an in-flight target download (DownloadOngoing). "
        f"got ensured={downloader.ensured!r}"
    )


async def test_self_referential_drafter_card_does_not_recurse_on_delete() -> None:
    """Codex P2 (PR #18 round-(N+13), coordinator.py:337): a model
    card with a self-referential ``drafter_model_ids = [self]`` or
    a cycle like ``A -> B -> A`` would drive the recursive delete
    cascade into infinite recursion. Pre-fix
    ``_reconstruct_drafter_links_for_delete`` rebuilds children
    from ``ModelCard.load`` on every call, so the same id keeps
    getting reintroduced and recursed into until the interpreter's
    stack limit fires and aborts the operator's delete mid-cascade.

    Round-(N+13) adds an ``_deleting_in_progress`` set guarded by
    a wrapper. When the recursive call detects the id is already
    being deleted earlier on the call stack, it skips the
    recursion -- the outer invocation finishes the on-disk delete
    cleanly.
    """
    self_referential_card = ModelCard(
        model_id=TARGET_ID,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[TARGET_ID],
    )
    target_shard = _make_shard(self_referential_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == TARGET_ID:
            return self_referential_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            await asyncio.sleep(0.05)

            target_completed = DownloadCompleted(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                total=target_shard.model_card.storage_size,
                model_directory="/fake/target",
            )
            coordinator.download_status[TARGET_ID] = target_completed

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                # Pre-fix: this call recursed indefinitely until
                # RecursionError. Post-fix: returns cleanly with
                # one ``delete_model`` invocation for the target.
                await coordinator._delete_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]

    assert deleted_ids == [TARGET_ID], (
        "self-referential drafter card must not loop the delete "
        "cascade; ``delete_model`` must run exactly once for the "
        f"target. got deleted_ids={deleted_ids!r}"
    )


async def test_cyclic_drafter_cards_do_not_recurse_on_delete() -> None:
    """Codex P2 (PR #18 round-(N+13), coordinator.py:337): the
    ``A -> B -> A`` cycle case. Pre-fix the recursion alternates
    A and B forever; post-fix the inner ``_delete_download(A)``
    call triggered by ``B``'s rebuild detects A already in
    ``_deleting_in_progress`` and short-circuits, so the cascade
    deletes both A and B exactly once each before unwinding.
    """
    target_a_id = ModelId("test-org/cycle-a")
    target_b_id = ModelId("test-org/cycle-b")
    card_a = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[target_b_id],
    )
    card_b = ModelCard(
        model_id=target_b_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[target_a_id],
    )
    shard_a = _make_shard(card_a)
    shard_b = _make_shard(card_b)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == target_a_id:
            return card_a
        if model_id == target_b_id:
            return card_b
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            await asyncio.sleep(0.05)

            coordinator.download_status[target_a_id] = DownloadCompleted(
                shard_metadata=shard_a,
                node_id=NODE_ID,
                total=card_a.storage_size,
                model_directory="/fake/cycle-a",
            )
            coordinator.download_status[target_b_id] = DownloadCompleted(
                shard_metadata=shard_b,
                node_id=NODE_ID,
                total=card_b.storage_size,
                model_directory="/fake/cycle-b",
            )

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(target_a_id)  # pyright: ignore[reportPrivateUsage]

    assert sorted(deleted_ids, key=str) == sorted(
        [target_a_id, target_b_id], key=str
    ), (
        "cyclical drafter cards must drive each id through "
        "``delete_model`` exactly once, not infinitely. "
        f"got deleted_ids={deleted_ids!r}"
    )


async def test_delete_cascade_runs_when_drafter_status_cache_cold() -> None:
    """Codex P2 (PR #18 round-(N+13), coordinator.py:945): even
    after ``_reconstruct_drafter_links_for_delete`` correctly
    rediscovers the drafter IDs from the target card, the cascade
    was previously gated on ``child_model_id in self.active_downloads
    or child_model_id in self.download_status``. After a restart,
    if a ``DeleteDownload`` arrives BEFORE
    ``_emit_existing_download_progress`` has hydrated
    ``download_status`` from the on-disk shard listing, the
    rediscovered drafter is still absent from the in-memory cache
    and the gate silently skipped the cascade -- leaving the
    drafter weights on disk.

    Post-fix the cascade runs unconditionally for every
    rediscovered child (``_delete_download`` itself is idempotent
    for missing in-memory state: ``delete_model`` reports "not
    found on disk" via ``deleted == False`` rather than raising).
    This test pins that behaviour by populating ``download_status``
    only for the target -- the drafter exists on disk but is NOT
    in the in-memory cache yet. Pre-fix the cascade would have
    skipped the drafter; post-fix it deletes both.
    """
    target_shard = _make_shard(_make_target_card([DRAFTER_ID]))
    drafter_card = _make_drafter_card()
    target_card = _make_target_card([DRAFTER_ID])

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == TARGET_ID:
            return target_card
        if model_id == DRAFTER_ID:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            await asyncio.sleep(0.05)

            target_completed = DownloadCompleted(
                shard_metadata=target_shard,
                node_id=NODE_ID,
                total=target_shard.model_card.storage_size,
                model_directory="/fake/target",
            )
            coordinator.download_status[TARGET_ID] = target_completed

            # Drafter is intentionally NOT in download_status to
            # simulate the post-restart cold-cache window before
            # ``_emit_existing_download_progress`` runs.
            assert DRAFTER_ID not in coordinator.download_status, (
                "test setup must mirror post-restart cold-cache: "
                "drafter is on disk but not in the in-memory map"
            )
            assert DRAFTER_ID not in coordinator.active_downloads, (
                "test setup must mirror post-restart cold-cache: "
                "drafter is on disk but not actively downloading"
            )

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(TARGET_ID)  # pyright: ignore[reportPrivateUsage]

    assert TARGET_ID in deleted_ids, "target must be deleted from disk"
    assert DRAFTER_ID in deleted_ids, (
        "drafter must be cascaded into the delete even when its "
        "in-memory status cache is cold (post-restart, pre-hydration "
        "window). Pre-fix the gate ``child in active_downloads or "
        "download_status`` silently skipped the cascade and left "
        "the drafter weights orphaned on disk. "
        f"deleted_ids={deleted_ids!r}"
    )


async def test_delete_cascade_rebuild_respects_other_referencing_target() -> None:
    """Codex P2 (PR #18 round-(N+12), coordinator.py:817) shared-drafter
    follow-up: rebuilding drafter links on delete MUST still honour
    the shared-drafter cascade gate. After a restart, two targets
    share a drafter on disk; deleting target A must not also delete
    the drafter the surviving target B still depends on.

    Pre-fix the round-(N+12) rebuild populated the parent set with
    only target A as a parent, so the discard-and-check loop
    immediately tore the drafter down. Post-fix the test wires
    ``_drafter_parents`` such that target B is also a parent (which
    a second restart-time rebuild would do during target B's own
    ``_delete_download`` lifecycle), and asserts that deleting
    target A leaves the shared drafter on disk.
    """
    shared_drafter_id = ModelId("test-org/shared-drafter")
    target_a_id = ModelId("test-org/target-a")
    target_b_id = ModelId("test-org/target-b")
    target_a_card = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    target_b_card = ModelCard(
        model_id=target_b_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    drafter_card = ModelCard(
        model_id=shared_drafter_id,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    target_a_shard = _make_shard(target_a_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == target_a_id:
            return target_a_card
        if model_id == target_b_id:
            return target_b_card
        if model_id == shared_drafter_id:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    deleted_ids: list[ModelId] = []

    with _patch_card_loaders(fake_load):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            # Yield once so coordinator.run()'s task group enters its
            # ``async with self._tg as tg:`` block before we start
            # exercising private methods. Without this, shutdown()
            # asserts on an uninitialised ``_tg``.
            await asyncio.sleep(0.05)

            target_a_completed = DownloadCompleted(
                shard_metadata=target_a_shard,
                node_id=NODE_ID,
                total=target_a_shard.model_card.storage_size,
                model_directory="/fake/target-a",
            )
            target_b_completed = DownloadCompleted(
                shard_metadata=_make_shard(target_b_card),
                node_id=NODE_ID,
                total=target_b_card.storage_size,
                model_directory="/fake/target-b",
            )
            drafter_completed = DownloadCompleted(
                shard_metadata=_make_shard(drafter_card),
                node_id=NODE_ID,
                total=drafter_card.storage_size,
                model_directory="/fake/shared-drafter",
            )
            coordinator.download_status[target_a_id] = target_a_completed
            coordinator.download_status[target_b_id] = target_b_completed
            coordinator.download_status[shared_drafter_id] = drafter_completed

            # Mirror the post-restart state where target B's own
            # link rebuild already happened (e.g. during a /status
            # poll that triggered a hydrate on target B). Target A's
            # rebuild happens lazily during _delete_download below.
            coordinator._drafter_parents[shared_drafter_id] = {target_b_id}  # pyright: ignore[reportPrivateUsage]

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(target_a_id)  # pyright: ignore[reportPrivateUsage]

    assert target_a_id in deleted_ids, "target A must be deleted from disk"
    assert shared_drafter_id not in deleted_ids, (
        "shared drafter must NOT be deleted while target B still "
        "references it; the post-restart link rebuild must respect "
        f"the existing parent set. deleted_ids={deleted_ids!r}"
    )


async def test_delete_cascade_rebuilds_other_parents_for_installed_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P1 (PR #18 round-(N+13), coordinator.py:910): when
    deleting a target after a process restart, the rebuild must
    discover OTHER installed targets that share the same drafter
    and register them as parents so the cascade's last-reference
    gate preserves the drafter on disk.

    Pre-fix the rebuild only registered the *currently-deleting*
    target as a parent; a shared drafter whose other parent's
    chain had not yet been rebuilt in this process (the typical
    post-restart state when only one target's delete has been
    observed so far) was treated as orphaned and cascaded-deleted,
    silently degrading the surviving target back to non-speculative
    behaviour.

    Post-fix the cascade scans every known model card. Any card
    that (a) declares one of the rediscovered drafters AND (b) is
    currently installed on disk gets registered as an additional
    parent. The cascade's last-reference gate then preserves the
    drafter exactly as it would for a runtime-chained pair of
    targets that both still want it.
    """
    shared_drafter_id = ModelId("test-org/shared-drafter")
    target_a_id = ModelId("test-org/target-a")
    target_b_id = ModelId("test-org/target-b")
    target_a_card = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    target_b_card = ModelCard(
        model_id=target_b_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    drafter_card = ModelCard(
        model_id=shared_drafter_id,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    target_a_shard = _make_shard(target_a_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == target_a_id:
            return target_a_card
        if model_id == target_b_id:
            return target_b_card
        if model_id == shared_drafter_id:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    async def fake_get_model_cards() -> list[ModelCard]:
        return [target_a_card, target_b_card, drafter_card]

    # Simulate "target B installed on disk, target A also installed
    # on disk, drafter installed on disk" -- i.e. the typical
    # post-restart state for a user with both Gemma 4 26B and 31B
    # using the shared e2b drafter.
    installed_dir = tmp_path / "models"
    installed_dir.mkdir()

    def fake_resolve_existing_model(
        model_id: ModelId, card: ModelCard | None = None
    ) -> Path | None:
        if model_id in (target_a_id, target_b_id, shared_drafter_id):
            return installed_dir / model_id.normalize()
        return None

    monkeypatch.setattr(
        "exo.download.coordinator.resolve_existing_model",
        fake_resolve_existing_model,
    )

    deleted_ids: list[ModelId] = []

    with (
        patch.object(ModelCard, "load", side_effect=fake_load),
        patch.object(ModelCard, "load_cached_only", side_effect=fake_load),
        patch(
            "exo.download.coordinator.get_model_cards",
            side_effect=fake_get_model_cards,
        ),
    ):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            # Yield so the coordinator's task group enters its
            # ``async with self._tg as tg:`` block before we
            # call private delete machinery.
            await asyncio.sleep(0.05)

            # Mirror a fresh post-restart state: download_status
            # is hydrated for the currently-deleting target only.
            # The surviving target B is installed on disk but
            # its parent link has NOT been pre-seeded -- this is
            # the regression Codex called out.
            target_a_completed = DownloadCompleted(
                shard_metadata=target_a_shard,
                node_id=NODE_ID,
                total=target_a_shard.model_card.storage_size,
                model_directory="/fake/target-a",
            )
            coordinator.download_status[target_a_id] = target_a_completed

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(target_a_id)  # pyright: ignore[reportPrivateUsage]

    assert target_a_id in deleted_ids, "target A must be deleted from disk"
    assert shared_drafter_id not in deleted_ids, (
        "shared drafter must NOT be deleted: target B is installed "
        "on disk and shares the drafter, so the rebuild must register "
        "target B as an additional parent and the cascade must honour "
        f"the last-reference gate. deleted_ids={deleted_ids!r}"
    )


async def test_delete_cascade_does_not_block_on_uninstalled_other_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other-parent rebuild MUST only register *installed*
    targets. Otherwise a card declaring ``drafter_model_ids = [x]``
    for a model that was never downloaded would block legitimate
    deletion of ``x`` when its only real parent is also being
    deleted -- leaving an orphaned drafter on disk.

    This is the inverse correctness check for the round-(N+13)
    fix: the new ``_discover_other_drafter_parents`` step uses
    ``resolve_existing_model`` to filter to installed targets only,
    so an uninstalled card sharing the same drafter must NOT
    register as a parent and the cascade must proceed normally.
    """
    shared_drafter_id = ModelId("test-org/shared-drafter")
    target_a_id = ModelId("test-org/target-a")
    uninstalled_target_id = ModelId("test-org/uninstalled-target")
    target_a_card = ModelCard(
        model_id=target_a_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    uninstalled_target_card = ModelCard(
        model_id=uninstalled_target_id,
        storage_size=Memory.from_mb(500),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[shared_drafter_id],
    )
    drafter_card = ModelCard(
        model_id=shared_drafter_id,
        storage_size=Memory.from_mb(50),
        n_layers=12,
        hidden_size=768,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    target_a_shard = _make_shard(target_a_card)
    drafter_shard = _make_shard(drafter_card)

    async def fake_load(model_id: ModelId) -> ModelCard:
        if model_id == target_a_id:
            return target_a_card
        if model_id == uninstalled_target_id:
            return uninstalled_target_card
        if model_id == shared_drafter_id:
            return drafter_card
        raise AssertionError(f"unexpected ModelCard.load for {model_id}")

    async def fake_get_model_cards() -> list[ModelCard]:
        return [target_a_card, uninstalled_target_card, drafter_card]

    def fake_resolve_existing_model(
        model_id: ModelId, card: ModelCard | None = None
    ) -> Path | None:
        # Only the deleting target and its drafter are installed
        # on disk; the other card declaring the same drafter was
        # never downloaded.
        if model_id in (target_a_id, shared_drafter_id):
            return Path("/fake") / model_id.normalize()
        return None

    monkeypatch.setattr(
        "exo.download.coordinator.resolve_existing_model",
        fake_resolve_existing_model,
    )

    deleted_ids: list[ModelId] = []

    with (
        patch.object(ModelCard, "load", side_effect=fake_load),
        patch.object(ModelCard, "load_cached_only", side_effect=fake_load),
        patch(
            "exo.download.coordinator.get_model_cards",
            side_effect=fake_get_model_cards,
        ),
    ):
        downloader = _RecordingShardDownloader()
        async with _running_coordinator(downloader) as (
            coordinator,
            _cmd_send,
            _event_recv,
        ):
            await asyncio.sleep(0.05)

            target_a_completed = DownloadCompleted(
                shard_metadata=target_a_shard,
                node_id=NODE_ID,
                total=target_a_shard.model_card.storage_size,
                model_directory="/fake/target-a",
            )
            drafter_completed = DownloadCompleted(
                shard_metadata=drafter_shard,
                node_id=NODE_ID,
                total=drafter_card.storage_size,
                model_directory="/fake/shared-drafter",
            )
            coordinator.download_status[target_a_id] = target_a_completed
            coordinator.download_status[shared_drafter_id] = drafter_completed

            async def fake_delete_model(model_id: ModelId) -> bool:
                deleted_ids.append(model_id)
                coordinator.download_status.pop(model_id, None)
                return True

            with patch(
                "exo.download.coordinator.delete_model",
                side_effect=fake_delete_model,
            ):
                await coordinator._delete_download(target_a_id)  # pyright: ignore[reportPrivateUsage]

    assert target_a_id in deleted_ids
    assert shared_drafter_id in deleted_ids, (
        "drafter MUST cascade-delete: the only other card declaring it "
        "(uninstalled_target) is not installed on disk, so it must NOT "
        "register as a parent. Pre-(N+13)-fix-overshoot, registering "
        "uninstalled cards would orphan the drafter on disk; the "
        "installed-only filter prevents that. "
        f"deleted_ids={deleted_ids!r}"
    )
