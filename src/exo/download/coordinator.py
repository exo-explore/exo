from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, current_time, to_thread
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    is_read_only_model_dir,
    map_repo_download_progress_to_download_progress_data,
    resolve_existing_model,
)
from exo.download.impl_shard_downloader import SingletonShardDownloader
from exo.download.peer_shard_downloader import PeerAwareShardDownloader
from exo.download.shard_downloader import ShardDownloader
from exo.shared.constants import EXO_DEFAULT_MODELS_DIR, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup

# Mirrors the same env var consumed by the worker's MLX loader. Keeping the
# string literal in lockstep so users only need to set one variable to opt
# out of speculative decoding entirely (skips both download and load).
_DRAFTER_DISABLED_VALUES = frozenset({"1", "true", "yes"})


def _drafter_disabled_by_env() -> bool:
    return os.environ.get("EXO_DISABLE_DRAFTER", "").lower() in _DRAFTER_DISABLED_VALUES


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    offline: bool = False

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    # Map of target model_id -> drafter model_ids spawned alongside it.
    # When the user cancels or deletes the target, we propagate the
    # cancellation/deletion to its chained drafters so they don't keep
    # consuming network/disk after the user revoked the original
    # download intent. Populated only when the drafter chain actually
    # runs (offline/disabled-by-env paths short-circuit and add no
    # children).
    _drafter_children: dict[ModelId, list[ModelId]] = field(default_factory=dict)

    # Codex P1 (PR #18 round-(N+11), coordinator.py:212/743): reverse
    # mapping of drafter -> {target_ids_that_reference_it}. With this
    # commit's Gemma 4 cards multiple targets share the same drafter
    # (e.g. ``gemma-4-26b`` and ``gemma-4-31b`` both name the
    # ``gemma-4-e2b`` / ``gemma-4-e4b`` drafters). Pre-fix the
    # cancel/delete cascade unconditionally tore down every linked
    # drafter for the canceled/deleted target, so canceling one
    # silently disabled speculative decoding on the *other* still-
    # installed target -- the user only saw a regression in tokens/sec
    # and would not connect that to the unrelated cancel they issued.
    #
    # The reverse map is updated transactionally with
    # ``_drafter_children``: every ``remember_drafter_link`` adds the
    # current target to ``_drafter_parents[drafter_id]``, and every
    # cascade pops the *current* target from each child's parent set
    # but only actually cascades the cancel/delete when the child has
    # no remaining parents. This is many-to-many bookkeeping but the
    # cardinality is bounded by the cluster's installed model set
    # (single-digit drafters per cluster in practice).
    _drafter_parents: dict[ModelId, set[ModelId]] = field(default_factory=dict)

    # Codex P2 (PR #18 round-(N+3), coordinator.py:224): per-model
    # in-flight marker for ``_start_download``. Pre-fix, the function
    # treated only ``DownloadOngoing``/``DownloadCompleted`` as
    # in-flight, so concurrent chain coroutines could both observe
    # ``DownloadPending`` (set during the early ``DownloadPending``
    # emit) and fall through to ``_start_download_task``, racing
    # ``ensure_shard()`` and producing a cancel/restart flap. The set
    # also has to coexist with the post-cancel restart-after-cancel
    # path: ``_cancel_download`` leaves ``download_status`` at
    # ``DownloadPending`` after a user cancel, but the cancelled
    # ``_start_download`` is no longer in ``_starting_downloads``,
    # so a follow-up ``StartDownload`` correctly re-enters the
    # download-launch flow. ``active_downloads`` cannot serve as the
    # gate by itself: it's only populated late in
    # ``_start_download_task``, after the ``DownloadPending`` emit
    # and the ``get_shard_download_status_for_shard`` await window
    # where the race occurs.
    _starting_downloads: set[ModelId] = field(default_factory=set)

    # ``_deleting_in_progress``: cycle-protection set for the
    # delete cascade. ``_reconstruct_drafter_links_for_delete``
    # rebuilds children from the model card on every call, so a
    # self-referential card (``A.drafter_model_ids = [A]``) or a
    # cycle (``A -> B -> A``) would otherwise drive the recursive
    # ``_delete_download`` into infinite recursion until the
    # interpreter's stack limit triggered. Add the current
    # ``model_id`` on entry, remove on exit (in a ``finally`` to
    # survive exceptions in ``delete_model``); the cascade loop
    # skips children already in the set. (Codex P2, PR #18
    # round-(N+13), coordinator.py:337).
    _deleting_in_progress: set[ModelId] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)

    @staticmethod
    def _default_model_dir(model_id: ModelId) -> str:
        return str(EXO_DEFAULT_MODELS_DIR / model_id.normalize())

    def _completed_from_path(
        self,
        shard: ShardMetadata,
        found: Path,
        total: Memory,
    ) -> DownloadCompleted:
        return DownloadCompleted(
            shard_metadata=shard,
            node_id=self.node_id,
            total=total,
            model_directory=str(found),
            read_only=is_read_only_model_dir(found),
        )

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        try:
            if progress.status == "complete":
                found = await to_thread.run_sync(
                    resolve_existing_model, model_id, callback_shard.model_card
                )
                if found is not None:
                    completed = self._completed_from_path(
                        callback_shard, found, progress.total
                    )
                else:
                    completed = DownloadCompleted(
                        shard_metadata=callback_shard,
                        node_id=self.node_id,
                        total=progress.total,
                        model_directory=self._default_model_dir(model_id),
                    )
                self.download_status[model_id] = completed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=completed)
                )
                self._last_progress_time.pop(model_id, None)
            elif (
                progress.status == "in_progress"
                and current_time() - self._last_progress_time.get(model_id, 0.0)
                > throttle_interval_secs
            ):
                ongoing = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=callback_shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = ongoing
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=ongoing)
                )
                self._last_progress_time[model_id] = current_time()
        except (BrokenResourceError, ClosedResourceError):
            logger.debug(
                f"Event stream closed while sending download progress for {model_id}, skipping update"
            )

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        finally:
            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Only process commands targeting this node
                if cmd.command.target_node_id != self.node_id:
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard, available_peers=peers):
                        # Pass peer endpoints to the shard downloader if it supports it
                        if isinstance(self.shard_downloader, PeerAwareShardDownloader):
                            self.shard_downloader.set_available_peers(shard, peers)
                        elif isinstance(
                            self.shard_downloader, SingletonShardDownloader
                        ) and isinstance(
                            self.shard_downloader.shard_downloader,
                            PeerAwareShardDownloader,
                        ):
                            self.shard_downloader.shard_downloader.set_available_peers(
                                shard, peers
                            )
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id):
                        await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            current_status = self.download_status[model_id]
            downloaded = Memory()
            total = Memory()
            if isinstance(current_status, DownloadOngoing):
                downloaded = current_status.download_progress.downloaded
                total = current_status.download_progress.total
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
                downloaded=downloaded,
                total=total,
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
        # Codex flagged (P2, PR #18 round 2) that cancelling a target
        # left chained drafters running in the background, consuming
        # network/disk after the user revoked the original download
        # intent. Pop the parent->children mapping (so we don't
        # double-cancel on a follow-up cancel of the same target) and
        # cascade the cancel.
        #
        # Codex P1 (PR #18 round-(N+3), coordinator.py:212): cascade
        # MUST recurse unconditionally, NOT only for children already
        # in ``active_downloads``. Children registered by
        # ``_maybe_chain_drafter_download`` (via ``remember_drafter_link``)
        # are tracked BEFORE ``await self._start_download(...)`` populates
        # ``active_downloads``. Pre-fix, a cancel that arrived during
        # that prep window skipped the child here -- the cascade saw
        # nothing to cancel -- and the chain's own ``cancelled()``
        # check upstream in the loop only fires *between* iterations,
        # not for the drafter that's mid-``_start_download``. So the
        # drafter download silently continued. The post-await re-check
        # in ``_maybe_chain_drafter_download`` is the live safety net,
        # but recursing unconditionally here keeps the cascade
        # symmetric with future state extensions and ensures the
        # cancel intent reaches every registered child.
        #
        # Codex P1 (PR #18 round-(N+11), coordinator.py:212): when
        # a drafter is shared across multiple targets (e.g. Gemma 4
        # 26B and 31B both name the same e2b/e4b drafters), only
        # cascade the cancel to the drafter once NO target still
        # references it. Pre-fix the cascade tore down a drafter the
        # other still-installed target depended on, silently
        # disabling speculative decoding on that target with no
        # signal back to the user beyond a tokens/sec regression.
        children = self._drafter_children.pop(model_id, [])
        for child_model_id in children:
            parents = self._drafter_parents.get(child_model_id)
            if parents is None:
                # Drafter may have been cancelled or deleted directly;
                # mapping was cleared. Recurse to be defensive.
                logger.info(
                    f"Cascading cancel to chained drafter {child_model_id} "
                    f"alongside target {model_id} (no parent map)"
                )
                await self._cancel_download(child_model_id)
                continue
            parents.discard(model_id)
            if parents:
                logger.info(
                    f"Drafter {child_model_id} is still referenced by "
                    f"{sorted(map(str, parents))}; skipping cancel "
                    f"cascade for it (parent cancel was for {model_id})"
                )
                continue
            # Last reference: clean up the empty parent set and cascade.
            self._drafter_parents.pop(child_model_id, None)
            logger.info(
                f"Cascading cancel to chained drafter {child_model_id} "
                f"alongside target {model_id} (last referencing target)"
            )
            await self._cancel_download(child_model_id)

    async def _start_download(
        self, shard: ShardMetadata, *, is_drafter_chain: bool = False
    ) -> None:
        """Start (or restart) a download.

        Args:
            shard: The shard to download.
            is_drafter_chain: ``True`` when this call originates from
                ``_maybe_chain_drafter_download`` for a drafter
                companion. Drafter chains are allowed to retry past
                a ``DownloadFailed`` status because the user
                reissuing ``StartDownload`` for the target is the
                supported retry trigger -- without this flag the
                ``DownloadFailed`` short-circuit below would block
                drafter retries forever (Codex P1, PR #18
                round-(N+9), coordinator.py:267). Top-level (target)
                calls keep the old behaviour: if the target itself
                previously failed, do not silently kick off a
                drafter download for a non-runnable model.
        """
        model_id = shard.model_card.model_id

        # Check if already downloading, complete, or recently failed
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadCompleted)):
                logger.debug(
                    f"Download for {model_id} already in progress or complete, skipping"
                )
                # Codex P2 (PR #18 round-(N+13), coordinator.py:337):
                # only chain drafters here when the target is
                # ``DownloadCompleted`` (target weights are already
                # on disk and runnable). Pre-fix the branch also
                # chained on ``DownloadOngoing``, so a re-issued
                # ``StartDownload`` during an in-flight target
                # download spawned drafters BEFORE the target's
                # ``ensure_shard()`` had succeeded -- defeating the
                # round-(N+12) success-gated path in
                # ``_start_download_task``. The in-flight target's
                # own ``download_wrapper`` will spawn the chain on
                # the success arm, so duplicating the spawn here
                # is both wasteful (re-enters the chain) and
                # incorrect (chain runs before target success when
                # target is still ``DownloadOngoing``).
                #
                # Drafter chain calls don't recurse into another chain
                # spawn here -- they're already inside one.
                if not is_drafter_chain and isinstance(status, DownloadCompleted):
                    self._spawn_drafter_chain(shard)
                return
            if isinstance(status, DownloadFailed):
                # Codex P2 (PR #18 round-(N+2), coordinator.py:231): the
                # round-(N+1) "backfill drafters even when target was
                # already tracked" branch swept failed targets into the
                # same fast-path, kicking off drafter downloads for a
                # target that won't itself download. Drafters served by
                # a non-runnable target are useless (the runner can't
                # boot speculative decoding without the target weights),
                # so consume the network/disk only when the target is
                # at least possibly going to be runnable.
                #
                # Codex P1 (PR #18 round-(N+9), coordinator.py:267):
                # this short-circuit must NOT apply to drafter
                # chains. Pre-fix the branch blocked all retries
                # through ``_start_download``, including the
                # drafter-chain path -- so a transient drafter
                # failure (network/HF) stayed permanent until manual
                # intervention even when the user reissued
                # ``StartDownload`` for the target. The supported
                # retry trigger is exactly that re-issue, so let
                # drafter chains fall through to the launch flow.
                if not is_drafter_chain:
                    logger.debug(
                        f"Download for {model_id} previously failed; "
                        f"skipping drafter chain (drafter is useless "
                        f"without target)"
                    )
                    return
                logger.info(
                    f"Drafter chain retry for previously-failed "
                    f"{model_id}: target was reissued so retry the "
                    f"drafter to resume speculative decoding"
                )

        # Codex P2 (PR #18 round-(N+3), coordinator.py:224): per-model
        # in-flight gate. We can't use ``download_status`` alone because
        # ``DownloadPending`` is also the state that ``_cancel_download``
        # leaves behind, so a follow-up ``StartDownload`` for the same
        # drafter MUST still re-launch the download (restart-after-cancel
        # is a supported flow). And we can't use ``active_downloads``
        # alone because it's only populated late in
        # ``_start_download_task``, AFTER the ``DownloadPending`` emit
        # and the ``get_shard_download_status_for_shard`` await window
        # where overlapping chain coroutines would otherwise both fall
        # through and call ``ensure_shard()`` -- which then cancels
        # itself and restarts in a flap. ``_starting_downloads`` is the
        # ephemeral marker that bridges that window: present strictly
        # while one ``_start_download`` is mid-launch for ``model_id``,
        # cleared in ``finally`` so a real cancel/failure doesn't leave
        # a stale lock.
        if model_id in self._starting_downloads:
            logger.debug(
                f"Download for {model_id} already in launch flow; "
                f"skipping duplicate start to avoid ensure_shard flap"
            )
            return
        self._starting_downloads.add(model_id)
        try:
            await self._start_download_inner(shard, is_drafter_chain=is_drafter_chain)
        finally:
            self._starting_downloads.discard(model_id)

    async def _start_download_inner(
        self, shard: ShardMetadata, *, is_drafter_chain: bool = False
    ) -> None:
        # Codex P2 (PR #18 round-(N+10), coordinator.py:347): thread
        # ``is_drafter_chain`` through ``_start_download_inner`` so the
        # ``_spawn_drafter_chain`` calls below remain gated when the
        # drafter is being downloaded as part of an already-active
        # chain. Pre-fix the flag was dropped at the inner-call
        # boundary, so a chained drafter that itself declares
        # ``drafter_model_ids`` (custom or accidentally self-
        # referential cards) would recursively re-chain another
        # drafter download whenever its inner path completed,
        # spawning unintended nested background fetches.
        model_id = shard.model_card.model_id

        # Check all model directories for pre-existing complete models
        found_path = await to_thread.run_sync(
            resolve_existing_model, model_id, shard.model_card
        )
        if found_path is not None:
            logger.info(f"DownloadCoordinator: Model {model_id} found at {found_path}")
            completed = self._completed_from_path(
                shard, found_path, shard.model_card.storage_size
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            if not is_drafter_chain:
                self._spawn_drafter_chain(shard)
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = progress
        await self.event_sender.send(NodeDownloadProgress(download_progress=progress))

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            found = await to_thread.run_sync(
                resolve_existing_model, model_id, shard.model_card
            )
            if found is not None:
                completed = self._completed_from_path(
                    shard, found, initial_progress.total
                )
            else:
                completed = DownloadCompleted(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    total=initial_progress.total,
                    model_directory=self._default_model_dir(model_id),
                )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            if not is_drafter_chain:
                self._spawn_drafter_chain(shard)
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._default_model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return

        # Codex P2 (PR #18 round-(N+12), coordinator.py:487): defer
        # ``_spawn_drafter_chain`` until ``ensure_shard()`` for the
        # target actually succeeds. Pre-fix, the chain was spawned
        # immediately after queuing the target download; if the
        # target subsequently failed (auth/rate-limit/transient
        # network/gated repo), the drafter downloads kept running
        # to completion and consumed bandwidth + disk for a model
        # that could never boot. ``download_wrapper`` (inside
        # ``_start_download_task``) now invokes the chain on the
        # success arm of ``ensure_shard()`` so drafters are only
        # fetched when the target is actually runnable. The earlier
        # already-cached / initial-progress-complete arms above
        # still call ``_spawn_drafter_chain`` directly because
        # those paths don't touch ``ensure_shard()`` at all -- the
        # target is already a runnable model on disk.
        self._start_download_task(
            shard, initial_progress, is_drafter_chain=is_drafter_chain
        )

    def _spawn_drafter_chain(self, target_shard: ShardMetadata) -> None:
        """Background the drafter chain so command processing doesn't block.

        Codex flagged (P1, PR #18 round 2) that
        ``_maybe_chain_drafter_download`` ran inline during
        ``StartDownload`` handling. ``ModelCard.load`` falls through
        to ``ModelCard.fetch_from_hf`` whenever the drafter card
        isn't already in ``_card_cache``, and a slow/unreachable HF
        fetch would block the command loop and delay unrelated
        ``CancelDownload``/``DeleteDownload`` commands until the
        client timeout. That turns a best-effort drafter step into
        control-plane backpressure whenever drafter metadata is cold.

        Fix: dispatch the chain on the coordinator's own task group
        via ``start_soon`` so the command processor returns
        immediately and remains responsive. Errors inside the chain
        are still logged-and-swallowed (best-effort semantics
        preserved); the only difference is that they no longer hold
        up unrelated commands.

        Codex P1 (PR #18 round-(N+1)): pre-register an empty
        ``_drafter_children`` entry synchronously here, BEFORE the
        async chain runs. Without this, a ``CancelDownload``
        arriving between ``_spawn_drafter_chain`` returning and
        the chain coroutine populating its child list cancelled the
        target but found no children to cascade into; the chain
        then merrily started drafter downloads in the background
        for a target the user just revoked. With pre-registration
        plus incremental appends inside the chain (and a
        membership re-check after every ``await``), the cancel
        cascade either pops the partial list (cancelling already-
        started drafters) or signals the in-flight chain to bail
        out before starting any further drafters.
        """
        self._drafter_children.setdefault(target_shard.model_card.model_id, [])
        self._tg.start_soon(self._maybe_chain_drafter_download, target_shard)

    async def _maybe_chain_drafter_download(self, target_shard: ShardMetadata) -> None:
        """Enqueue downloads for every drafter declared on ``target_shard``'s
        model card.

        We download *all* candidate drafters so the runner can switch between
        them at startup time via ``EXO_DRAFTER_PREFERENCE`` without an
        on-demand fetch. Drafters are small (typically <2GB) so the storage
        overhead is fine.

        Drafter downloads are silent best-effort: anything that fails (no
        cards, env opt-out, HF unreachable, drafter already tracked) is
        logged and swallowed. The target download is the source of truth for
        the user's intent; speculative decoding is best-effort.

        Each drafter is downloaded as a single ``PipelineShardMetadata`` for
        the entire model. Speculative decoding is single-device today (see
        ``mlx_generate``), so we never need a sharded drafter.


        Cancellation contract (Codex P1 PR #18 round-(N+1)): the parent
        ``_drafter_children[target_id]`` entry is the cancellation
        signal. ``_spawn_drafter_chain`` pre-creates an empty list so
        the cancel cascade in ``_cancel_download`` always finds the
        parent. We pop-on-not-found in this coroutine to detect a
        cancel that arrived between scheduling and entry, and we
        re-check after every ``await`` (model card load and
        ``_start_download`` itself can yield) to avoid starting new
        drafter downloads after the user revoked the parent intent.
        Each drafter is appended to the parent's list BEFORE the
        ``_start_download`` await so a concurrent cancel pops a list
        that includes this drafter and cascades into it correctly.
        """
        target_model_id = target_shard.model_card.model_id

        def cancelled() -> bool:
            return target_model_id not in self._drafter_children

        def discard_chain_signal() -> None:
            # Drop the placeholder when no drafter work will run; we
            # don't want a dangling empty entry leaking into the
            # cancel cascade for any future re-trigger.
            self._drafter_children.pop(target_model_id, None)

        if cancelled():
            logger.debug(
                f"Drafter chain for {target_model_id} aborted before start: "
                f"target was cancelled before chain coroutine ran."
            )
            return

        drafter_ids = list(target_shard.model_card.drafter_model_ids)
        if not drafter_ids:
            discard_chain_signal()
            return
        if _drafter_disabled_by_env():
            logger.debug(
                f"EXO_DISABLE_DRAFTER set; skipping drafter downloads "
                f"{drafter_ids} for {target_model_id}"
            )
            discard_chain_signal()
            return
        if self.offline:
            # Offline mode: ``ModelCard.load`` falls through to
            # ``fetch_from_hf`` whenever the drafter card isn't already
            # in ``_card_cache``, which is an outbound HuggingFace
            # request. Drafter downloads are silent best-effort, so the
            # subsequent ``DownloadFailed`` would have been swallowed
            # anyway, but the HF call itself can stall command
            # processing for the full client timeout (the upstream
            # offline guard at ``_start_download`` line 263 only fires
            # *after* this path has already issued the network call).
            # Skip drafter chaining outright in offline mode -- if the
            # operator wants a drafter, they need to ship it locally.
            logger.debug(
                f"Offline mode: skipping drafter card resolution "
                f"{drafter_ids} for {target_model_id}"
            )
            discard_chain_signal()
            return

        # Codex P2 (PR #18 round-(N+2), coordinator.py:442): we MUST
        # keep the same list object across re-chained downloads.
        # Pre-fix this slot was reassigned to a fresh empty list at
        # the start of every chain run, so a chain that had captured
        # the previous list reference (e.g. after the user re-issued
        # ``StartDownload`` for a target already
        # ``DownloadOngoing``/``DownloadCompleted``) would keep
        # appending into the *orphaned* list. The cancel cascade
        # only pops the dict's current value, so those appends became
        # invisible and the corresponding drafter downloads kept
        # running in the background after a cancel.
        #
        # The fix: mutate-in-place. ``setdefault`` (already done by
        # ``_spawn_drafter_chain``) guarantees the key exists, and a
        # cancellation pops it -- so by the time we get here, the
        # list is either:
        #   - empty (first chain run) or
        #   - a shared accumulator across overlapping chain runs.
        # Appending with a dedup guard avoids duplicates while
        # ensuring every drafter id ever started for this target is
        # in the live cancel-cascade list.
        chained = self._drafter_children[target_model_id]

        def remember_drafter_link(drafter_id: ModelId) -> None:
            if drafter_id not in chained:
                chained.append(drafter_id)
            # Codex P1 (PR #18 round-(N+11)): keep the reverse map in
            # sync. ``setdefault`` makes the first observer create the
            # set; subsequent ``.add`` calls are idempotent. This must
            # be invoked unconditionally (not only on first append)
            # because a drafter that is *already* tracked for one
            # target may become referenced by a NEW target via a
            # later chain run -- e.g. user starts gemma-4-26b
            # (drafter linked once), then starts gemma-4-31b which
            # shares the same drafter; without this re-add the second
            # target would not appear in the parent set and a cancel
            # of the first target would tear the drafter down even
            # though the second target still depends on it.
            self._drafter_parents.setdefault(drafter_id, set()).add(target_model_id)

        for drafter_id in drafter_ids:
            if cancelled():
                logger.info(
                    f"Drafter chain for {target_model_id} aborted mid-flight: "
                    f"target was cancelled."
                )
                return

            existing_status = self.download_status.get(drafter_id)
            if isinstance(existing_status, (DownloadOngoing, DownloadCompleted)):
                # Already in flight or already on disk: record the
                # parent->child link so a subsequent target cancel
                # propagates to the live drafter download. Avoids
                # the case where a drafter started by an earlier
                # target stays alive after the user cancels the only
                # target that references it. (We don't check for
                # OTHER targets also referencing this drafter -- if
                # needed, the drafter is small enough that
                # re-downloading it later is cheap, and tracking a
                # many-to-many graph would balloon the coordinator
                # state.)
                remember_drafter_link(drafter_id)
                continue
            # Codex P2 (PR #18 round-(N+2), coordinator.py:437):
            # ``DownloadPending`` (e.g. after the user cancelled the
            # drafter via ``CancelDownload`` cascade) and
            # ``DownloadFailed`` are NOT terminal for re-chains. A
            # subsequent ``StartDownload`` for the same target is a
            # fresh user intent and should bring the drafter back to
            # life. Pre-fix, ``drafter_id in self.download_status``
            # short-circuited regardless of state, so once a drafter
            # was cancelled it never restarted -- speculative
            # decoding silently stayed disabled until the operator
            # manually started each drafter. Falling through to the
            # ``ModelCard.load`` + ``_start_download`` block below
            # restores the drafter on the next chain run.

            # Codex P1 (PR #18, coordinator.py:723): use the cache-
            # only loader so the command-processing coroutine does not
            # block on a Hugging Face round-trip when ``drafter_id``
            # is not on local disk. ``_command_processor`` serves a
            # single coroutine; an HTTP stall here freezes every
            # subsequent ``StartDownload`` / ``DeleteDownload`` /
            # ``CancelDownload`` until the request times out, and in
            # offline / disconnected environments the queue can stay
            # frozen indefinitely. Treating "card not cached locally"
            # (return ``None``) or "disk read failure" (caught
            # exception) as "skip this drafter for now"; a subsequent
            # ``StartDownload`` for the same target after the operator
            # brings the cluster online (or pre-loads the drafter card
            # via the dashboard) will re-attempt the chain.
            try:
                drafter_card = await ModelCard.load_cached_only(drafter_id)
            except Exception as exc:
                logger.warning(
                    f"Could not load drafter card {drafter_id} for "
                    f"{target_model_id} from local cache; skipping "
                    f"drafter download: {exc}"
                )
                continue
            if drafter_card is None:
                logger.warning(
                    f"Drafter card {drafter_id} for {target_model_id} "
                    f"is not cached locally; skipping drafter download. "
                    f"Run with the drafter card pre-loaded to enable "
                    f"speculative decoding for this target."
                )
                continue

            # Re-check after the card-load await: a cancel could have
            # arrived during the cache lookup. Without this re-check
            # we'd kick off ``_start_download`` for a drafter whose
            # parent the user has already cancelled.
            if cancelled():
                logger.info(
                    f"Drafter chain for {target_model_id} aborted after "
                    f"card load for {drafter_id}: target was cancelled."
                )
                return

            drafter_shard = PipelineShardMetadata(
                model_card=drafter_card,
                device_rank=0,
                world_size=1,
                start_layer=0,
                end_layer=drafter_card.n_layers,
                n_layers=drafter_card.n_layers,
            )
            # Append BEFORE the await so a concurrent cancel pops a
            # list that includes this drafter and cascades into it.
            remember_drafter_link(drafter_id)
            logger.info(f"Chaining drafter download {drafter_id} for {target_model_id}")
            # Codex P1 (PR #18 round-(N+9), coordinator.py:267):
            # mark this as a drafter-chain call so a previously
            # failed drafter is retried (the user reissuing
            # ``StartDownload`` for the target is the supported
            # retry trigger). Without this flag the failed-state
            # short-circuit in ``_start_download`` would silently
            # leave speculative decoding off until manual intervention.
            await self._start_download(drafter_shard, is_drafter_chain=True)

            # Codex P1 (PR #18 round-(N+3), coordinator.py:212): close
            # the cancel-cascade race window. The cascade in
            # ``_cancel_download`` recurses into every registered child,
            # but ``_cancel_download`` itself can only honor a cancel if
            # the child has reached ``active_downloads``. If the parent
            # is cancelled while we're awaiting ``_start_download``
            # above, the cascade arrives BEFORE ``_start_download_task``
            # has populated ``active_downloads`` -- the cascade no-ops
            # for this child, then ``_start_download_task`` runs and
            # the drafter download proceeds despite the user revoking
            # the parent. Re-check ``cancelled()`` here and explicitly
            # cancel the now-launched drafter so the user's intent
            # propagates regardless of timing.
            if cancelled():
                logger.info(
                    f"Drafter chain for {target_model_id} aborted after "
                    f"starting {drafter_id}: target was cancelled mid-launch; "
                    f"cancelling drafter to honor cascade."
                )
                await self._cancel_download(drafter_id)
                return

    def _start_download_task(
        self,
        shard: ShardMetadata,
        initial_progress: RepoDownloadProgress,
        *,
        is_drafter_chain: bool = False,
    ) -> None:
        model_id = shard.model_card.model_id

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            target_succeeded = False
            try:
                with cancel_scope:
                    await self.shard_downloader.ensure_shard(shard)
                    target_succeeded = True
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)
            # Codex P2 (PR #18 round-(N+12), coordinator.py:487):
            # only chain drafters once the target download actually
            # succeeded -- skip on failure (DownloadFailed branch
            # above) AND on cancellation (cancel_scope.cancel_called
            # implies the user revoked the intent before we even
            # finished). ``is_drafter_chain`` short-circuits drafter
            # subchains so a drafter being downloaded as part of an
            # already-active chain doesn't spawn its own (already
            # enforced upstream in ``_start_download_inner``, but
            # mirrored here for the post-success entrypoint).
            if (
                target_succeeded
                and not cancel_scope.cancel_called
                and not is_drafter_chain
            ):
                self._spawn_drafter_chain(shard)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _reconstruct_drafter_links_for_delete(
        self, model_id: ModelId
    ) -> list[ModelId]:
        """Pop the existing drafter children for ``model_id`` and merge
        them with the drafter ids declared on its model card.

        The merge handles the post-restart case where
        ``_drafter_children`` is empty (process-local state, not
        rehydrated on startup) but the user is deleting a target that
        had drafters chained in an earlier process. Pre-fix, deleting
        such a target left the drafter weights orphaned on disk and
        the only signal back to the operator was disk usage that
        slowly grew over time.

        Resolution order:

        1. Pop the existing chain entry (preserves the
           "delete-once" semantics of the prior implementation --
           re-deleting the same target after this call is a no-op).
        2. Load the target's model card via ``ModelCard.load`` to
           extract ``drafter_model_ids``. ``ModelCard.load`` reads
           from the on-disk card cache first, so this is cheap when
           the target's model files (including its card) are still
           on disk -- which is the only case where the delete
           cascade is meaningful anyway. ``ModelCard.load`` may
           still fall through to ``fetch_from_hf``; the failure path
           swallows the exception and returns just the in-memory
           list.
        3. Repopulate ``_drafter_parents`` for any rediscovered
           drafter so that other still-referencing targets continue
           to gate this delete cascade on "last reference"
           semantics. Without this step, deleting target A would
           also delete a drafter target B still depends on, even
           when target B's chain in this process had already
           registered its parent link.
        4. Codex P1 (PR #18 round-(N+13), coordinator.py:910): scan
           ALL known model cards (built-in + custom) for *other*
           targets that declare any of these drafters as a chain
           dependency, and add those targets as parents whenever
           the other target is **installed on disk**. Pre-fix the
           rebuild only registered the current ``model_id`` as a
           parent, so a shared drafter whose other parent's chain
           had not run in this process (e.g. the user only ever
           downloaded one of the two targets that share the
           drafter, OR the process restarted before any chain ran)
           was incorrectly treated as orphaned and deleted by the
           cascade -- silently degrading the surviving target back
           to non-speculative behaviour. We restrict the discovered
           parents to *installed* targets so a card declaring
           ``drafter_model_ids = [x]`` for a model that was never
           downloaded does not block legitimate deletion of ``x``;
           the runtime ``_spawn_drafter_chain`` path uses the same
           "only after the parent has actually been downloaded"
           semantic, so this matches it.
        """
        existing = list(self._drafter_children.pop(model_id, []))
        # Codex P1 (PR #18, coordinator.py:908): cache-only load so
        # the delete-cascade does not block on a Hugging Face round-
        # trip when ``model_id``'s card is not on local disk. This
        # path runs from ``_command_processor``, so an HTTP stall
        # would freeze every subsequent download command.
        #
        # ``None`` from :meth:`load_cached_only` means "no card cached
        # locally"; an exception means a disk-read failure during
        # ``_refresh_card_cache``. Both fall back to the in-memory
        # ``_drafter_children`` entries (which captures any links
        # established during this process's lifetime). A post-restart
        # delete of a target whose card is neither cached nor in
        # memory is rare in practice (the target had to have been
        # downloaded to be deletable, and downloading caches the
        # card) and the graceful skip is preferable to blocking the
        # command queue.
        try:
            target_card = await ModelCard.load_cached_only(model_id)
        except Exception as exc:
            logger.debug(
                f"Could not reload card for {model_id} during delete "
                f"cascade rebuild ({exc}); using in-memory drafter "
                f"links only ({len(existing)} entries)"
            )
            return existing
        if target_card is None:
            logger.debug(
                f"Card for {model_id} not in local cache during delete "
                f"cascade rebuild; using in-memory drafter links only "
                f"({len(existing)} entries)"
            )
            return existing

        merged: list[ModelId] = list(existing)
        seen: set[ModelId] = set(existing)
        for drafter_id in target_card.drafter_model_ids:
            if drafter_id in seen:
                continue
            merged.append(drafter_id)
            seen.add(drafter_id)
            # Treat the rediscovered link as if the chain ran in
            # this process so the shared-drafter cascade gate
            # behaves identically to the runtime path. ``setdefault``
            # creates the parent set if it doesn't yet exist; we add
            # the current ``model_id`` so the discard-and-check loop
            # below removes it correctly.
            self._drafter_parents.setdefault(drafter_id, set()).add(model_id)

        if merged:
            await self._discover_other_drafter_parents(
                deleting_model_id=model_id, drafters=merged
            )
        return merged

    async def _discover_other_drafter_parents(
        self,
        *,
        deleting_model_id: ModelId,
        drafters: list[ModelId],
    ) -> None:
        """Codex P1 (PR #18 round-(N+13), coordinator.py:910): rebuild
        the inverse parent->drafter mapping for OTHER installed
        targets that share any drafter in ``drafters``.

        ``_reconstruct_drafter_links_for_delete`` only records the
        currently-deleting target as a parent, so a shared drafter
        whose other parent's chain has not run in this process
        (typical post-restart) would be treated as unreferenced and
        cascaded-deleted alongside the first target's removal --
        breaking speculative decoding for the surviving target. We
        scan every known card and, for each card that declares any
        of these drafters AND whose own model is installed on disk,
        register that card as a parent so the cascade's
        last-reference gate correctly preserves the drafter.

        Implementation notes:
        * We deliberately exclude ``deleting_model_id`` from the
          iteration: ``_reconstruct_drafter_links_for_delete`` has
          already added it as a parent and the cascade loop
          ``parents.discard(model_id)`` will pop it back out when
          the delete proceeds.
        * "Installed on disk" is determined via
          ``resolve_existing_model``, which mirrors the post-restart
          hydration path used by ``_emit_existing_download_progress``.
          This intentionally ignores cards whose models were never
          downloaded -- registering uninstalled cards as parents
          would block legitimate deletes of orphaned drafters that
          no installed target needs.
        * ``get_model_cards`` failures are swallowed: the rebuild
          is best-effort and the runtime parent map (set during
          ``_spawn_drafter_chain``) remains the authoritative
          source whenever it has been populated.
        """
        try:
            all_cards = await get_model_cards()
        except Exception as exc:
            logger.debug(
                f"Could not enumerate model cards while rebuilding "
                f"shared-drafter parents during delete of "
                f"{deleting_model_id} ({exc}); proceeding with the "
                "current parent map. Other installed targets that "
                "share a drafter may have been registered already "
                "via runtime chain-spawn; if not, the cascade may "
                "delete a still-referenced drafter."
            )
            return

        drafter_set = set(drafters)
        for other_card in all_cards:
            other_id = other_card.model_id
            if other_id == deleting_model_id:
                continue
            shared = drafter_set.intersection(other_card.drafter_model_ids)
            if not shared:
                continue
            installed = await to_thread.run_sync(
                resolve_existing_model, other_id, other_card
            )
            if installed is None:
                continue
            for drafter_id in shared:
                parents = self._drafter_parents.setdefault(drafter_id, set())
                if other_id not in parents:
                    parents.add(other_id)
                    logger.debug(
                        f"Registered installed target {other_id} as a "
                        f"parent of shared drafter {drafter_id} so the "
                        f"delete cascade for {deleting_model_id} "
                        f"preserves the drafter on disk."
                    )

    async def _delete_download(self, model_id: ModelId) -> None:
        # Codex P2 (PR #18 round-(N+13), coordinator.py:337): cycle
        # protection. ``_reconstruct_drafter_links_for_delete``
        # rebuilds children from ``ModelCard.load`` on every call,
        # so a self-referential card
        # (``A.drafter_model_ids = [A]``) or a cycle
        # (``A -> B -> A``) would otherwise drive the recursive
        # cascade into infinite recursion until the interpreter's
        # stack limit fired (and aborted the operator's delete
        # mid-cascade rather than performing a safe no-op). When we
        # detect we're already deleting this id earlier on the
        # call stack, skip the recursive call -- the outer
        # invocation will finish the on-disk delete.
        if model_id in self._deleting_in_progress:
            logger.debug(
                f"Skipping recursive delete cascade for {model_id}: "
                f"already in progress earlier on the call stack "
                f"(self-referential or cyclical drafter card)"
            )
            return
        self._deleting_in_progress.add(model_id)
        try:
            await self._delete_download_inner(model_id)
        finally:
            self._deleting_in_progress.discard(model_id)

    async def _delete_download_inner(self, model_id: ModelId) -> None:
        # Protect read-only models from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(f"Refusing to delete read-only model {model_id}")
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Cascade cancellation/deletion to chained drafters: the user
        # is removing the target's download intent, so the drafters
        # spawned alongside it should not keep running or stay on disk
        # past the target's lifetime. Pop the mapping so we don't
        # double-cascade on a subsequent delete of the same target.
        #
        # Codex P1 (PR #18 round-(N+11), coordinator.py:743): when
        # the drafter is shared across multiple targets (Gemma 4 26B
        # and 31B both name e2b/e4b), only delete it once NO other
        # target still references it. Pre-fix deleting one target
        # would also remove the drafter the other still-installed
        # target depended on, silently degrading that target back to
        # non-speculative behaviour and forcing an unnecessary
        # re-download next time the user reissued StartDownload for
        # it.
        #
        # Codex P2 (PR #18 round-(N+12), coordinator.py:817):
        # ``_drafter_children`` is process-local state populated
        # during runtime chaining and not rehydrated on startup.
        # After an exo restart, deleting a target whose drafters
        # were chained in a previous process would find the parent
        # entry empty and leave the drafter weights orphaned on
        # disk. Rebuild the parent->children list from the model
        # card's ``drafter_model_ids`` here so the cascade still
        # works post-restart (and the inverse parent set rebuilds
        # alongside it so other still-referencing targets continue
        # to protect the drafter from premature delete).
        children = await self._reconstruct_drafter_links_for_delete(model_id)
        for child_model_id in children:
            parents = self._drafter_parents.get(child_model_id)
            if parents is not None:
                parents.discard(model_id)
                if parents:
                    logger.info(
                        f"Drafter {child_model_id} is still referenced by "
                        f"{sorted(map(str, parents))}; preserving on disk "
                        f"and in-flight (delete cascade was for {model_id})"
                    )
                    continue
                # Last reference: clean up the empty parent set so
                # the drafter is genuinely orphaned for this delete.
                self._drafter_parents.pop(child_model_id, None)
            # Codex P2 (PR #18 round-(N+13), coordinator.py:945):
            # cascade unconditionally when we reach this point.
            # ``_reconstruct_drafter_links_for_delete`` already
            # populated ``children`` from the target's
            # ``drafter_model_ids``, so the rediscovered IDs are
            # *expected to exist on disk* even when neither
            # ``active_downloads`` nor ``download_status`` knows
            # about them yet (the typical post-restart window
            # before ``_emit_existing_download_progress`` has
            # hydrated). Pre-fix the cascade silently skipped a
            # rediscovered drafter in that window, leaving its
            # weights orphaned on disk -- the very regression the
            # rebuild path was meant to repair.
            # ``_delete_download`` itself is idempotent for missing
            # state: ``delete_model`` reports "not found on disk"
            # via ``deleted == False`` rather than raising, the
            # read-only guard is keyed on ``download_status`` so a
            # cold cache simply skips it, and the post-delete
            # status emit short-circuits when ``download_status``
            # is empty.
            logger.info(
                f"Deleting chained drafter {child_model_id} alongside "
                f"target {model_id} (last referencing target)"
            )
            await self._delete_download(child_model_id)

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        try:
            deleted = await delete_model(model_id)
        except Exception:
            logger.exception(f"Failed to delete model files for {model_id}")
            return

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _emit_existing_download_progress(self) -> None:
        while True:
            try:
                logger.debug(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    model_id = progress.shard.model_card.model_id

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if progress.status == "complete":
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status: DownloadProgress = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        else:
                            status = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                total=progress.total,
                                model_directory=self._default_model_dir(model_id),
                            )
                    elif progress.status in ["in_progress", "not_started"]:
                        # TODO(ciaran): temporary solution
                        # Don't downgrade a model that is already confirmed complete.
                        if isinstance(
                            self.download_status.get(model_id), DownloadCompleted
                        ):
                            continue
                        # The per-file size check compares local files against
                        # the latest HF "main" revision, which is a moving
                        # target.  When HF updates text files (README, YAML,
                        # jinja) in a new commit, the cached file list has new
                        # sizes while local files still match the old revision.
                        # Fall back to the authoritative completeness check
                        # (is_model_directory_complete) which validates that all
                        # safetensors weight files are present.
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        elif progress.downloaded.in_bytes == 0:
                            continue
                        elif progress.downloaded_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._default_model_dir(model_id),
                                downloaded=progress.downloaded,
                                total=progress.total,
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                                model_directory=self._default_model_dir(model_id),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                # Scan read-only directories for pre-downloaded models
                if EXO_MODELS_READ_ONLY_DIRS:
                    for card in await get_model_cards():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = await to_thread.run_sync(
                            resolve_existing_model, mid, card
                        )
                        if found is not None and is_read_only_model_dir(found):
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = (
                                self._completed_from_path(
                                    path_shard, found, card.storage_size
                                )
                            )
                            self.download_status[mid] = path_completed
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=path_completed)
                            )

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
