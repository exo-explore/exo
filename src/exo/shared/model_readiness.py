"""Whether a model can serve a request now, and if not, what it is doing.

A request for a model whose instance isn't ready used to be sent anyway and wait, with no
error, until the instance was ready: for the whole download (minutes, or forever if the
download never progressed) or until the instance was deleted. A request now waits for the
model for a bounded time, EXO_MODEL_READY_WAIT_SECONDS, then gets 503 Service Unavailable
with what the model is doing and a Retry-After, as OpenAI-compatible servers do while a model
loads. One whose model failed to start gets its 503 at once.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import final

from exo.shared.request_errors import MODEL_NOT_READY, user_facing_detail
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
)
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)

# How long a request waits for its model to be ready before it gets a 503. Loading and warming
# up usually take seconds; a download takes minutes. 30 s answers within the 60 s timeout many
# clients use. A client that would rather wait for a whole download (the OpenAI SDKs wait up to
# 600 s by default) can be served by raising it.
MODEL_READY_WAIT_SECONDS = float(os.getenv("EXO_MODEL_READY_WAIT_SECONDS", "30"))
# Retry-After for a model that is loading or starting, and for one whose runners failed (the
# worker starts them again after a few seconds)
_RETRY_LOADING_SECONDS = 5
_RETRY_FAILED_SECONDS = 10
# Bounds on Retry-After for a model that is downloading. Clients (the OpenAI SDKs, for one)
# ignore a Retry-After over a minute and pick their own delay.
_RETRY_DOWNLOAD_MIN_SECONDS = 5
_RETRY_DOWNLOAD_MAX_SECONDS = 60


@final
@dataclass(frozen=True)
class NotReady:
    """Why no instance of a model can serve a request yet"""

    # What the model is doing, said to the client: "downloading (42%, about 3 min left)"
    status: str
    retry_after_seconds: int
    # Whether it failed to start, so that waiting for it is pointless
    failed: bool = False

    @property
    def message(self) -> str:
        return (
            f"{MODEL_NOT_READY}{self.status}. "
            f"Try again in {self.retry_after_seconds} s."
        )


def instance_is_ready(
    instance: Instance, runners: Mapping[RunnerId, RunnerStatus]
) -> bool:
    """Whether every runner of the instance is ready to serve, or serving"""
    return all(
        isinstance(runners.get(runner_id), (RunnerReady, RunnerRunning))
        for runner_id in instance.shard_assignments.runner_to_shard
    )


def model_readiness(state: State, model_id: ModelId) -> NotReady | None:
    """None if an instance of the model is ready to serve, else what the most advanced of its
    instances is doing. The model must have an instance."""
    statuses = [
        instance_readiness(state, instance)
        for instance in state.instances.values()
        if instance.shard_assignments.model_id == model_id
    ]
    if not statuses or any(status is None for status in statuses):
        return None
    not_ready = [status for status in statuses if status is not None]
    # Loading beats downloading beats failed
    return min(
        not_ready, key=lambda status: (status.failed, status.retry_after_seconds)
    )


def instance_readiness(state: State, instance: Instance) -> NotReady | None:
    """None if the instance is ready to serve, else what it is doing"""
    if instance_is_ready(instance, state.runners):
        return None
    runners = [
        state.runners.get(runner_id)
        for runner_id in instance.shard_assignments.runner_to_shard
    ]
    if any(isinstance(runner, RunnerFailed) for runner in runners):
        return NotReady(
            status="it failed to start, and is being started again",
            retry_after_seconds=_RETRY_FAILED_SECONDS,
            failed=True,
        )
    model_id = instance.shard_assignments.model_id
    nodes = list(instance.shard_assignments.node_to_runner)
    if (failed := _download_failure(state, model_id, nodes)) is not None:
        return NotReady(
            status=f"its download failed: {user_facing_detail(failed).rstrip('.')}",
            retry_after_seconds=_RETRY_FAILED_SECONDS,
            failed=True,
        )
    if (downloading := _download_progress(state, model_id, nodes)) is not None:
        return downloading
    loading = [runner for runner in runners if isinstance(runner, RunnerLoading)]
    if loading:
        layers_loaded = sum(runner.layers_loaded for runner in loading)
        total_layers = sum(runner.total_layers for runner in loading)
        status = (
            f"loading ({layers_loaded} of {total_layers} layers)"
            if total_layers > 0
            else "loading"
        )
    elif any(isinstance(runner, RunnerWarmingUp) for runner in runners):
        status = "warming up"
    else:
        status = "starting"
    return NotReady(status=status, retry_after_seconds=_RETRY_LOADING_SECONDS)


def _download_failure(
    state: State, model_id: ModelId, nodes: list[NodeId]
) -> str | None:
    for node_id in nodes:
        for download in state.downloads.get(node_id, ()):
            if (
                isinstance(download, DownloadFailed)
                and download.shard_metadata.model_card.model_id == model_id
            ):
                return download.error_message
    return None


def _download_progress(
    state: State, model_id: ModelId, nodes: list[NodeId]
) -> NotReady | None:
    """The progress of the model's download on the instance's nodes, if any is downloading"""
    downloaded = total = 0
    eta_ms = 0
    downloading = started = False
    for node_id in nodes:
        for download in state.downloads.get(node_id, ()):
            if download.shard_metadata.model_card.model_id != model_id:
                continue
            match download:
                case DownloadOngoing(download_progress=progress):
                    downloading = started = True
                    downloaded += progress.downloaded.in_bytes
                    total += progress.total.in_bytes
                    eta_ms = max(eta_ms, progress.eta_ms)
                case DownloadPending():
                    downloading = True
                    downloaded += download.downloaded.in_bytes
                    total += download.total.in_bytes
                case DownloadCompleted():
                    downloaded += download.total.in_bytes
                    total += download.total.in_bytes
                case DownloadFailed():
                    pass
    if not downloading:
        return None
    if not started:
        # Pending: it hasn't fetched the model's file list yet (Hugging Face may be slow or
        # unreachable)
        return NotReady(
            status="waiting for its download to start",
            retry_after_seconds=_RETRY_DOWNLOAD_MIN_SECONDS,
        )
    details: list[str] = []
    if total > 0:
        details.append(f"{100 * downloaded / total:.0f}%")
    if eta_ms > 0:
        details.append(f"about {_duration(eta_ms / 1000)} left")
    eta_seconds = round(eta_ms / 1000)
    return NotReady(
        status=f"downloading ({', '.join(details)})" if details else "downloading",
        retry_after_seconds=min(
            max(eta_seconds, _RETRY_DOWNLOAD_MIN_SECONDS), _RETRY_DOWNLOAD_MAX_SECONDS
        ),
    )


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(round(seconds), 1)} s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"
