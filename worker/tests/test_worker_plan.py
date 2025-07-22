from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, List, Optional, Type

import pytest

from shared.types.common import NodeId
from shared.types.models import ModelId
from shared.types.state import State

# WorkerState import below after RunnerCase definition to avoid forward reference issues
from shared.types.worker.common import InstanceId, NodeStatus, RunnerId
from shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from shared.types.worker.instances import Instance, InstanceParams, TypeOfInstance
from shared.types.worker.ops import DownloadOp
from shared.types.worker.runners import (
    DownloadingRunnerStatus,
    ReadyRunnerStatus,
    RunnerStatus,
    ShardAssignments,
)
from shared.types.worker.shards import PipelineShardMetadata
from worker.download.download_utils import build_model_path
from worker.main import AssignedRunner, Worker


@dataclass(slots=True, frozen=True)
class RunnerCase:
    """Important, minimal state for a *single* runner relevant to planning."""

    status: RunnerStatus
    downloaded: bool  # Does the model shard already exist on disk?


@dataclass(slots=True, frozen=True)
class PlanTestCase:
    """Table-driven description of an entire planning scenario."""

    description: str
    runners: List[RunnerCase]
    # If we expect an op, specify the precise type and the index of the runner it targets.
    expected_op_type: Optional[Type[DownloadOp]]  # Currently only DownloadOp handled.
    expected_op_runner_idx: Optional[int] = None
    # Allow overriding the WorkerState passed to Worker.plan.  When None, a default state
    # is constructed from `runners` via helper `_build_worker_state`.
    worker_state_override: Optional[State] = None

    def id(self) -> str:  # noqa: D401
        return self.description.replace(" ", "_")


def _make_downloading_status(node_id: NodeId) -> DownloadingRunnerStatus:
    """Factory for a *Downloading* status with placeholder progress."""
    return DownloadingRunnerStatus(
        download_progress=DownloadOngoing(
            node_id=node_id,
            download_progress=DownloadProgressData(total_bytes=1, downloaded_bytes=0),
        )
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

TEST_CASES: Final[List[PlanTestCase]] = [
    PlanTestCase(
        description="no runners ⇢ no-op",
        runners=[],
        expected_op_type=None,
        expected_op_runner_idx=None,
    ),
    PlanTestCase(
        description="single ready runner, model missing ⇢ expect DownloadOp",
        runners=[
            RunnerCase(status=ReadyRunnerStatus(), downloaded=False),
        ],
        expected_op_type=DownloadOp,
        expected_op_runner_idx=0,
    ),
    PlanTestCase(
        description="runner already downloading ⇢ no-op",
        runners=[
            RunnerCase(status=_make_downloading_status(NodeId()), downloaded=False),
        ],
        expected_op_type=None,
        expected_op_runner_idx=None,
    ),
    PlanTestCase(
        description="ready runner, model present ⇢ no-op",
        runners=[
            RunnerCase(status=ReadyRunnerStatus(), downloaded=True),
        ],
        expected_op_type=None,
        expected_op_runner_idx=None,
    ),
    PlanTestCase(
        description="instance for other node ⇢ no-op",
        runners=[
            RunnerCase(status=ReadyRunnerStatus(), downloaded=False),
        ],
        expected_op_type=None,
        expected_op_runner_idx=None,
        worker_state_override=State(
            node_status={NodeId(): NodeStatus.Idle},
            instances={},
        ),
    ),
]


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunnerContext:
    runner_id: RunnerId
    instance_id: InstanceId
    shard_metadata: PipelineShardMetadata
    instance_params: InstanceParams


# TODO: generalize this it's in conftest.
def _build_worker_state(
    *,
    tmp_path: Path,
    node_id: NodeId,
    pipeline_shard_metadata: PipelineShardMetadata,
    runner_cases: List[RunnerCase],
) -> tuple[State, List[RunnerContext]]:
    """Construct a WorkerState plus per-runner context objects."""

    instances: dict[InstanceId, Instance] = {}
    runner_contexts: list[RunnerContext] = []

    for idx, _ in enumerate(runner_cases):
        runner_id = RunnerId()
        instance_id = InstanceId()
        model_id = ModelId()

        # Unique sub-directory per runner to allow selective `downloaded` mocking.
        model_subdir = tmp_path / f"runner_{idx}"
        model_subdir.mkdir(exist_ok=True)

        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: pipeline_shard_metadata},
            node_to_runner={node_id: runner_id},
        )

        instance_params = InstanceParams(
            shard_assignments=shard_assignments,
            hosts=[],
        )

        instance = Instance(
            instance_id=instance_id,
            instance_params=instance_params,
            instance_type=TypeOfInstance.ACTIVE,
        )

        instances[instance_id] = instance

        runner_contexts.append(
            RunnerContext(
                runner_id=runner_id,
                instance_id=instance_id,
                shard_metadata=pipeline_shard_metadata,
                instance_params=instance_params,
            )
        )

    worker_state = State(
        node_status={node_id: NodeStatus.Idle},
        instances=instances,
    )

    return worker_state, runner_contexts


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


# Pre-compute readable identifiers for each case to avoid lambda typing issues.
@pytest.mark.parametrize("case", TEST_CASES, ids=[case.id() for case in TEST_CASES])
def test_worker_plan(case: PlanTestCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline_shard_meta: Callable[..., PipelineShardMetadata]) -> None:
    """Exercise Worker.plan across declarative scenarios."""

    # Fresh identifier for isolation of node
    node_id = NodeId()

    # Assemble WorkerState and surrounding objects ---------------------------------------
    worker_state, runner_contexts = _build_worker_state(
        tmp_path=tmp_path,
        node_id=node_id,
        pipeline_shard_metadata=pipeline_shard_meta(1, 0),
        runner_cases=case.runners,
    )

    # Replace with explicit override if provided by the scenario.
    if case.worker_state_override is not None:
        worker_state = case.worker_state_override

    logger = logging.getLogger("test_worker_plan")
    worker = Worker(node_id=node_id, initial_state=worker_state, logger=logger)

    # Build assigned_runners and a path→downloaded lookup --------------------------------
    path_downloaded_map: dict[str, bool] = {}

    for idx, runner_case in enumerate(case.runners):
        runner_status = runner_case.status
        ctx = runner_contexts[idx]

        assigned_runner = AssignedRunner(
            runner_id=ctx.runner_id,
            instance_id=ctx.instance_id,
            shard_metadata=ctx.shard_metadata,
            hosts=ctx.instance_params.hosts,
            status=runner_status,
            runner=None,
        )
        worker.assigned_runners[ctx.runner_id] = assigned_runner

        path_downloaded_map[str(build_model_path(ctx.shard_metadata.model_meta.model_id))] = runner_case.downloaded

    # Stub filesystem existence check ------------------------------------------------------
    from worker import main as worker_main  # local import for module-scoped os

    def _fake_exists(path: str | Path) -> bool:  # noqa: ANN001  – match os.path.exists signature
        return path_downloaded_map.get(str(path), False)

    monkeypatch.setattr(worker_main.os.path, "exists", _fake_exists)

    # Plan and assert ----------------------------------------------------------------------
    op = worker.plan(worker_state)

    if case.expected_op_type is None:
        assert op is None, f"Unexpected op {op} for scenario: {case.description}"
    else:
        assert isinstance(op, case.expected_op_type), (
            f"Expected {case.expected_op_type.__name__}, got {type(op).__name__ if op else 'None'}"
        )

        assert case.expected_op_runner_idx is not None, "Runner index must be set when expecting an op"
        target_ctx = runner_contexts[case.expected_op_runner_idx]

        assert op.runner_id == target_ctx.runner_id
        assert op.instance_id == target_ctx.instance_id
        assert op.shard_metadata == target_ctx.shard_metadata
