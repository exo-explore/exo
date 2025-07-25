from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from shared.types.api import ChatCompletionMessage
from shared.types.state import State
from shared.types.tasks import (
    ChatCompletionTask,
    ChatCompletionTaskParams,
    TaskStatus,
    TaskType,
)
from shared.types.worker.common import NodeStatus
from shared.types.worker.downloads import DownloadPending
from shared.types.worker.instances import Instance, InstanceStatus
from shared.types.worker.ops import (
    AssignRunnerOp,
    DownloadOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from shared.types.worker.runners import (
    AssignedRunnerStatus,
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    LoadedRunnerStatus,
    ReadyRunnerStatus,
    RunningRunnerStatus,
    ShardAssignments,
)
from shared.types.worker.shards import PipelineShardMetadata
from worker.download.download_utils import build_model_path
from worker.main import Worker

from .test_worker_plan_utils import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
    InProcessRunner,
    OverrideAssignedRunner,
    PlanTestCase,
    make_downloading_status,
    make_model_meta,
    make_shard_metadata,
)

"""
The idea with these tests is to define declaratively the input and expected output of the worker.plan function.

We initialize a Worker with InProcessRunners. We then construct a State which gets passed to Worker.plan.
We then check what operation is returned by Worker.plan.
"""

def _get_test_cases(tmp_path: Path) -> list[PlanTestCase]:
    # The `model_path` for `RUNNER_1_ID` must exist for the `DownloadOp` test case to pass validation.
    (tmp_path / f"model_for_runner_{RUNNER_1_ID}").mkdir(exist_ok=True, parents=True)
    model_a_meta = make_model_meta(MODEL_A_ID)
    return [
        PlanTestCase(
            description="no runners -> no-op",
            in_process_runners=[],
            state=State(node_status={NODE_A: NodeStatus.Idle}, instances={}, runners={}),
            expected_op=None,
        ),

        # I don't think this should ever happen, as if it's currently downloading then the worker loop will be blocked
        # Potentially useful for future compatibility when worker becomes non-blocking
        PlanTestCase(
            description="runner state assigned, runner is assigned and downloading -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=make_downloading_status(NODE_A),
                    downloaded=False,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.INACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: make_downloading_status(NODE_A)},
            ),
            expected_op=None,
        ),

        PlanTestCase(
            description="runner state downloading, runner is downloading -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=make_downloading_status(NODE_A),
                    downloaded=False,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.INACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: make_downloading_status(NODE_A)},
            ),
            expected_op=None,
        ),

        PlanTestCase(
            description="ready runner, model present -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.INACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: ReadyRunnerStatus()},
            ),
            expected_op=None,
        ),

        PlanTestCase(
            description="runner assigned and not in state -> AssignRunnerOp",
            in_process_runners=[],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE, # Either active or inactive should yield the same.
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: AssignedRunnerStatus()},
            ),
            expected_op=AssignRunnerOp(
                instance_id=INSTANCE_1_ID,
                runner_id=RUNNER_1_ID,
                shard_metadata=PipelineShardMetadata(
                    device_rank=0,
                    world_size=1,
                    model_meta=model_a_meta,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                ),
                hosts=[]
            ),
        ),

        PlanTestCase(
            description="runner assigned but no longer in state -> UnassignRunnerOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=AssignedRunnerStatus(),
                    downloaded=False,
                )
            ],
            state=State(node_status={NODE_A: NodeStatus.Idle}, instances={}, runners={}),
            expected_op=UnassignRunnerOp(runner_id=RUNNER_1_ID),
        ),

        PlanTestCase(
            description="runner state assigned, runner is assigned, not downloaded -> expect DownloadOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=AssignedRunnerStatus(),
                    downloaded=False,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: AssignedRunnerStatus()},
            ),
            expected_op=DownloadOp(
                runner_id=RUNNER_1_ID,
                instance_id=INSTANCE_1_ID,
                shard_metadata=PipelineShardMetadata(
                    device_rank=0,
                    world_size=1,
                    model_meta=model_a_meta,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                ),
                hosts=[],
            ),
        ),

        PlanTestCase(
            description="ready runner (and state up) -> expect RunnerUpOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: ReadyRunnerStatus()},
                tasks={},
            ),
            expected_op=RunnerUpOp(runner_id=RUNNER_1_ID),
        ),

        PlanTestCase(
            description="1 ready, 1 downloading (and state up) -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=DownloadingRunnerStatus(
                        download_progress=DownloadPending(node_id=NODE_A)
                    ),
                    downloaded=False,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: ReadyRunnerStatus(), RUNNER_2_ID: DownloadingRunnerStatus(download_progress=DownloadPending(node_id=NODE_A))},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=None
        ),

        PlanTestCase(
            description="2 ready runners (and state up) -> expect RunnerUpOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: ReadyRunnerStatus(), RUNNER_2_ID: ReadyRunnerStatus()},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=RunnerUpOp(runner_id=RUNNER_1_ID)
        ),

        PlanTestCase(
            description="loaded runner (and state down) -> expect RunnerDownOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.INACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus()},
                tasks={},
            ),
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),

        PlanTestCase(
            description="failed runner (and state down) -> expect RunnerDownOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=FailedRunnerStatus(),
                    downloaded=True,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.INACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: FailedRunnerStatus()},
                tasks={},
            ),
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),

        PlanTestCase(
            description="loaded runner, model present, task pending -> expect ExecuteTaskOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus()},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=ExecuteTaskOp(runner_id=RUNNER_1_ID, task=ChatCompletionTask(
                task_id=TASK_1_ID,
                instance_id=INSTANCE_1_ID,
                task_type=TaskType.CHAT_COMPLETION,
                task_status=TaskStatus.PENDING,
                task_params=ChatCompletionTaskParams(
                    model=str(MODEL_A_ID),
                    messages=[ChatCompletionMessage(role="user", content="Hello, world!")]
                ),
            )),
        ),

        PlanTestCase(
            # We should only run rank 0 once all other ranks are running.
            description="two loaded runners & task, i'm rank 0 -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus(), RUNNER_2_ID: LoadedRunnerStatus()},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=None
        ),

        PlanTestCase(
            description="two loaded runners & task, i'm rank 1 -> expect ExecuteTaskOp on rank 1",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=1, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=0, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus(), RUNNER_2_ID: LoadedRunnerStatus()},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=ExecuteTaskOp(
                runner_id=RUNNER_1_ID,
                task=ChatCompletionTask(
                    task_id=TASK_1_ID,
                    instance_id=INSTANCE_1_ID,
                    task_type=TaskType.CHAT_COMPLETION,
                    task_params=ChatCompletionTaskParams(
                        model=str(MODEL_A_ID),
                        messages=[ChatCompletionMessage(role="user", content="Hello, world!")],
                    ),
                    task_status=TaskStatus.PENDING,
                ),
            ),
        ),

        PlanTestCase(
            description="rank 1 loaded, rank 0 ready, i'm rank 0  -> expect ExecuteTaskOp on rank 0",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=RunningRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Running},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus(), RUNNER_2_ID: RunningRunnerStatus()},
                tasks={TASK_1_ID: ChatCompletionTask(task_id=TASK_1_ID, task_type=TaskType.CHAT_COMPLETION, task_status=TaskStatus.PENDING, task_params=ChatCompletionTaskParams(model=str(MODEL_A_ID), messages=[ChatCompletionMessage(role="user", content="Hello, world!")]), instance_id=INSTANCE_1_ID)},
            ),
            expected_op=ExecuteTaskOp(
                runner_id=RUNNER_1_ID,
                task=ChatCompletionTask(
                    task_id=TASK_1_ID,
                    instance_id=INSTANCE_1_ID,
                    task_type=TaskType.CHAT_COMPLETION,
                    task_params=ChatCompletionTaskParams(
                        model=str(MODEL_A_ID),
                        messages=[ChatCompletionMessage(role="user", content="Hello, world!")],
                    ),
                    task_status=TaskStatus.PENDING,
                ),
            ),
        ),

        PlanTestCase(
            description="other runner failed -> RunnerDownOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=FailedRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: LoadedRunnerStatus(), RUNNER_2_ID: FailedRunnerStatus()},
            ),
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID)
        ),

        PlanTestCase(
            description="this runner failed (1 node) -> RunnerDownOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=FailedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=1),
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: FailedRunnerStatus()},
            ),
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID)
        ),


        PlanTestCase(
            description="this runner failed (2 nodes) -> no-op",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=FailedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=LoadedRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                            shard_assignments=ShardAssignments(
                                model_id=MODEL_A_ID,
                                runner_to_shard={
                                    RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                    RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                                },
                                node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                            ),
                            hosts=[]
                    )
                },
                runners={RUNNER_1_ID: FailedRunnerStatus(), RUNNER_2_ID: LoadedRunnerStatus()},
            ),
            expected_op=None
        ),

        PlanTestCase(
            description="this node failed, other node spun down -> RunnerDownOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=FailedRunnerStatus(),
                    downloaded=True,
                    device_rank=0,
                ),
                InProcessRunner(
                    runner_id=RUNNER_2_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=ReadyRunnerStatus(),
                    downloaded=True,
                    device_rank=1,
                ),
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle, NODE_B: NodeStatus.Idle},
                instances={
                    INSTANCE_1_ID: Instance(
                        instance_type=InstanceStatus.ACTIVE,
                        instance_id=INSTANCE_1_ID,
                        shard_assignments=ShardAssignments(
                            model_id=MODEL_A_ID,
                            runner_to_shard={
                                RUNNER_1_ID: make_shard_metadata(device_rank=0, world_size=2),
                                RUNNER_2_ID: make_shard_metadata(device_rank=1, world_size=2)
                            },
                            node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID}
                        ),
                        hosts=[]
                    )
                },
                runners={RUNNER_1_ID: FailedRunnerStatus(), RUNNER_2_ID: ReadyRunnerStatus()},
            ),
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID)
        ),

    ]


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


# Pre-compute readable identifiers for each case to avoid lambda typing issues.
@pytest.mark.parametrize(
    "case",
    # We use a factory to delay test case generation until tmp_path is available.
    [pytest.param(c, id=c.id()) for c in _get_test_cases(Path(tempfile.TemporaryDirectory().name))],
)
def test_worker_plan(case: PlanTestCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise Worker.plan across declarative scenarios."""

    print(f"----- case: {case.description}")

    # Regenerate test cases with the actual tmp_path fixture
    test_cases = {c.description: c for c in _get_test_cases(tmp_path)}
    case = test_cases[case.description]

    node_id = NODE_A

    logger = logging.getLogger("test_worker_plan")
    worker = Worker(node_id=node_id, worker_events=None, logger=logger)

    path_downloaded_map: dict[str, bool] = {}

    runner_config: InProcessRunner
    for runner_config in case.in_process_runners:
        
        model_path = tmp_path / f"model_for_runner_{runner_config.runner_id}"
        model_path.mkdir(exist_ok=True, parents=True)

        if len(case.state.instances) == 1: 
            instance_id = next(iter(case.state.instances))

            shard_assignments = case.state.instances[instance_id].shard_assignments
            shard_metadata = shard_assignments.runner_to_shard[runner_config.runner_id]
            
            # Only add this runner if it belongs to our node
            runner_node = None
            for node, runner in shard_assignments.node_to_runner.items():
                if runner == runner_config.runner_id:
                    runner_node = node
                    break
            
            if runner_node != node_id:
                # This runner belongs to a different node, skip it
                continue
                
        elif len(case.state.instances) == 0:
            shard_metadata = PipelineShardMetadata(
                device_rank=runner_config.device_rank,
                world_size=1,
                model_meta=make_model_meta(runner_config.model_id),
                start_layer=0,
                end_layer=1,
                n_layers=1,
            )
        else:
            raise Exception('test_worker_plan not currently designed to have more than 1 instance.')


        assigned_runner = OverrideAssignedRunner(
            runner_id=runner_config.runner_id,
            instance_id=runner_config.instance_id,
            shard_metadata=shard_metadata,
            hosts=[],
            status=runner_config.status,
            runner=None,
            downloaded=runner_config.downloaded
        )
        worker.assigned_runners[runner_config.runner_id] = assigned_runner
        path_downloaded_map[str(build_model_path(shard_metadata.model_meta.model_id))] = runner_config.downloaded

    # Stub filesystem existence check ------------------------------------------------------
    from worker import main as worker_main  # local import for module-scoped os

    def _fake_exists(path: str | Path) -> bool:  # noqa: ANN001  – match os.path.exists signature
        return path_downloaded_map.get(str(path), False)

    monkeypatch.setattr(worker_main.os.path, "exists", _fake_exists)

    op = worker.plan(case.state)
    assert op == case.expected_op
