from __future__ import annotations

import logging

import pytest

from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ChatCompletionTask,
    ChatCompletionTaskParams,
    TaskStatus,
    TaskType,
)
from exo.shared.types.worker.common import NodeStatus
from exo.shared.types.worker.downloads import (
    DownloadPending,
)
from exo.shared.types.worker.instances import InstanceStatus
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from exo.shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunningRunnerStatus,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.common import AssignedRunner
from exo.worker.download.shard_downloader import NoopShardDownloader
from exo.worker.main import Worker
from exo.worker.plan import plan
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
)
from exo.worker.tests.test_plan.test_worker_plan_utils import (
    InProcessRunner,
    PlanTestCase,
    make_downloading_status,
    make_model_meta,
    make_state,
    make_test_case,
)

"""
The idea with these tests is to define declaratively the input and expected output of the worker.plan function.

We initialize a Worker with InProcessRunners. We then construct a State which gets passed to Worker.plan.
We then check what operation is returned by Worker.plan.

Note that the 'self' node will always be NODE_A. This leads to the swapped-around cases when checking failure cases etc.
"""


def _get_test_cases() -> list[PlanTestCase]:
    # The `model_path` for `RUNNER_1_ID` must exist for the `DownloadOp` test case to pass validation.
    model_a_meta = make_model_meta(MODEL_A_ID)
    return [
        PlanTestCase(
            description="no runners -> no-op",
            in_process_runners=[],
            state=State(
                node_status={NODE_A: NodeStatus.Idle}, instances={}, runners={}
            ),
            expected_op=None,
        ),
        # Both 'assigned' and 'downloading' should be blocking ops - so if we are in either of these we should unassign to retry.
        # This needs to change when we move to an async worker
        make_test_case(
            description="runner state assigned, runner is assigned and downloading -> unassign",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": make_downloading_status(NODE_A),
                    "downloaded": False,
                }
            ],
            instance_status=InstanceStatus.INACTIVE,
            expected_op=UnassignRunnerOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="ready runner, model present -> no-op",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                }
            ],
            instance_status=InstanceStatus.INACTIVE,
            expected_op=None,
        ),
        PlanTestCase(
            description="runner assigned and not in state -> AssignRunnerOp",
            in_process_runners=[],
            state=make_state(
                runner_specs_per_instance={
                    INSTANCE_1_ID: [(RUNNER_1_ID, NODE_A, 0, InactiveRunnerStatus())]
                },
                model_id=MODEL_A_ID,
                instance_status=InstanceStatus.ACTIVE,  # Either active or inactive should yield the same.
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
                hosts=[],
            ),
        ),
        PlanTestCase(
            description="runner assigned but no longer in state -> UnassignRunnerOp",
            in_process_runners=[
                InProcessRunner(
                    runner_id=RUNNER_1_ID,
                    instance_id=INSTANCE_1_ID,
                    model_id=MODEL_A_ID,
                    status=InactiveRunnerStatus(),
                    downloaded=False,
                )
            ],
            state=State(
                node_status={NODE_A: NodeStatus.Idle}, instances={}, runners={}
            ),
            expected_op=UnassignRunnerOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="ready runner (and state up) -> expect RunnerUpOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=RunnerUpOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="1 ready, 1 downloading (and state up) -> no-op",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": DownloadingRunnerStatus(
                        download_progress=DownloadPending(node_id=NODE_A)
                    ),
                    "downloaded": False,
                },
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=None,
        ),
        make_test_case(
            description="2 ready runners (and state up) -> expect RunnerUpOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                },
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=RunnerUpOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="loaded runner (and state down) -> expect RunnerDownOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                }
            ],
            instance_status=InstanceStatus.INACTIVE,
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="failed runner (and state down) -> expect RunnerDownOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": FailedRunnerStatus(),
                    "downloaded": True,
                }
            ],
            instance_status=InstanceStatus.INACTIVE,
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="loaded runner, model present, task pending -> expect ExecuteTaskOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                }
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=ExecuteTaskOp(
                runner_id=RUNNER_1_ID,
                task=ChatCompletionTask(
                    task_id=TASK_1_ID,
                    command_id=COMMAND_1_ID,
                    instance_id=INSTANCE_1_ID,
                    task_type=TaskType.CHAT_COMPLETION,
                    task_status=TaskStatus.PENDING,
                    task_params=ChatCompletionTaskParams(
                        model=str(MODEL_A_ID),
                        messages=[
                            ChatCompletionMessage(role="user", content="Hello, world!")
                        ],
                    ),
                ),
            ),
        ),
        # We should only run rank 0 once all other ranks are running.
        make_test_case(
            description="two loaded runners & task, i'm rank 0 -> no-op",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=None,
        ),
        make_test_case(
            description="two loaded runners & task, i'm rank 1 -> expect ExecuteTaskOp on rank 1",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 1,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=ExecuteTaskOp(
                runner_id=RUNNER_1_ID,
                task=ChatCompletionTask(
                    task_id=TASK_1_ID,
                    command_id=COMMAND_1_ID,
                    instance_id=INSTANCE_1_ID,
                    task_type=TaskType.CHAT_COMPLETION,
                    task_params=ChatCompletionTaskParams(
                        model=str(MODEL_A_ID),
                        messages=[
                            ChatCompletionMessage(role="user", content="Hello, world!")
                        ],
                    ),
                    task_status=TaskStatus.PENDING,
                ),
            ),
        ),
        make_test_case(
            description="rank 1 loaded, rank 0 ready, i'm rank 0  -> expect ExecuteTaskOp on rank 0",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": RunningRunnerStatus(),
                    "downloaded": True,
                },
            ],
            tasks=[
                {
                    "task_id": TASK_1_ID,
                    "instance_id": INSTANCE_1_ID,
                    "status": TaskStatus.PENDING,
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=ExecuteTaskOp(
                runner_id=RUNNER_1_ID,
                task=ChatCompletionTask(
                    task_id=TASK_1_ID,
                    command_id=COMMAND_1_ID,
                    instance_id=INSTANCE_1_ID,
                    task_type=TaskType.CHAT_COMPLETION,
                    task_params=ChatCompletionTaskParams(
                        model=str(MODEL_A_ID),
                        messages=[
                            ChatCompletionMessage(role="user", content="Hello, world!")
                        ],
                    ),
                    task_status=TaskStatus.PENDING,
                ),
            ),
        ),
        make_test_case(
            description="this runner failed (1 node) -> RunnerDownOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": FailedRunnerStatus(),
                    "downloaded": True,
                }
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="other runner failed -> RunnerDownOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": FailedRunnerStatus(),
                    "downloaded": True,
                },
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),
        make_test_case(
            description="this runner failed (2 nodes) -> no-op",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": FailedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": LoadedRunnerStatus(),
                    "downloaded": True,
                },
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=None,
        ),
        make_test_case(
            description="this node failed, other node spun down -> RunnerDownOp",
            runner_specs=[
                {
                    "runner_id": RUNNER_1_ID,
                    "node_id": NODE_A,
                    "device_rank": 0,
                    "status": FailedRunnerStatus(),
                    "downloaded": True,
                },
                {
                    "runner_id": RUNNER_2_ID,
                    "node_id": NODE_B,
                    "device_rank": 1,
                    "status": InactiveRunnerStatus(),
                    "downloaded": True,
                },
            ],
            instance_status=InstanceStatus.ACTIVE,
            expected_op=RunnerDownOp(runner_id=RUNNER_1_ID),
        ),
    ]


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


# Pre-compute readable identifiers for each case to avoid lambda typing issues.
@pytest.mark.parametrize(
    "case",
    # We use a factory to delay test case generation until tmp_path is available.
    [pytest.param(c, id=c.id()) for c in _get_test_cases()],
)
def test_worker_plan(case: PlanTestCase) -> None:
    """Exercise Worker.plan across declarative scenarios."""

    print(f"----- case: {case.description}")

    # Regenerate test cases with the actual tmp_path fixture
    test_cases = {c.description: c for c in _get_test_cases()}
    case = test_cases[case.description]

    node_id = NODE_A

    logger = logging.getLogger("test_worker_plan")
    shard_downloader = NoopShardDownloader()
    worker = Worker(
        node_id=node_id,
        shard_downloader=shard_downloader,
        worker_events=None,
        global_events=None,
        logger=logger,
    )

    runner_config: InProcessRunner
    for runner_config in case.in_process_runners:
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
            raise Exception(
                "test_worker_plan not currently designed to have more than 1 instance."
            )

        assigned_runner = AssignedRunner(
            runner_id=runner_config.runner_id,
            instance_id=runner_config.instance_id,
            shard_metadata=shard_metadata,
            hosts=[],
            status=runner_config.status,
            runner=None,
        )
        worker.assigned_runners[runner_config.runner_id] = assigned_runner

    op = plan(
        worker.assigned_runners,
        NODE_A,
        case.state.instances,
        case.state.runners,
        case.state.tasks,
    )
    assert op == case.expected_op
