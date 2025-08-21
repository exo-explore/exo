from typing import Mapping

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    InstanceId,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.instances import Instance, InstanceStatus
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerFailedOp,
    RunnerOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from exo.shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunnerStatus,
    RunnerStatusType,
    RunningRunnerStatus,
)
from exo.worker.common import AssignedRunner


def unassign_runners(instances: Mapping[InstanceId, Instance], state_runners: Mapping[RunnerId, RunnerStatus], assigned_runners: dict[RunnerId, AssignedRunner]) -> UnassignRunnerOp | None:
    runner_ids: set[RunnerId] = {
        runner_id
        for instance in instances.values()
        for runner_id in instance.shard_assignments.runner_to_shard
    }
    for runner_id, _ in assigned_runners.items():
        if runner_id not in runner_ids:
            return UnassignRunnerOp(runner_id=runner_id)

    # If our instance is in 'downloading' or 'assigned' state, then we know the runner is stale. These are part of AssignRunnerOp and should be blocking.
    for assigned_runner_id in assigned_runners:
        if assigned_runner_id in state_runners and \
            isinstance(state_runners[assigned_runner_id], DownloadingRunnerStatus):
            return UnassignRunnerOp(runner_id=assigned_runner_id)

    return None

def failed_runners(assigned_runners: dict[RunnerId, AssignedRunner]) -> RunnerFailedOp | None:
    for runner_id, assigned_runner in assigned_runners.items():
        if assigned_runner.runner is not None and \
            not assigned_runner.runner.healthy and \
            not isinstance(assigned_runner.status, FailedRunnerStatus):
            return RunnerFailedOp(runner_id=runner_id)
    return None

def spin_down_runners(
    instances: Mapping[InstanceId, Instance], 
    assigned_runners: dict[RunnerId, AssignedRunner], 
    state_runners: Mapping[RunnerId, RunnerStatus],
    worker_node_id: NodeId) -> RunnerDownOp | None:
    for _instance_id, instance in instances.items():
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            if node_id != worker_node_id:
                continue

            # We spin down a runner if it's meant to be inactive and it's Loaded.
            if runner_id in assigned_runners and \
                isinstance(assigned_runners[runner_id].status, LoadedRunnerStatus) and \
                instance.instance_type == InstanceStatus.INACTIVE:
                return RunnerDownOp(runner_id=runner_id)

    # If we are part of an instance that has a dead node - and we aren't the dead node - we should spin down
    for _instance_id, instance in instances.items():
        if worker_node_id in instance.shard_assignments.node_to_runner and \
            instance.shard_assignments.node_to_runner[worker_node_id] in assigned_runners and \
            not isinstance(assigned_runners[instance.shard_assignments.node_to_runner[worker_node_id]].status, InactiveRunnerStatus): # make sure that our runner has not already been spun down into ready state
            other_node_in_instance_has_failed = False
            for runner_id in instance.shard_assignments.runner_to_shard:
                if runner_id in state_runners and \
                    isinstance(state_runners[runner_id], FailedRunnerStatus) and \
                    runner_id not in assigned_runners:
                    other_node_in_instance_has_failed= True

            if other_node_in_instance_has_failed:
                # Spin down *our* runner
                return RunnerDownOp(runner_id=instance.shard_assignments.node_to_runner[worker_node_id])

    # If we are failed - and *all of the other nodes have spun down* - then we can spin down too.
    for _instance_id, instance in instances.items():
        if worker_node_id in instance.shard_assignments.node_to_runner and \
            instance.shard_assignments.node_to_runner[worker_node_id] in state_runners and \
            instance.shard_assignments.node_to_runner[worker_node_id] in assigned_runners and \
            isinstance(assigned_runners[instance.shard_assignments.node_to_runner[worker_node_id]].status, FailedRunnerStatus):

            num_spundown_nodes = 0
            for runner_id in instance.shard_assignments.runner_to_shard:
                if runner_id in state_runners and \
                    isinstance(state_runners[runner_id], InactiveRunnerStatus) and \
                    runner_id not in assigned_runners:
                    num_spundown_nodes += 1
                # Suggested:
                # if runner_id in state_runners and isinstance(state.runners[runner_id], InactiveRunnerStatus):
                #     if runner_id != instance.shard_assignments.node_to_runner[worker_node_id]:
                #         num_spundown_nodes += 1

            if num_spundown_nodes == next(iter(instance.shard_assignments.runner_to_shard.values())).world_size - 1:
                # All the other nodes are spun down - so now we can spin down too.
                # This also catches the case of 1-node. If there's one node in the instance then we should spin down straight away
                return RunnerDownOp(runner_id=instance.shard_assignments.node_to_runner[worker_node_id])
    return None

def assign_runners(instances: Mapping[InstanceId, Instance], assigned_runners: dict[RunnerId, AssignedRunner], worker_node_id: NodeId) -> AssignRunnerOp | None:
    for instance_id, instance in instances.items():
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            if node_id != worker_node_id:
                continue

            if runner_id not in assigned_runners:
                return AssignRunnerOp(
                    runner_id=runner_id,
                    instance_id=instance_id,
                    shard_metadata=instance.shard_assignments.runner_to_shard[runner_id],
                    hosts=instance.hosts
                )
    return None

def spin_up_runners(instances: Mapping[InstanceId, Instance], assigned_runners: dict[RunnerId, AssignedRunner], state_runners: Mapping[RunnerId, RunnerStatus], worker_node_id: NodeId) -> RunnerUpOp | None:
    for _instance_id, instance in instances.items():
        if worker_node_id in instance.shard_assignments.node_to_runner and \
            assigned_runners[instance.shard_assignments.node_to_runner[worker_node_id]].runner is None and \
            instance.instance_type == InstanceStatus.ACTIVE:

            # We are part of this instance, we want it up but it hasn't been spun up yet.
            # Need to assert all other runners are ready before we can spin up.
            ready_to_spin = True
            for runner_id in instance.shard_assignments.node_to_runner.values():
                if runner_id in state_runners and state_runners[runner_id].runner_status != RunnerStatusType.Inactive:
                    ready_to_spin = False

            if ready_to_spin:
                return RunnerUpOp(runner_id=instance.shard_assignments.node_to_runner[worker_node_id])
    return None

def execute_task_op(instances: Mapping[InstanceId, Instance], assigned_runners: dict[RunnerId, AssignedRunner], state_runners: Mapping[RunnerId, RunnerStatus], tasks: Mapping[TaskId, Task], worker_node_id: NodeId) -> ExecuteTaskOp | None:
    for instance_id, instance in instances.items():
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            if node_id != worker_node_id:
                continue
            assert runner_id in assigned_runners
            runner = assigned_runners[runner_id]
            if runner.status.runner_status != RunnerStatusType.Loaded:
                continue # The only previous state to get to Running is from Loaded

            for _, task in tasks.items():
                if task.instance_id == instance_id and (
                    task.task_status == TaskStatus.PENDING or task.task_status == TaskStatus.FAILED
                ):
                    if (runner.shard_metadata.device_rank >= 1 or runner.shard_metadata.world_size == 1):
                        return ExecuteTaskOp(runner_id=runner_id, task=task)
                    else:
                        # We already know our own status is Loaded. We are rank 0,
                        # so let's check that all the other runners are running - ready for us to fire the prompt.
                        running_runner_count = 0
                        for other_runner_id, other_runner_status in state_runners.items():
                            if other_runner_id in instance.shard_assignments.node_to_runner.values() and \
                                    isinstance(other_runner_status, RunningRunnerStatus):
                                running_runner_count += 1

                        if running_runner_count == runner.shard_metadata.world_size - 1:
                            return ExecuteTaskOp(runner_id=runner_id, task=task)

    return None



def plan(assigned_runners: dict[RunnerId, AssignedRunner], 
         worker_node_id: NodeId, 
         instances: Mapping[InstanceId, Instance], 
         state_runners: Mapping[RunnerId, RunnerStatus], # all global
         tasks: Mapping[TaskId, Task]) -> RunnerOp | None:
    # First, unassign assigned runners that are no longer in the state.
    if unop := unassign_runners(instances, state_runners, assigned_runners):
        return unop

    # mark failed runners that are not marked yet as failed
    if failed_op := failed_runners(assigned_runners):
        return failed_op

    # spin down runners that are no longer needed
    if down_op := spin_down_runners(instances, assigned_runners, state_runners, worker_node_id):
        return down_op

    # Then assign runners we do want
    if assign_op := assign_runners(instances, assigned_runners, worker_node_id):
        return assign_op

    # Then spin up 'ready' runners that should be active
    if runner_up_op := spin_up_runners(instances, assigned_runners, state_runners, worker_node_id):
        return runner_up_op

    # Then make sure things are running based on tasks.
    if exec_op := execute_task_op(instances, assigned_runners, state_runners, tasks, worker_node_id):
        return exec_op

    return None
