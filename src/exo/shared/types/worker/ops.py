from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, Instance
from exo.shared.types.worker.runners import RunnerId
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerOp(TaggedModel):
    runner_id: RunnerId


class AssignRunnerOp(BaseRunnerOp):
    instance: Instance

    def bound_instance(self) -> BoundInstance:
        return BoundInstance(instance=self.instance, bound_runner_id=self.runner_id)


class UnassignRunnerOp(BaseRunnerOp):
    pass


class RunnerUpOp(BaseRunnerOp):
    pass


class RunnerDownOp(BaseRunnerOp):
    pass


class ExecuteTaskOp(BaseRunnerOp):
    task: Task


RunnerOp = AssignRunnerOp | ExecuteTaskOp | UnassignRunnerOp | RunnerUpOp | RunnerDownOp
