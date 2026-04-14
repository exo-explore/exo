from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable

from exo.shared.types.tasks import CANCEL_ALL_TASKS, GenerationTask, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    ModelLoadingResponse,
    Response,
)


class Engine(ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: GenerationTask,
    ) -> None: ...

    @abstractmethod
    def step(self) -> Iterable[tuple[TaskId, Response]]: ...

    @abstractmethod
    def close(self) -> None: ...


class Builder(ABC):
    @abstractmethod
    def connect(self, bound_instance: BoundInstance) -> None: ...

    @abstractmethod
    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]: ...

    @abstractmethod
    def build(self) -> Engine: ...

    @abstractmethod
    def close(self) -> None: ...
