from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Self

from exo_core.types.tasks import TaskId


class Cancelled: ...


class Finished: ...


CANCEL_ALL_TASKS = TaskId("CANCEL_TALL_TASKS")


class Engine[TaskType, ResponseType](ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    def cancel_task(self, task_id: TaskId):
        self._cancelled_tasks.add(task_id)

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: TaskType,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[tuple[TaskId, ResponseType | Cancelled | Finished]]: ...

    @abstractmethod
    def close(self) -> None: ...


class EngineBuilder[SetupType, TaskType, ResponseType](ABC):
    @classmethod
    @abstractmethod
    def create(
        cls,
        bound_instance: SetupType,
    ) -> Self: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def load(
        self,
        on_timeout: Callable[[], None],
        on_layer_loaded: Callable[[int, int], None],
    ) -> None: ...

    @abstractmethod
    def build(self) -> Engine[TaskType, ResponseType]: ...

    @abstractmethod
    def close(self) -> None: ...
