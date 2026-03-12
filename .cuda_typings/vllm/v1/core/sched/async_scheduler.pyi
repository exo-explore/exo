from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as Scheduler
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus

logger: Incomplete

class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None: ...
