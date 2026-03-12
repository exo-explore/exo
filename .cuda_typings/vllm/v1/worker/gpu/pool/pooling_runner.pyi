import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.model_executor.models import (
    VllmModelForPooling as VllmModelForPooling,
    is_pooling_model as is_pooling_model,
)
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch
from vllm.v1.worker.gpu.states import RequestState as RequestState

class PoolingRunner:
    model: Incomplete
    def __init__(self, model: nn.Module) -> None: ...
    def get_supported_pooling_tasks(self) -> list[PoolingTask]: ...
    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None: ...
