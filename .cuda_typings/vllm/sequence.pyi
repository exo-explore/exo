import torch
from dataclasses import dataclass
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorOutput as KVConnectorOutput,
)

@dataclass
class IntermediateTensors:
    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None: ...
    def __getitem__(self, key: str | slice): ...
    def __setitem__(self, key: str, value: torch.Tensor): ...
    def items(self): ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object): ...
