from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.pooling_params import PoolingParams

__all__ = ["ActivationFn", "ClassifierFn", "ProjectorFn", "PoolingParamsUpdate"]

ProjectorFn: Incomplete
ClassifierFn: Incomplete
ActivationFn: Incomplete

@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = ...
    def __or__(self, other: PoolingParamsUpdate) -> PoolingParamsUpdate: ...
    def apply(self, params: PoolingParams) -> None: ...
