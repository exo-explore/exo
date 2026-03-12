from _typeshed import Incomplete
from pydantic import BaseModel, ValidationInfo as ValidationInfo

class KVCacheQuantSchema(BaseModel):
    dtype: str
    scaling_factor: dict[int, dict[int, float]]
    def check_is_fp8(self) -> KVCacheQuantSchema: ...
    def check_tp_ranks(self, info: ValidationInfo) -> KVCacheQuantSchema: ...
    def check_current_rank(self, info: ValidationInfo) -> KVCacheQuantSchema: ...

class QuantParamSchema(BaseModel):
    model_config: Incomplete
    model_type: str | None
    kv_cache: KVCacheQuantSchema
    def check_model_type(self, info: ValidationInfo) -> QuantParamSchema: ...
