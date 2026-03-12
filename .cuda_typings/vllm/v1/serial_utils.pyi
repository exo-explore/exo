from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from pydantic import GetCoreSchemaHandler as GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Any, TypeAlias
from vllm import envs as envs
from vllm.logger import init_logger as init_logger
from vllm.multimodal.inputs import (
    BaseMultiModalField as BaseMultiModalField,
    MultiModalBatchedField as MultiModalBatchedField,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalFieldElem as MultiModalFieldElem,
    MultiModalFlatField as MultiModalFlatField,
    MultiModalKwargsItem as MultiModalKwargsItem,
    MultiModalKwargsItems as MultiModalKwargsItems,
    MultiModalSharedField as MultiModalSharedField,
    NestedTensors as NestedTensors,
)
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.v1.utils import tensor_data as tensor_data

logger: Incomplete
CUSTOM_TYPE_PICKLE: int
CUSTOM_TYPE_CLOUDPICKLE: int
CUSTOM_TYPE_RAW_VIEW: int
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str]
bytestr: TypeAlias

class UtilityResult:
    result: Incomplete
    def __init__(self, r: Any = None) -> None: ...

class MsgpackEncoder:
    encoder: Incomplete
    aux_buffers: list[bytestr] | None
    size_threshold: Incomplete
    def __init__(self, size_threshold: int | None = None) -> None: ...
    def encode(self, obj: Any) -> Sequence[bytestr]: ...
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]: ...
    def enc_hook(self, obj: Any) -> Any: ...

class MsgpackDecoder:
    share_mem: Incomplete
    pin_tensors: Incomplete
    decoder: Incomplete
    aux_buffers: Sequence[bytestr]
    def __init__(self, t: Any | None = None, share_mem: bool = True) -> None: ...
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any: ...
    def dec_hook(self, t: type, obj: Any) -> Any: ...
    def ext_hook(self, code: int, data: memoryview) -> Any: ...

def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any: ...

class PydanticMsgspecMixin:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema: ...
