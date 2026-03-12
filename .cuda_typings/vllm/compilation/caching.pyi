import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Generator, Sequence
from torch._dynamo.aot_compile import SerializableCallable
from typing import Any, Literal
from vllm.compilation.compiler_interface import (
    get_inductor_factors as get_inductor_factors,
)
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.config.utils import hash_factors as hash_factors
from vllm.logger import init_logger as init_logger
from vllm.utils.hashing import safe_hash as safe_hash

SerializableCallable = object
logger: Incomplete

class StandaloneCompiledArtifacts:
    submodule_bytes: dict[str, str]
    submodule_bytes_store: dict[str, bytes]
    loaded_submodule_store: dict[str, Any]
    def __init__(self) -> None: ...
    def insert(self, submod_name: str, shape: str, entry: bytes) -> None: ...
    def get(self, submod_name: str, shape: str) -> bytes: ...
    def get_loaded(self, submod_name: str, shape: str) -> Any: ...
    def size_bytes(self) -> int: ...
    def num_artifacts(self) -> int: ...
    def num_entries(self) -> int: ...
    def submodule_names(self) -> list[str]: ...
    def load_all(self) -> None: ...

@contextlib.contextmanager
def patch_pytree_map_over_slice() -> Generator[None, None, Incomplete]: ...

class VllmSerializableFunction(SerializableCallable):
    graph_module: Incomplete
    example_inputs: Incomplete
    prefix: Incomplete
    optimized_call: Incomplete
    is_encoder: Incomplete
    shape_env: Incomplete
    vllm_backend: Incomplete
    sym_tensor_indices: Incomplete
    aot_autograd_config: Incomplete
    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        prefix: str,
        optimized_call: Callable[..., Any],
        is_encoder: bool = False,
        vllm_backend: Any | None = None,
        sym_tensor_indices: list[int] | None = None,
        aot_autograd_config: dict[str, Any] | None = None,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    @classmethod
    def serialize_compile_artifacts(
        cls, compiled_fn: VllmSerializableFunction
    ) -> bytes: ...
    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> VllmSerializableFunction: ...
    def finalize_loading(self, vllm_config: VllmConfig) -> None: ...
    @property
    def co_name(self) -> Literal["VllmSerializableFunction"]: ...

def reconstruct_serializable_fn_from_mega_artifact(
    state: dict[str, Any],
    standalone_compile_artifacts: StandaloneCompiledArtifacts,
    vllm_config: VllmConfig,
    sym_shape_indices_map: dict[str, list[int]],
    returns_tuple_map: dict[str, bool],
) -> VllmSerializableFunction: ...
def aot_compile_hash_factors(vllm_config: VllmConfig) -> list[str]: ...
