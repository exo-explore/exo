import torch.fx
from ..fx_utils import is_func as is_func
from ..vllm_inductor_pass import VllmInductorPass as VllmInductorPass
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import SymInt as SymInt
from vllm.logger import init_logger as init_logger

logger: Incomplete

class NoOpEliminationPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def dims_equivalent(self, dim: int | SymInt, i_dim: int | SymInt) -> bool: ...
    def all_dims_equivalent(
        self, dims: Iterable[int | SymInt], i_dims: Iterable[int | SymInt]
    ) -> bool: ...
