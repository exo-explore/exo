import torch
from ..fx_utils import is_func as is_func
from ..vllm_inductor_pass import VllmInductorPass as VllmInductorPass
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class FixFunctionalizationPass(VllmInductorPass):
    nodes_to_remove: list[torch.fx.Node]
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def defunctionalize(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        mutated_args: dict[int, torch.fx.Node | str],
        args: tuple[torch.fx.Node | str, ...] | None = None,
    ) -> None: ...
    def replace_users_with_mutated_args(
        self, node: torch.fx.Node, mutated_args: dict[int, torch.fx.Node | str]
    ) -> None: ...
    def getitem_users(self, node: torch.fx.Node) -> dict[int, torch.fx.Node]: ...
    def insert_defunctionalized(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        args: tuple[torch.fx.Node | str, ...] | None = None,
    ) -> None: ...
