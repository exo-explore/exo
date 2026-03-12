from vllm.distributed.parallel_state import (
    get_pp_group as get_pp_group,
    get_tp_group as get_tp_group,
    get_world_group as get_world_group,
)
from vllm.distributed.stateless_coordinator import (
    StatelessGroupCoordinator as StatelessGroupCoordinator,
)

def get_standby_dp_group() -> StatelessGroupCoordinator | None: ...
def get_standby_ep_group() -> StatelessGroupCoordinator | None: ...
def get_standby_eplb_group() -> StatelessGroupCoordinator | None: ...
def get_standby_world_group() -> StatelessGroupCoordinator | None: ...
def create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    world_group_ports: list[list[int]],
    dp_group_ports: list[list[int]],
    ep_group_ports: list[list[int]],
    eplb_group_ports: list[list[int]] | None = None,
    backend: str | None = None,
) -> None: ...
def pop_standby_groups() -> dict: ...
