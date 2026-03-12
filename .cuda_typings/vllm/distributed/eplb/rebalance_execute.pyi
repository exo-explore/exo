import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from torch.distributed import ProcessGroup

__all__ = ["transfer_layer", "move_from_buffer", "RecvMetadata"]

@dataclass
class RecvMetadata:
    recv_primary_mask: np.ndarray
    recv_count: int
    recv_expert_ids: np.ndarray
    recv_dst_rows: np.ndarray

MoveToBufferResult = tuple[np.ndarray, np.ndarray, RecvMetadata]

def move_from_buffer(
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: list[torch.Tensor],
    is_unchanged: np.ndarray,
    is_received_locally: np.ndarray,
    recv_metadata: RecvMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None: ...
async def transfer_layer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    is_profile: bool = False,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
) -> MoveToBufferResult: ...
