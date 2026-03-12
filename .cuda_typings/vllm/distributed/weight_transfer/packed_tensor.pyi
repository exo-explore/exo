import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterator
from typing import Any

DEFAULT_PACKED_BUFFER_SIZE_BYTES: Incomplete
DEFAULT_PACKED_NUM_BUFFERS: int

def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = ...,
    num_buffers: int = ...,
) -> None: ...
def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = ...,
    num_buffers: int = ...,
) -> None: ...
