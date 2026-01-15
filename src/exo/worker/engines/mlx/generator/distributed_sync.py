"""Distributed sync utilities using mx.distributed.all_sum() to broadcast from rank 0."""

# pyright: reportAny=false

import pickle
from typing import TypeVar, cast

import mlx.core as mx

T = TypeVar("T")


def share_object(obj: T | None, rank: int, group: mx.distributed.Group) -> T | None:
    """Broadcast object from rank 0 to all ranks. Two-phase: size then data."""
    if rank == 0:
        if obj is None:
            mx.eval(mx.distributed.all_sum(mx.array([0]), group=group))
            return None
        data = mx.array(list(pickle.dumps(obj)), dtype=mx.uint8)
        mx.eval(mx.distributed.all_sum(mx.array([data.size]), group=group))
        mx.eval(mx.distributed.all_sum(data, group=group))
        return obj
    else:
        size = int(mx.distributed.all_sum(mx.array([0]), group=group).item())
        if size == 0:
            return None
        data = mx.zeros(size, dtype=mx.uint8)
        data = mx.distributed.all_sum(data, group=group)
        mx.eval(data)
        return cast(T, pickle.loads(bytes(cast(list[int], data.tolist()))))
