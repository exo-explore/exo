"""Distributed sync utilities using mx.distributed.all_sum() to broadcast from rank 0."""

# pyright: reportAny=false

import pickle
from typing import cast

import mlx.core as mx


def share_object[T](obj: T | None, rank: int, group: mx.distributed.Group) -> T:
    """Broadcast object from rank 0 to all ranks. Two-phase: size then data.

    Rank 0 must always provide a non-None object. Non-rank-0 callers pass None
    (they are receivers only). Use mx_barrier() instead if no data needs to be shared.
    """
    if rank == 0:
        assert obj is not None, (
            "Rank 0 must provide data; use mx_barrier() to sync without data"
        )
        data = mx.array(list(pickle.dumps(obj)), dtype=mx.uint8)
        mx.eval(mx.distributed.all_sum(mx.array([data.size]), group=group))
        mx.eval(mx.distributed.all_sum(data, group=group))
        return obj
    else:
        size = int(mx.distributed.all_sum(mx.array([0]), group=group).item())
        if size == 0:
            raise RuntimeError(
                "share_object received size=0 from rank 0 — protocol violation"
            )
        data = mx.zeros(size, dtype=mx.uint8)
        data = mx.distributed.all_sum(data, group=group)
        mx.eval(data)
        return cast(T, pickle.loads(bytes(cast(list[int], data.tolist()))))
