import contextlib
import importlib
from collections.abc import Generator

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import wired_limit as _original_wired_limit

from exo.worker.engines.mlx.utils_mlx import wire_memory_enabled

# mlx_lm.generate is shadowed by the generate function mlx_lm exports.
_generate_module = importlib.import_module("mlx_lm.generate")


@contextlib.contextmanager
def _wired_limit(
    model: nn.Module, streams: list[mx.Stream] | None = None
) -> Generator[None, None, None]:
    """Leave memory unwired around ``stream_generate`` when exo does.

    Prefill runs through ``stream_generate``, which wired the working set
    (53GB on a 64GB M5 Pro) for every prompt and unwired it afterwards. On a
    JACCL instance, where exo keeps memory unwired, a prompt of a few hundred
    tokens then took 18-41s on every rank while another request was decoding.
    """
    if wire_memory_enabled():
        with _original_wired_limit(model, streams):
            yield
        return
    try:
        yield
    finally:
        for stream in streams or []:
            mx.synchronize(stream)


def patch_unwired_generation() -> None:
    setattr(_generate_module, "wired_limit", _wired_limit)  # noqa: B010
