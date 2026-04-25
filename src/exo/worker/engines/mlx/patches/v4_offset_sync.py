from typing import Callable, cast

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import Compressor, DeepseekV4Model

_current_int_offset: int | None = None
_applied: bool = False


def _extract_int_offset(cache: object | None) -> int | None:
    if cache is None:
        return None
    for entry in cast(list[object], cache):
        inner_caches = getattr(entry, "caches", None)
        win = inner_caches[0] if inner_caches is not None else entry
        int_off = getattr(win, "_offset", None)
        if isinstance(int_off, int):
            return int_off
        maybe_int = getattr(win, "offset", None)
        if isinstance(maybe_int, int):
            return maybe_int
    return None


_ModelCall = Callable[[DeepseekV4Model, mx.array, list[object] | None], mx.array]
_CompressorCall = Callable[[Compressor, mx.array, object, object, str], mx.array]


def apply() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    original_model_call = cast(_ModelCall, DeepseekV4Model.__call__)

    def patched_model_call(
        self: DeepseekV4Model,
        inputs: mx.array,
        cache: list[object] | None = None,
    ) -> mx.array:
        global _current_int_offset
        prev = _current_int_offset
        _current_int_offset = _extract_int_offset(cache)
        try:
            return original_model_call(self, inputs, cache)
        finally:
            _current_int_offset = prev

    DeepseekV4Model.__call__ = patched_model_call

    original_compressor_call = cast(_CompressorCall, Compressor.__call__)

    def patched_compressor_call(
        self: Compressor,
        x: mx.array,
        cache: object,
        offset: object,
        key: str = "compressor",
    ) -> mx.array:
        if isinstance(offset, mx.array) and _current_int_offset is not None:
            offset = _current_int_offset
        return original_compressor_call(self, x, cache, offset, key)

    Compressor.__call__ = patched_compressor_call


apply()
