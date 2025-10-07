from typing import Optional

from mlx_lm.models.cache import KVCache

import mlx.core as mx
import mlx.nn as nn  # type: ignore

# These are wrapper functions to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function


class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(self, x: mx.array, cache: Optional[list[KVCache]]) -> mx.array: ...


class Detokenizer:
    def reset(self) -> None: ...
    def add_token(self, token: int) -> None: ...
    def finalize(self) -> None: ...

    @property
    def last_segment(self) -> str: ...


class TokenizerWrapper:
    bos_token: Optional[str]
    eos_token_ids: list[int]
    detokenizer: Detokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
