import copy
from typing import cast

from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

_PROTOTYPE = "_exo_detokenizer_prototype"
_build = cast(property, TokenizerWrapper.detokenizer).fget


def _fresh_detokenizer(self: TokenizerWrapper) -> StreamingDetokenizer:
    prototype = cast(StreamingDetokenizer | None, self.__dict__.get(_PROTOTYPE))
    if prototype is None:
        assert _build is not None
        prototype = cast(StreamingDetokenizer, _build(self))
        self.__dict__[_PROTOTYPE] = prototype
    detokenizer = copy.copy(prototype)
    detokenizer.reset()
    return detokenizer


def patch_detokenizer() -> None:
    """Build a tokenizer's streaming detokenizer once, and hand out copies.

    mlx-lm builds a new streaming detokenizer whenever one is asked for, and every
    request asks twice (for its prefill and for its generation). Building one walks the
    whole vocabulary: 0.14 s for Qwen3.5's 248k tokens on an M3 Ultra, on the thread that
    decodes every other request in the batch. A copy shares the vocabulary map, which is
    only read, and gets its own state from reset().
    """
    TokenizerWrapper.detokenizer = property(_fresh_detokenizer)  # pyright: ignore[reportAttributeAccessIssue]
