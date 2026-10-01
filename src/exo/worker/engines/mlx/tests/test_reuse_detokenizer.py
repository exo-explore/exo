"""A tokenizer builds its streaming detokenizer once, and every request still gets its own."""

from typing import cast

import pytest
from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    StreamingDetokenizer,
    TokenizerWrapper,
)

from exo.worker.engines.mlx.patches.reuse_detokenizer import (
    _fresh_detokenizer,  # pyright: ignore[reportPrivateUsage]
)


class CountingTokenizer:
    """Stands in for a Hugging Face tokenizer, counting how often its vocabulary is read."""

    clean_up_tokenization_spaces = False

    def __init__(self) -> None:
        self.vocab_reads = 0

    @property
    def vocab(self) -> dict[str, int]:
        self.vocab_reads += 1
        return {"Hello": 0, "Ġthere": 1, "Ġfriend": 2}


def wrap(hf_tokenizer: CountingTokenizer) -> TokenizerWrapper:
    # Only the detokenizer is used here: skip the wrapper's set-up, which reads much more
    # of the tokenizer
    tokenizer = TokenizerWrapper.__new__(TokenizerWrapper)
    tokenizer.__dict__.update(
        _tokenizer=hf_tokenizer, _detokenizer_class=BPEStreamingDetokenizer
    )
    return tokenizer


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TokenizerWrapper, "detokenizer", property(_fresh_detokenizer))


@pytest.mark.usefixtures("patched")
def test_the_vocabulary_is_walked_once_per_tokenizer() -> None:
    hf_tokenizer = CountingTokenizer()
    tokenizer = wrap(hf_tokenizer)

    _ = tokenizer.detokenizer
    reads_for_one = hf_tokenizer.vocab_reads
    for _ in range(5):
        _ = tokenizer.detokenizer

    assert reads_for_one > 0
    assert hf_tokenizer.vocab_reads == reads_for_one


@pytest.mark.usefixtures("patched")
def test_each_request_gets_its_own_detokenizer_state() -> None:
    tokenizer = wrap(CountingTokenizer())

    def decode(detokenizer: StreamingDetokenizer, tokens: list[int]) -> str:
        text = ""
        for token in tokens:
            detokenizer.add_token(token)
            text += detokenizer.last_segment
        detokenizer.finalize()
        return text + detokenizer.last_segment

    first = cast(StreamingDetokenizer, tokenizer.detokenizer)
    first.add_token(0)
    # A second request starts while the first is under way
    second = cast(StreamingDetokenizer, tokenizer.detokenizer)

    assert decode(second, [0, 2]) == "Hello friend"
    assert first.last_segment + decode(first, [1]) == "Hello there"
