"""A tokenizer builds its streaming detokenizer once, and every request still gets its own.

exo used to patch mlx-lm for this; mlx-lm's TokenizerWrapper now does it itself (it builds the
detokenizer in __init__ and hands out reset copies). These tests keep it that way."""

from typing import cast

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    StreamingDetokenizer,
    TokenizerWrapper,
)


class CountingTokenizer:
    """Stands in for a Hugging Face tokenizer, counting how often its vocabulary is read."""

    clean_up_tokenization_spaces = False
    tokens = ("Hello", "Ġthere", "Ġfriend")

    def __init__(self) -> None:
        self.vocab_reads = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        self.vocab_reads += 1
        return [self.tokens[i] for i in ids]


def wrap(hf_tokenizer: CountingTokenizer) -> TokenizerWrapper:
    # Only the detokenizer is used here: skip the rest of the wrapper's set-up, which
    # reads much more of the tokenizer
    tokenizer = TokenizerWrapper.__new__(TokenizerWrapper)
    tokenizer.__dict__.update(
        _tokenizer=hf_tokenizer, _detokenizer=BPEStreamingDetokenizer(hf_tokenizer)
    )
    return tokenizer


def test_the_vocabulary_is_walked_once_per_tokenizer() -> None:
    hf_tokenizer = CountingTokenizer()
    tokenizer = wrap(hf_tokenizer)

    _ = tokenizer.detokenizer
    reads_for_one = hf_tokenizer.vocab_reads
    for _ in range(5):
        _ = tokenizer.detokenizer

    assert reads_for_one > 0
    assert hf_tokenizer.vocab_reads == reads_for_one


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
