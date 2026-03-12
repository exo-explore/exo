import itertools
from collections.abc import Iterable, Iterator, MutableSequence
from dataclasses import dataclass, field
from typing import overload

@dataclass
class Logprob:
    logprob: float
    rank: int | None = ...
    decoded_token: str | None = ...

LogprobsOnePosition = dict[int, Logprob]

@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)
    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None: ...
    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None],
    ) -> None: ...
    def extend(self, logprobs_multi_positions) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...
    @overload
    def __getitem__(self, s: slice) -> FlatLogprobs: ...
    def __setitem__(self, item, value) -> None: ...
    def __delitem__(self, item) -> None: ...
    def insert(self, index: int, value: dict[int, Logprob] | None) -> None: ...
    def __iter__(self) -> Iterator[LogprobsOnePosition]: ...

PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]

def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs: ...
def create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs: ...
def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None: ...
