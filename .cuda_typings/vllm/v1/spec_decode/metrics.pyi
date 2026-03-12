import prometheus_client
from _typeshed import Incomplete
from dataclasses import dataclass, field
from vllm.config import SpeculativeConfig as SpeculativeConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

@dataclass
class SpecDecodingStats:
    num_spec_tokens: int
    num_drafts: int = ...
    num_draft_tokens: int = ...
    num_accepted_tokens: int = ...
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    @classmethod
    def new(cls, num_spec_tokens: int) -> SpecDecodingStats: ...
    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int): ...

class SpecDecodingLogging:
    def __init__(self) -> None: ...
    num_drafts: list[int]
    num_draft_tokens: list[int]
    num_accepted_tokens: list[int]
    accepted_tokens_per_pos_lists: list[list[int]]
    last_log_time: Incomplete
    def reset(self) -> None: ...
    def observe(self, spec_decoding_stats: SpecDecodingStats): ...
    def log(self, log_fn=...) -> None: ...

class SpecDecodingProm:
    spec_decoding_enabled: Incomplete
    counter_spec_decode_num_drafts: Incomplete
    counter_spec_decode_num_draft_tokens: Incomplete
    counter_spec_decode_num_accepted_tokens: Incomplete
    counter_spec_decode_num_accepted_tokens_per_pos: dict[
        int, list[prometheus_client.Counter]
    ]
    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None: ...
    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0): ...

def make_per_engine(
    counter: prometheus_client.Counter, per_engine_labelvalues: dict[int, list[object]]
): ...
