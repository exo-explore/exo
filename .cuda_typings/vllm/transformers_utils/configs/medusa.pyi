import os
from _typeshed import Incomplete
from transformers import PretrainedConfig
from vllm.transformers_utils.utils import (
    without_trust_remote_code as without_trust_remote_code,
)

class MedusaConfig(PretrainedConfig):
    model_type: str
    hidden_size: Incomplete
    vocab_size: Incomplete
    num_heads: Incomplete
    num_hidden_layers: Incomplete
    max_paths: Incomplete
    topk: Incomplete
    max_seq_len: Incomplete
    truncated_vocab_size: Incomplete
    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32001,
        num_heads: int = 5,
        num_hidden_layers: int = 1,
        max_paths: int = 64,
        topk: int = 10,
        truncated_vocab_size: int | None = None,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> MedusaConfig: ...
    @property
    def num_attention_heads(self): ...
    @property
    def num_lookahead_tokens(self): ...
    @num_lookahead_tokens.setter
    def num_lookahead_tokens(self, num_lookahead_tokens: int): ...
