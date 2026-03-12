import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    SamplingParams as SamplingParams,
)

logger: Incomplete

class RequestLogger:
    max_log_len: Incomplete
    def __init__(self, *, max_log_len: int | None) -> None: ...
    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None: ...
    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None: ...
