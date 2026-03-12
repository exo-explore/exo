import numpy as np
import torch
from _typeshed import Incomplete
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
)
from vllm.multimodal.inputs import MultiModalKwargsItem as MultiModalKwargsItem
from vllm.multimodal.utils import group_and_batch_mm_kwargs as group_and_batch_mm_kwargs
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache as EncoderCache
from vllm.v1.worker.utils import (
    sanity_check_mm_encoder_outputs as sanity_check_mm_encoder_outputs,
)

class EncoderRunner:
    model: Incomplete
    max_num_tokens: Incomplete
    hidden_size: Incomplete
    encoder_cache: Incomplete
    dtype: Incomplete
    device: Incomplete
    inputs_embeds: Incomplete
    def __init__(
        self,
        model: SupportsMultiModal,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None: ...
    def prepare_mm_inputs(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]: ...
    def execute_mm_encoder(
        self, mm_kwargs: list[tuple[str, MultiModalKwargsItem]]
    ) -> list[torch.Tensor]: ...
    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        computed_prefill_lens: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor]: ...
    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor: ...
