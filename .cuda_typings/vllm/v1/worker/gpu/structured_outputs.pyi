import numpy as np
import torch
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu as async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch

class StructuredOutputsWorker:
    logits_indices: Incomplete
    grammar_bitmask: Incomplete
    device: Incomplete
    copy_stream: Incomplete
    def __init__(
        self, max_num_logits: int, vocab_size: int, device: torch.device
    ) -> None: ...
    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None: ...
