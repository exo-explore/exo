import torch
import torch.nn as nn
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache as EncoderCache

def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
): ...
