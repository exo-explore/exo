import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader import get_model as get_model
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer as SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    create_vllm_config_for_draft_model as create_vllm_config_for_draft_model,
)

logger: Incomplete

class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None: ...
