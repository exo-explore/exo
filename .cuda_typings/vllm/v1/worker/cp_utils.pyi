from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed import (
    get_dcp_group as get_dcp_group,
    get_pcp_group as get_pcp_group,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)

def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None: ...
def get_total_cp_world_size(): ...
