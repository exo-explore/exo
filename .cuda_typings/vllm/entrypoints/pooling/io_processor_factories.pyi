from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig as ChatTemplateConfig
from vllm.entrypoints.pooling.base.io_processor import (
    PoolingIOProcessor as PoolingIOProcessor,
)
from vllm.renderers import BaseRenderer as BaseRenderer
from vllm.tasks import SupportedTask as SupportedTask

def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    model_config: ModelConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]: ...
