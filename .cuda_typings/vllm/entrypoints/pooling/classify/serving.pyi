from .io_processor import ClassifyIOProcessor as ClassifyIOProcessor
from .protocol import (
    ClassificationData as ClassificationData,
    ClassificationRequest as ClassificationRequest,
    ClassificationResponse as ClassificationResponse,
)
from _typeshed import Incomplete
from typing import TypeAlias
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig as ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo as UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing as PoolingServing
from vllm.entrypoints.pooling.typing import PoolingServeContext as PoolingServeContext
from vllm.logger import init_logger as init_logger
from vllm.outputs import ClassificationOutput as ClassificationOutput
from vllm.renderers import BaseRenderer as BaseRenderer

logger: Incomplete
ClassificationServeContext: TypeAlias = PoolingServeContext[ClassificationRequest]

class ServingClassification(PoolingServing):
    request_id_prefix: str
    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> ClassifyIOProcessor: ...
