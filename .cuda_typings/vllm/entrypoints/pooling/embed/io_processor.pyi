from _typeshed import Incomplete
from vllm.entrypoints.pooling.base.io_processor import (
    PoolingIOProcessor as PoolingIOProcessor,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext as PoolingServeContext
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    token_inputs as token_inputs,
)
from vllm.outputs import (
    PoolingOutput as PoolingOutput,
    PoolingRequestOutput as PoolingRequestOutput,
)
from vllm.utils.collection_utils import chunk_list as chunk_list

class EmbedIOProcessor(PoolingIOProcessor):
    name: str
    pooler_config: Incomplete
    enable_chunked_processing: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def pre_process_online(self, ctx: PoolingServeContext): ...
    def post_process_online(self, ctx: PoolingServeContext): ...
