from vllm.entrypoints.pooling.base.io_processor import (
    PoolingIOProcessor as PoolingIOProcessor,
)

class ClassifyIOProcessor(PoolingIOProcessor):
    name: str
