import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Generic, TypeVar
from vllm.config import VllmConfig as VllmConfig
from vllm.inputs.data import PromptType as PromptType
from vllm.outputs import PoolingRequestOutput as PoolingRequestOutput
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import BaseRenderer as BaseRenderer
from vllm.sampling_params import SamplingParams as SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(
    ABC, Generic[IOProcessorInput, IOProcessorOutput], metaclass=abc.ABCMeta
):
    vllm_config: Incomplete
    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer) -> None: ...
    def parse_data(self, data: object) -> IOProcessorInput: ...
    def merge_sampling_params(
        self, params: SamplingParams | None = None
    ) -> SamplingParams: ...
    def merge_pooling_params(
        self, params: PoolingParams | None = None
    ) -> PoolingParams: ...
    @abstractmethod
    def pre_process(
        self, prompt: IOProcessorInput, request_id: str | None = None, **kwargs
    ) -> PromptType | Sequence[PromptType]: ...
    async def pre_process_async(
        self, prompt: IOProcessorInput, request_id: str | None = None, **kwargs
    ) -> PromptType | Sequence[PromptType]: ...
    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput: ...
    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput: ...
