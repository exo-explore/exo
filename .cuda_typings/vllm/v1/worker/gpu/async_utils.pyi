import contextlib
import numpy as np
import torch
from _typeshed import Incomplete
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    LogprobsTensors as LogprobsTensors,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput as SamplerOutput

class AsyncOutput(AsyncModelRunnerOutput):
    model_runner_output: Incomplete
    sampler_output: Incomplete
    num_sampled_tokens: Incomplete
    copy_event: Incomplete
    sampled_token_ids: Incomplete
    logprobs_tensors: LogprobsTensors | None
    num_nans: np.ndarray | None
    num_sampled_tokens_np: Incomplete
    prompt_logprobs_dict: Incomplete
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ) -> None: ...
    def get_output(self) -> ModelRunnerOutput: ...

class AsyncPoolingOutput(AsyncModelRunnerOutput):
    model_runner_output: Incomplete
    pooler_output: Incomplete
    is_valid: Incomplete
    copy_event: Incomplete
    pooler_output_cpu: Incomplete
    is_valid_cpu: Incomplete
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ) -> None: ...
    def get_output(self) -> ModelRunnerOutput: ...

def async_copy_to_np(x: torch.Tensor) -> np.ndarray: ...
@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream): ...
