import argparse
import torch
from typing import Any
from vllm.benchmarks.datasets import (
    AIMODataset as AIMODataset,
    BurstGPTDataset as BurstGPTDataset,
    ConversationDataset as ConversationDataset,
    InstructCoderDataset as InstructCoderDataset,
    MultiModalConversationDataset as MultiModalConversationDataset,
    PrefixRepetitionRandomDataset as PrefixRepetitionRandomDataset,
    RandomDataset as RandomDataset,
    RandomDatasetForReranking as RandomDatasetForReranking,
    RandomMultiModalDataset as RandomMultiModalDataset,
    SampleRequest as SampleRequest,
    ShareGPTDataset as ShareGPTDataset,
    SonnetDataset as SonnetDataset,
    VisionArenaDataset as VisionArenaDataset,
    add_random_dataset_base_args as add_random_dataset_base_args,
    add_random_multimodal_dataset_args as add_random_multimodal_dataset_args,
)
from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format as convert_to_pytorch_benchmark_format,
    write_to_json as write_to_json,
)
from vllm.engine.arg_utils import (
    AsyncEngineArgs as AsyncEngineArgs,
    EngineArgs as EngineArgs,
)
from vllm.inputs import TextPrompt as TextPrompt, TokensPrompt as TokensPrompt
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.outputs import RequestOutput as RequestOutput
from vllm.platforms import current_platform as current_platform
from vllm.sampling_params import BeamSearchParams as BeamSearchParams
from vllm.tokenizers import (
    TokenizerLike as TokenizerLike,
    get_tokenizer as get_tokenizer,
)
from vllm.utils.async_utils import merge_async_iterators as merge_async_iterators

def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput] | None]: ...
def run_vllm_chat(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput]]: ...
async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    do_profile: bool,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float: ...
def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: TokenizerLike,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
    dtype: torch.dtype | None = ...,
    enable_torch_compile: bool = False,
) -> float: ...
def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None: ...
def get_requests(args, tokenizer): ...
def filter_requests_for_dp(requests, data_parallel_size): ...
def validate_args(args) -> None: ...
def add_cli_args(parser: argparse.ArgumentParser): ...
def main(args: argparse.Namespace): ...
