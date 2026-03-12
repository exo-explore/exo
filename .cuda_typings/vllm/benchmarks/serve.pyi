import aiohttp
import argparse
import ssl
from _typeshed import Incomplete
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from vllm.benchmarks.datasets import (
    SampleRequest as SampleRequest,
    add_dataset_parser as add_dataset_parser,
    get_samples as get_samples,
)
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS as ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS as OPENAI_COMPATIBLE_BACKENDS,
    POOLING_BACKENDS as POOLING_BACKENDS,
    RequestFuncInput as RequestFuncInput,
    RequestFuncOutput as RequestFuncOutput,
)
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint as wait_for_endpoint
from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format as convert_to_pytorch_benchmark_format,
    write_to_json as write_to_json,
)
from vllm.tokenizers import (
    TokenizerLike as TokenizerLike,
    get_tokenizer as get_tokenizer,
)
from vllm.utils.gc_utils import freeze_gc_heap as freeze_gc_heap
from vllm.utils.network_utils import join_host_port as join_host_port

MILLISECONDS_TO_SECONDS_CONVERSION: int
TERM_PLOTLIB_AVAILABLE: Incomplete

async def get_first_model_from_server(
    base_url: str,
    headers: dict | None = None,
    ssl_context: ssl.SSLContext | bool | None = None,
) -> tuple[str, str]: ...
@dataclass
class SpecDecodeMetrics:
    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    accepted_per_pos: dict[int, int]

async def fetch_spec_decode_metrics(
    base_url: str, session: aiohttp.ClientSession
) -> SpecDecodeMetrics | None: ...

class TaskType(Enum):
    GENERATION = "generation"
    POOLING = "pooling"

@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    max_output_tokens_per_s: float
    max_concurrent_requests: int
    rtfx: float = ...

@dataclass
class EmbedBenchmarkMetrics:
    completed: int
    failed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    mean_e2el_ms: float
    std_e2el_ms: float
    median_e2el_ms: float
    percentiles_e2el_ms: float

async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]: ...
def calculate_metrics_for_embeddings(
    outputs: list[RequestFuncOutput], dur_s: float, selected_percentiles: list[float]
) -> EmbedBenchmarkMetrics: ...
def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: TokenizerLike,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]: ...
async def benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: TokenizerLike,
    input_requests: list[SampleRequest],
    logprobs: int | None,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: int | None,
    lora_modules: Iterable[str] | None,
    extra_headers: dict | None,
    extra_body: dict | None,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
    ready_check_timeout_sec: int = 600,
    ssl_context: ssl.SSLContext | bool | None = None,
): ...
def check_goodput_args(args): ...
def parse_goodput(slo_pairs): ...
def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None: ...
def compute_result_filename(
    args: argparse.Namespace, model_id: str, label: str, current_dt: str
) -> str | None: ...
def add_cli_args(parser: argparse.ArgumentParser): ...
def main(args: argparse.Namespace) -> dict[str, Any]: ...
async def main_async(args: argparse.Namespace) -> dict[str, Any]: ...
