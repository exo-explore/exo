import torch
from _typeshed import Incomplete
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, IO
from vllm import envs as envs
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
    get_quantization_config as get_quantization_config,
)
from vllm.model_executor.layers.quantization.torchao import (
    torchao_version_at_least as torchao_version_at_least,
)
from vllm.platforms import current_platform as current_platform
from vllm.tracing import instrument as instrument
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

runai_model_streamer: Incomplete
fastsafetensors: Incomplete
logger: Incomplete
temp_dir: Incomplete

def enable_hf_transfer() -> None: ...
def enable_xet_high_performance() -> None: ...

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs) -> None: ...

def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None): ...
@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]: ...
def maybe_download_from_modelscope(
    model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str | None: ...
def convert_bin_to_safetensor_file(pt_filename: str, sf_filename: str) -> None: ...
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig: ...
def get_sparse_attention_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> dict[str, Any]: ...
def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str: ...
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str: ...
def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    revision: str | None = None,
) -> None: ...
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]: ...
def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]: ...
def enable_tqdm(use_tqdm_on_load: bool): ...
def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def multi_thread_safetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool, max_workers: int = 4
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def runai_safetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool, is_distributed: bool = False
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def fastsafetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def multi_thread_pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def get_gguf_extra_tensor_names(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> list[str]: ...
def get_gguf_weight_type_map(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]: ...
def gguf_quant_weights_iterator(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]: ...
def convert_pyslice_to_tensor(x: Any) -> torch.Tensor: ...
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None: ...
def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None: ...

LoaderFunction: Incomplete

def sharded_weight_loader(shard_axis: int) -> LoaderFunction: ...
def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction: ...
def initialize_dummy_weights(
    model: torch.nn.Module,
    model_config: ModelConfig,
    low: float = -0.001,
    high: float = 0.001,
    seed: int = 1234,
) -> None: ...
def initialize_single_dummy_weight(
    param: torch.Tensor, low: float = -0.001, high: float = 0.001, seed: int = 1234
) -> None: ...
def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None: ...
