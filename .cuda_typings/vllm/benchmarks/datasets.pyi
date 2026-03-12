import abc
import argparse
import numpy as np
from PIL import Image
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.utils import get_adapter_absolute_path as get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict as MultiModalDataDict
from vllm.multimodal.image import convert_image_mode as convert_image_mode
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.argparse_utils import FlexibleArgumentParser as FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

datasets: Incomplete
logger: Incomplete
DEFAULT_NUM_PROMPTS: int

@dataclass
class SampleRequest:
    prompt: str | list[str]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: MultiModalDataDict | dict | list[dict] | None = ...
    lora_request: LoRARequest | None = ...
    request_id: str | None = ...

class BenchmarkDataset(ABC, metaclass=abc.ABCMeta):
    DEFAULT_SEED: int
    IS_MULTIMODAL: bool
    dataset_path: Incomplete
    random_seed: Incomplete
    disable_shuffle: Incomplete
    data: Incomplete
    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = ...,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None: ...
    def apply_multimodal_chat_transformation(
        self,
        prompt: str,
        mm_content: MultiModalDataDict | dict | list[dict] | None = None,
    ) -> list[dict]: ...
    def load_data(self) -> None: ...
    def get_random_lora_request(
        self, max_loras: int | None = None, lora_path: str | None = None
    ) -> LoRARequest | None: ...
    @abstractmethod
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> list[SampleRequest]: ...
    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> None: ...

def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool: ...
@cache
def lora_path_on_disk(lora_path: str) -> str: ...

lora_tokenizer_cache: dict[int, TokenizerLike]

def process_image(image: Any) -> Mapping[str, Any]: ...
def process_video(video: Any) -> Mapping[str, Any]: ...
def gen_prompt_decode_to_target_len(
    tokenizer: TokenizerLike,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[int], int]: ...

class RandomDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN: int
    DEFAULT_RANGE_RATIO: float
    DEFAULT_INPUT_LEN: int
    DEFAULT_OUTPUT_LEN: int
    def __init__(self, **kwargs) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = ...,
        range_ratio: float = ...,
        input_len: int = ...,
        output_len: int = ...,
        batchsize: int = 1,
        **kwargs,
    ) -> list[SampleRequest]: ...
    def get_prefix(
        self, tokenizer: TokenizerLike, allowed_tokens: np.ndarray, prefix_len: int
    ) -> list[int]: ...
    def get_sampling_params(
        self,
        num_requests: int,
        range_ratio: float,
        input_len: int,
        output_len: int,
        tokenizer: TokenizerLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def generate_token_sequence(
        self,
        *,
        tokenizer: TokenizerLike,
        prefix_token_ids: list[int],
        prefix_len: int,
        vocab_size: int,
        input_len: int,
        offset: int,
        index: int,
        allowed_tokens: np.ndarray,
    ) -> tuple[str, int, int]: ...

class RandomDatasetForReranking(RandomDataset):
    def __init__(self, **kwargs) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        range_ratio: float = ...,
        input_len: int = ...,
        batchsize: int = 1,
        is_reranker: bool = True,
        **kwargs,
    ) -> list[SampleRequest]: ...

class RandomMultiModalDataset(RandomDataset):
    IS_MULTIMODAL: bool
    DEFAULT_LIMIT_MM_PER_PROMPT: Incomplete
    DEFAULT_BASE_ITEMS_PER_REQUEST: int
    DEFAULT_NUM_MM_ITEMS_RANGE_RATIO: float
    DEFAULT_MM_ITEM_BUCKET_CONFIG: Incomplete
    DEFAULT_ENABLE_MULTIMODAL_CHAT: bool
    def __init__(self, **kwargs) -> None: ...
    def generate_synthetic_image(self, width: int, height: int) -> Image.Image: ...
    def generate_synthetic_video(
        self, width: int, height: int, num_frames: int
    ) -> dict: ...
    def map_config_to_modality(self, config: tuple[int, int, int]) -> str: ...
    def normalize_bucket_config(
        self, bucket_config: dict[tuple[int, int, int], float]
    ) -> dict[tuple[int, int, int], float]: ...
    def generate_mm_item(
        self, mm_item_config: tuple[int, int, int]
    ) -> Mapping[str, Any]: ...
    def get_mm_item_sampling_params(
        self,
        base_items_per_request: int,
        num_mm_items_range_ratio: float,
        limit_mm_per_prompt: dict[str, int],
        bucket_config: dict[tuple[int, int, int], float],
    ) -> tuple[int, int, dict[str, int], dict[tuple[int, int, int], float]]: ...
    def get_mm_item_iterator(
        self,
        min_num_mm_items: int,
        max_num_mm_items: int,
        bucket_config: dict[tuple[int, int, int], float],
        limit_mm_per_prompt: dict[str, int],
    ) -> Iterator[tuple[int, int, int]]: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = ...,
        range_ratio: float = ...,
        input_len: int = ...,
        output_len: int = ...,
        limit_mm_per_prompt: dict[str, int] = ...,
        base_items_per_request: int = ...,
        num_mm_items_range_ratio: float = ...,
        bucket_config: dict[tuple[int, int, int], float] = ...,
        enable_multimodal_chat: bool = ...,
        **kwargs,
    ) -> list[SampleRequest]: ...

class ShareGPTDataset(BenchmarkDataset):
    def __init__(self, **kwargs) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        lora_path: str | None = None,
        max_loras: int | None = None,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class _ValidateDatasetArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None: ...

def add_dataset_parser(parser: FlexibleArgumentParser): ...
def add_random_dataset_base_args(
    parser_or_group: FlexibleArgumentParser | argparse._ArgumentGroup,
) -> None: ...
def add_random_multimodal_dataset_args(
    parser_or_group: FlexibleArgumentParser | argparse._ArgumentGroup,
) -> None: ...
def get_samples(args, tokenizer: TokenizerLike) -> list[SampleRequest]: ...

class CustomDataset(BenchmarkDataset):
    def __init__(self, **kwargs) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...
    num_available_samples: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        lora_path: str | None = None,
        max_loras: int | None = None,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class CustomMMDataset(CustomDataset):
    IS_MULTIMODAL: bool
    num_available_samples: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class SpecBench(CustomDataset):
    category: Incomplete
    def __init__(self, **kwargs) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...
    def sample(self, **kwargs) -> list: ...

class SonnetDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN: int
    DEFAULT_INPUT_LEN: int
    DEFAULT_OUTPUT_LEN: int
    def __init__(self, **kwargs) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        prefix_len: int = ...,
        input_len: int = ...,
        output_len: int = ...,
        return_prompt_formatted: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class BurstGPTDataset(BenchmarkDataset):
    def __init__(self, **kwargs) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        max_loras: int | None = None,
        lora_path: str | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]: ...

class HuggingFaceDataset(BenchmarkDataset, metaclass=abc.ABCMeta):
    SUPPORTED_DATASET_PATHS: set[str] | dict[str, Callable]
    dataset_split: Incomplete
    dataset_subset: Incomplete
    load_stream: Incomplete
    hf_name: Incomplete
    trust_remote_code: Incomplete
    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        no_stream: bool = False,
        dataset_subset: str | None = None,
        hf_name: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None: ...
    data: Incomplete
    def load_data(self) -> None: ...

class ConversationDataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    IS_MULTIMODAL: bool
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class MultiModalConversationDataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    IS_MULTIMODAL: bool
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class VisionArenaDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    IS_MULTIMODAL: bool
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class MMVUDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class InstructCoderDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]: ...
    def sample_prompts(self, n: int) -> Iterator[str]: ...

class MTBenchDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class BlazeditDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        min_distance: float = 0.0,
        max_distance: float = 1.0,
        **kwargs,
    ) -> list: ...

class AIMODataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

zeta_prompt: str

class NextEditPredictionDataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    MAPPING_PROMPT_FUNCS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ): ...

class ASRDataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    DEFAULT_OUTPUT_LEN: int
    IS_MULTIMODAL: bool
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list: ...

class MLPerfDataset(HuggingFaceDataset):
    SUPPORTED_DATASET_PATHS: Incomplete
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]: ...

class PrefixRepetitionRandomDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN: int
    DEFAULT_SUFFIX_LEN: int
    DEFAULT_NUM_PREFIXES: int
    DEFAULT_OUTPUT_LEN: int
    def __init__(self, **kwargs) -> None: ...
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        prefix_len: int = ...,
        suffix_len: int = ...,
        num_prefixes: int = ...,
        output_len: int = ...,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]: ...

class MMStarDataset(HuggingFaceDataset):
    DEFAULT_OUTPUT_LEN: int
    SUPPORTED_DATASET_PATHS: Incomplete
    IS_MULTIMODAL: bool
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]: ...
