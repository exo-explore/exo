from .protocol import TokenizerLike as TokenizerLike
from _typeshed import Incomplete
from dataclasses import dataclass, field
from pathlib import Path
from vllm.config.model import ModelConfig as ModelConfig, RunnerType as RunnerType
from vllm.logger import init_logger as init_logger
from vllm.transformers_utils.gguf_utils import (
    check_gguf_file as check_gguf_file,
    get_gguf_file_path_from_hf as get_gguf_file_path_from_hf,
    is_gguf as is_gguf,
    is_remote_gguf as is_remote_gguf,
    split_remote_gguf as split_remote_gguf,
)
from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files as any_pattern_in_repo_files,
    is_mistral_model_repo as is_mistral_model_repo,
)
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

@dataclass
class _TokenizerRegistry:
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)
    def register(self, tokenizer_mode: str, module: str, class_name: str) -> None: ...
    def load_tokenizer_cls(self, tokenizer_mode: str) -> type[TokenizerLike]: ...
    def load_tokenizer(self, tokenizer_mode: str, *args, **kwargs) -> TokenizerLike: ...

TokenizerRegistry: Incomplete

def resolve_tokenizer_args(
    tokenizer_name: str | Path,
    *args,
    runner_type: RunnerType = "generate",
    tokenizer_mode: str = "auto",
    **kwargs,
): ...

cached_resolve_tokenizer_args: Incomplete

def tokenizer_args_from_config(config: ModelConfig, **kwargs): ...
def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_cls: type[_T] = ...,
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> _T: ...

cached_get_tokenizer: Incomplete

def cached_tokenizer_from_config(model_config: ModelConfig, **kwargs): ...
