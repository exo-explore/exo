import torch
from collections.abc import Sequence
from typing import Any, TypeAlias
from typing_extensions import Required, TypedDict
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker as BaseMultiModalItemTracker,
    ChatCompletionContentPartImageEmbedsParam as ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam as ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam as ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam as ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoParam as ChatCompletionContentPartVideoParam,
    ChatTemplateResolutionError as ChatTemplateResolutionError,
    ConversationMessage as ConversationMessage,
    MultiModalItemTracker as MultiModalItemTracker,
)
from vllm.inputs import TokensPrompt as TokensPrompt
from vllm.inputs.data import PromptType as PromptType, TextPrompt as TextPrompt
from vllm.model_executor.models.interfaces import (
    supports_score_template as supports_score_template,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalUUIDDict as MultiModalUUIDDict,
)
from vllm.outputs import PoolingRequestOutput as PoolingRequestOutput
from vllm.platforms import current_platform as current_platform
from vllm.renderers.hf import safe_apply_chat_template as safe_apply_chat_template
from vllm.tokenizers import TokenizerLike as TokenizerLike

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartTextParam
    | ChatCompletionContentPartVideoParam
)

def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor: ...
def compute_maxsim_scores(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 16,
    max_score_matrix_elements: int = 16000000,
    use_gpu_for_pooling_score: bool = False,
) -> list[torch.Tensor]: ...

class ScoreMultiModalParam(TypedDict, total=False):
    content: Required[list[ScoreContentPartParam]]

ScoreInput = str | ScoreMultiModalParam
ScoreInputs = ScoreInput | list[ScoreInput]
ScoreData = str | list[ScoreContentPartParam]

def validate_score_input(
    data_1: ScoreInputs,
    data_2: ScoreInputs,
    is_multimodal_model: bool,
    architecture: str,
) -> tuple[list[ScoreData], list[ScoreData]]: ...
def parse_score_data(
    data_1: ScoreData, data_2: ScoreData, model_config: ModelConfig
) -> tuple[str, str, MultiModalDataDict | None, MultiModalUUIDDict | None]: ...
def parse_score_data_single(
    data: ScoreData, role: str, model_config: ModelConfig
) -> tuple[str, MultiModalDataDict | None, MultiModalUUIDDict | None]: ...
def score_data_to_prompts(
    data_list: list[ScoreData], role: str, model_config: ModelConfig
) -> list[PromptType]: ...
def post_process_tokens(model_config: ModelConfig, prompt: TokensPrompt) -> None: ...
def get_score_prompt(
    model_config: ModelConfig,
    tokenizer: TokenizerLike,
    tokenization_kwargs: dict[str, Any],
    data_1: ScoreData,
    data_2: ScoreData,
    score_template: str | None = None,
) -> tuple[str, TokensPrompt]: ...
def compress_token_type_ids(token_type_ids: list[int]) -> int: ...
