import outlines_core as oc
import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.import_utils import LazyLoader as LazyLoader
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.worker.gpu_input_batch import InputBatch as InputBatch

logger: Incomplete
CACHE: Incomplete

def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None: ...

class OutlinesVocabulary:
    inner: Incomplete
    def __init__(self, vocabulary: oc.Vocabulary) -> None: ...

def get_outlines_cache_path() -> str: ...
def get_outlines_cache(): ...

re_llama_byte_token: Incomplete
re_replacement_seq: Incomplete

def get_outlines_vocabulary(tokenizer: TokenizerLike) -> oc.Vocabulary: ...
def grammar_is_likely_lark(grammar_str: str) -> bool: ...
def convert_lark_to_ebnf(grammar_str: str) -> str: ...
def choice_as_grammar(choice: list[str]) -> str: ...
