from _typeshed import Incomplete
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias
from vllm.logger import init_logger as init_logger

logger: Incomplete
CHAT_TEMPLATES_DIR: Incomplete
ChatTemplatePath: TypeAlias = Path | Callable[[str], Path | None]

def register_chat_template_fallback_path(
    model_type: str, chat_template: ChatTemplatePath
) -> None: ...
def get_chat_template_fallback_path(
    model_type: str, tokenizer_name_or_path: str
) -> Path | None: ...
