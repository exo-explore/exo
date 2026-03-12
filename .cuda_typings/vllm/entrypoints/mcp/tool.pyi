import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any
from vllm.entrypoints.openai.responses.context import (
    ConversationContext as ConversationContext,
)
from vllm.logger import init_logger as init_logger
from vllm.utils import random_uuid as random_uuid

logger: Incomplete
MIN_GPT_OSS_VERSION: str

def validate_gpt_oss_install() -> None: ...

class Tool(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    async def get_result(self, context: ConversationContext) -> Any: ...
    @abstractmethod
    async def get_result_parsable_context(
        self, context: ConversationContext
    ) -> Any: ...

class HarmonyBrowserTool(Tool):
    enabled: bool
    browser_tool: Incomplete
    def __init__(self) -> None: ...
    async def get_result(self, context: ConversationContext) -> Any: ...
    async def get_result_parsable_context(
        self, context: ConversationContext
    ) -> Any: ...
    @property
    def tool_config(self) -> Any: ...

class HarmonyPythonTool(Tool):
    enabled: bool
    python_tool: Incomplete
    def __init__(self) -> None: ...
    async def validate(self) -> None: ...
    async def get_result(self, context: ConversationContext) -> Any: ...
    async def get_result_parsable_context(
        self, context: ConversationContext
    ) -> Any: ...
    @property
    def tool_config(self) -> Any: ...
