import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from mcp.types import ListToolsResult as ListToolsResult
from openai_harmony import ToolNamespaceConfig
from typing import Any
from vllm.entrypoints.mcp.tool import (
    HarmonyBrowserTool as HarmonyBrowserTool,
    HarmonyPythonTool as HarmonyPythonTool,
    Tool as Tool,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

async def list_server_and_tools(server_url: str): ...
def trim_schema(schema: dict) -> dict: ...
def post_process_tools_description(
    list_tools_result: ListToolsResult,
) -> ListToolsResult: ...

class ToolServer(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def has_tool(self, tool_name: str) -> bool: ...
    @abstractmethod
    def get_tool_description(
        self, tool_name: str, allowed_tools: list[str] | None = None
    ) -> ToolNamespaceConfig | None: ...
    @abstractmethod
    def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ) -> AbstractAsyncContextManager[Any]: ...

class MCPToolServer(ToolServer):
    harmony_tool_descriptions: Incomplete
    def __init__(self) -> None: ...
    urls: dict[str, str]
    async def add_tool_server(self, server_url: str): ...
    def has_tool(self, tool_name: str): ...
    def get_tool_description(
        self, server_label: str, allowed_tools: list[str] | None = None
    ) -> ToolNamespaceConfig | None: ...
    @asynccontextmanager
    async def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ): ...

class DemoToolServer(ToolServer):
    tools: dict[str, Tool]
    def __init__(self) -> None: ...
    async def init_and_validate(self) -> None: ...
    def has_tool(self, tool_name: str) -> bool: ...
    def get_tool_description(
        self, tool_name: str, allowed_tools: list[str] | None = None
    ) -> ToolNamespaceConfig | None: ...
    @asynccontextmanager
    async def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ): ...
