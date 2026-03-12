from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.logger import init_logger as init_logger
from vllm.parser.abstract_parser import Parser as Parser
from vllm.reasoning import ReasoningParser as ReasoningParser
from vllm.tool_parsers import ToolParser as ToolParser
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import import_from_path as import_from_path

logger: Incomplete

class ParserManager:
    parsers: dict[str, type[Parser]]
    lazy_parsers: dict[str, tuple[str, str]]
    @classmethod
    def get_parser_internal(cls, name: str) -> type[Parser]: ...
    @classmethod
    def register_lazy_module(
        cls, name: str, module_path: str, class_name: str
    ) -> None: ...
    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[Parser] | None = None,
    ) -> type[Parser] | Callable[[type[Parser]], type[Parser]]: ...
    @classmethod
    def list_registered(cls) -> list[str]: ...
    @classmethod
    def import_parser(cls, plugin_path: str) -> None: ...
    @classmethod
    def get_tool_parser(
        cls,
        tool_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
    ) -> type[ToolParser] | None: ...
    @classmethod
    def get_reasoning_parser(
        cls, reasoning_parser_name: str | None
    ) -> type[ReasoningParser] | None: ...
    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
    ) -> type[Parser] | None: ...
