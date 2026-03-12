from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.plugins import (
    IO_PROCESSOR_PLUGINS_GROUP as IO_PROCESSOR_PLUGINS_GROUP,
    load_plugins_by_group as load_plugins_by_group,
)
from vllm.plugins.io_processors.interface import IOProcessor as IOProcessor
from vllm.renderers import BaseRenderer as BaseRenderer
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

def get_io_processor(
    vllm_config: VllmConfig, renderer: BaseRenderer, plugin_from_init: str | None = None
) -> IOProcessor | None: ...
