from .punica_base import PunicaWrapperBase as PunicaWrapperBase
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase: ...
