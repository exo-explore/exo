from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config import ECTransferConfig as ECTransferConfig, VllmConfig as VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase as ECConnectorBase,
    ECConnectorRole as ECConnectorRole,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

class ECConnectorFactory:
    @classmethod
    def register_connector(
        cls, name: str, module_path: str, class_name: str
    ) -> None: ...
    @classmethod
    def create_connector(
        cls, config: VllmConfig, role: ECConnectorRole
    ) -> ECConnectorBase: ...
    @classmethod
    def get_connector_class(
        cls, ec_transfer_config: ECTransferConfig
    ) -> type[ECConnectorBase]: ...
