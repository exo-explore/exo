from .media import MediaWithBytes as MediaWithBytes
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable
from vllm.logger import init_logger as init_logger

logger: Incomplete

class MultiModalHasher:
    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]: ...
    @classmethod
    def iter_item_to_bytes(
        cls, key: str, obj: object
    ) -> Iterable[bytes | memoryview]: ...
    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str: ...
