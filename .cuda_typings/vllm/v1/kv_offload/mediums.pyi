import abc
from _typeshed import Incomplete
from abc import ABC
from vllm.v1.kv_offload.abstract import LoadStoreSpec as LoadStoreSpec

class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC, metaclass=abc.ABCMeta):
    block_ids: Incomplete
    def __init__(self, block_ids: list[int]) -> None: ...

class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: ...

class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: ...
