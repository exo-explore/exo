import ctypes
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.distributed import ReduceOp
from typing import Any

__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
]

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
ncclWindow_t = ctypes.c_void_p

class ncclUniqueId(ctypes.Structure): ...

cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int

class ncclDataTypeEnum:
    ncclInt8: int
    ncclChar: int
    ncclUint8: int
    ncclInt32: int
    ncclInt: int
    ncclUint32: int
    ncclInt64: int
    ncclUint64: int
    ncclFloat16: int
    ncclHalf: int
    ncclFloat32: int
    ncclFloat: int
    ncclFloat64: int
    ncclDouble: int
    ncclBfloat16: int
    ncclFloat8e4m3: int
    ncclNumTypes: int
    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int: ...

ncclRedOp_t = ctypes.c_int

class ncclRedOpTypeEnum:
    ncclSum: int
    ncclProd: int
    ncclMax: int
    ncclMin: int
    ncclAvg: int
    ncclNumOps: int
    @classmethod
    def from_torch(cls, op: ReduceOp) -> int: ...

@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]

class NCCLLibrary:
    exported_functions: Incomplete
    path_to_library_cache: dict[str, Any]
    path_to_dict_mapping: dict[str, dict[str, Any]]
    lib: Incomplete
    def __init__(self, so_file: str | None = None) -> None: ...
    def ncclGetErrorString(self, result: ncclResult_t) -> str: ...
    def NCCL_CHECK(self, result: ncclResult_t) -> None: ...
    def ncclGetRawVersion(self) -> int: ...
    def ncclGetVersion(self) -> str: ...
    def ncclGetUniqueId(self) -> ncclUniqueId: ...
    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId: ...
    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t: ...
    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None: ...
    def ncclCommDestroy(self, comm: ncclComm_t) -> None: ...
    def ncclGroupStart(self) -> None: ...
    def ncclGroupEnd(self) -> None: ...
    def ncclCommWindowRegister(
        self, comm: ncclComm_t, buff: buffer_type, size: int, win_flags: int
    ) -> ncclWindow_t: ...
    def ncclCommWindowDeregister(
        self, comm: ncclComm_t, window: ncclWindow_t
    ) -> None: ...
