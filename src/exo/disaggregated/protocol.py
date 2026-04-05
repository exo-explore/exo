from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

import torch

MSG_KV_CHUNK: int = 0x01
MSG_ARRAYS_STATE: int = 0x02
MSG_DONE: int = 0x03


@dataclass
class KVChunk:
    layer_idx: int
    num_tokens: int
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class ArraysState:
    layer_idx: int
    arrays: list[torch.Tensor]


@dataclass
class Done:
    total_tokens: int


Message = KVChunk | ArraysState | Done


def _write_exactly(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return b""
            raise ConnectionError(f"Connection closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def _str_to_dtype(s: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[s]


def _dtype_size(dtype: torch.dtype) -> int:
    return {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]


def write_header(stream: BinaryIO, header: dict[str, object]) -> None:
    payload = json.dumps(header).encode("utf-8")
    _write_exactly(stream, struct.pack(">I", len(payload)))
    _write_exactly(stream, payload)


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().tobytes()  # type: ignore
    return t.contiguous().numpy().tobytes()  # type: ignore


def write_kv_chunk(stream: BinaryIO, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.dim() == 4:
        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        values = values.reshape(-1, values.shape[-2], values.shape[-1])
    keys_bytes = _tensor_to_bytes(keys)
    values_bytes = _tensor_to_bytes(values)
    num_tokens: int = keys.shape[0]
    n_heads: int = keys.shape[1]
    head_dim: int = keys.shape[2]
    header = struct.pack(">BIIII", MSG_KV_CHUNK, layer_idx, num_tokens, n_heads, head_dim)
    _write_exactly(stream, header + keys_bytes + values_bytes)


def _dtype_to_str(dtype: torch.dtype) -> str:
    return {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype]


def write_arrays_state(stream: BinaryIO, layer_idx: int, arrays: list[torch.Tensor]) -> None:
    buf = io.BytesIO()
    buf.write(struct.pack(">BI", MSG_ARRAYS_STATE, layer_idx))
    buf.write(struct.pack(">I", len(arrays)))
    for arr in arrays:
        dtype_str = _dtype_to_str(arr.dtype).encode("utf-8")
        buf.write(struct.pack(">I", len(dtype_str)))
        buf.write(dtype_str)
        shape: tuple[int, ...] = tuple(arr.shape)
        buf.write(struct.pack(">I", len(shape)))
        for dim in shape:
            buf.write(struct.pack(">I", dim))
        buf.write(_tensor_to_bytes(arr))
    _write_exactly(stream, buf.getvalue())


def write_done(stream: BinaryIO, total_tokens: int) -> None:
    _write_exactly(stream, struct.pack(">BI", MSG_DONE, total_tokens))


def read_header(stream: BinaryIO) -> dict[str, object]:
    raw = _read_exactly(stream, 4)
    if not raw:
        raise ConnectionError("No header received")
    length: int = struct.unpack(">I", raw)[0]  # pyright: ignore[reportAny]
    payload = _read_exactly(stream, length)
    return json.loads(payload.decode("utf-8"))  # pyright: ignore[reportAny]


def read_message(stream: BinaryIO, header: dict[str, object]) -> Message | None:
    type_byte = _read_exactly(stream, 1)
    if not type_byte:
        return None
    msg_type = type_byte[0]

    if msg_type == MSG_KV_CHUNK:
        layer_idx: int
        num_tokens: int
        n_heads: int
        head_dim: int
        layer_idx, num_tokens, n_heads, head_dim = struct.unpack(">IIII", _read_exactly(stream, 16))  # pyright: ignore[reportAny]
        dtype = _str_to_dtype(str(header["dtype"]))
        elem_size = _dtype_size(dtype)
        tensor_bytes: int = num_tokens * n_heads * head_dim * elem_size
        keys_raw = _read_exactly(stream, tensor_bytes)
        values_raw = _read_exactly(stream, tensor_bytes)
        shape = (num_tokens, n_heads, head_dim)
        if dtype == torch.bfloat16:
            keys: torch.Tensor = torch.frombuffer(bytearray(keys_raw), dtype=torch.int16).view(torch.bfloat16).reshape(shape).clone()  # type: ignore
            values: torch.Tensor = torch.frombuffer(bytearray(values_raw), dtype=torch.int16).view(torch.bfloat16).reshape(shape).clone()  # type: ignore
        else:
            keys = torch.frombuffer(bytearray(keys_raw), dtype=dtype).reshape(shape).clone()  # type: ignore
            values = torch.frombuffer(bytearray(values_raw), dtype=dtype).reshape(shape).clone()  # type: ignore
        return KVChunk(layer_idx=layer_idx, num_tokens=num_tokens, keys=keys, values=values)  # pyright: ignore[reportUnknownArgumentType]

    if msg_type == MSG_ARRAYS_STATE:
        arr_layer_idx: int
        num_arrays: int
        arr_layer_idx, = struct.unpack(">I", _read_exactly(stream, 4))  # pyright: ignore[reportAny]
        num_arrays, = struct.unpack(">I", _read_exactly(stream, 4))  # pyright: ignore[reportAny]
        fallback_dtype = _str_to_dtype(str(header["dtype"]))
        arrays: list[torch.Tensor] = []
        for _ in range(num_arrays):
            dtype_len_raw = _read_exactly(stream, 4)
            dtype_len: int = struct.unpack(">I", dtype_len_raw)[0]  # pyright: ignore[reportAny]
            if dtype_len > 0 and dtype_len < 20:
                dtype_str_bytes = _read_exactly(stream, dtype_len)
                arr_dtype = _str_to_dtype(dtype_str_bytes.decode("utf-8"))
            else:
                arr_dtype = fallback_dtype
            elem_size = _dtype_size(arr_dtype)
            ndim: int
            ndim, = struct.unpack(">I", _read_exactly(stream, 4))  # pyright: ignore[reportAny]
            shape_arr = struct.unpack(f">{ndim}I", _read_exactly(stream, ndim * 4))
            total_elems = 1
            for d in shape_arr:  # pyright: ignore[reportAny]
                total_elems *= d  # pyright: ignore[reportAny]
            raw = _read_exactly(stream, total_elems * elem_size)
            if arr_dtype == torch.bfloat16:
                t: torch.Tensor = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).reshape(shape_arr).clone()  # type: ignore
            else:
                t = torch.frombuffer(bytearray(raw), dtype=arr_dtype).reshape(shape_arr).clone()  # type: ignore
            arrays.append(t)  # pyright: ignore[reportUnknownArgumentType]
        return ArraysState(layer_idx=arr_layer_idx, arrays=arrays)

    if msg_type == MSG_DONE:
        total_tokens: int
        total_tokens, = struct.unpack(">I", _read_exactly(stream, 4))  # pyright: ignore[reportAny]
        return Done(total_tokens=total_tokens)

    raise ValueError(f"Unknown message type: {msg_type:#x}")
