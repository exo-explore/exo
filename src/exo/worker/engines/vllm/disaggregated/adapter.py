# type: ignore
"""vLLM-side disaggregation adapter.

Mirrors `engines/mlx/disaggregated/adapter.py` for the vLLM engine: owns
torch dtype ↔ wire dtype, byte (de)serialization, layout conversion (vLLM's
paged block storage → NHD per-token), and the wire-write helpers used by
the producer connector + serve_prefill flow.

Wire format is `engines/.../disaggregated/protocol.py` (msgpack), shared with
the MLX side.
"""

import os
from typing import BinaryIO

import torch

from exo.worker.disaggregated.protocol import (
    DType,
    Header,
    TensorBlob,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)

_TORCH_TO_WIRE: dict[torch.dtype, DType] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_WIRE_TO_TORCH: dict[DType, torch.dtype] = {v: k for k, v in _TORCH_TO_WIRE.items()}


def torch_dtype_to_wire(dtype: torch.dtype) -> DType:
    if dtype not in _TORCH_TO_WIRE:
        raise ValueError(f"Unsupported torch dtype on wire: {dtype}")
    return _TORCH_TO_WIRE[dtype]


def wire_to_torch_dtype(dtype: DType) -> torch.dtype:
    return _WIRE_TO_TORCH[dtype]


def tensor_to_wire_bytes(t: torch.Tensor) -> bytes:
    """Serialize an NHD-laid-out tensor to wire bytes.

    bfloat16 has no native numpy dtype — bitcast through uint16.
    """
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return bytes(t.view(torch.uint16).numpy().tobytes())
    return bytes(t.numpy().tobytes())


def to_nhd(t: torch.Tensor) -> torch.Tensor:
    """Permute HND → NHD when vLLM's KV cache layout is HND."""
    if os.environ.get("VLLM_KV_CACHE_LAYOUT", "HND") == "HND" and t.dim() == 3:
        return t.permute(1, 0, 2)
    return t


def to_bf16(t: torch.Tensor) -> torch.Tensor:
    """Coerce to bfloat16, dequantizing fp8 / uint8-encoded fp8 if needed."""
    if t.dtype == torch.uint8:
        t = t.view(torch.float8_e4m3fn)
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.to(torch.float32).to(torch.bfloat16)
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return t
    return t.to(torch.bfloat16)


def extract_kv_via_slot_mapping(
    kv_layer: torch.Tensor, slot_mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull (keys, values) for the fresh tokens of one layer using slot_mapping.

    `kv_layer` is vLLM's per-layer paged storage. Layout depends on attention
    backend: either `[2, num_blocks, block_size, H, D]` or `[num_blocks, 2,
    block_size, H, D]`. NHD is enforced via `VLLM_KV_CACHE_LAYOUT=NHD`.
    `slot_mapping` is the per-token slot index; entries `< 0` are padding.

    Returned tensors stay on the GPU — the D2H copy is deferred to the
    writer thread (`tensor_to_wire_bytes` calls `.cpu()`) so it doesn't
    block forward of subsequent layers.
    """
    if kv_layer.shape[0] == 2:
        k_all = to_nhd(kv_layer[0])
        v_all = to_nhd(kv_layer[1])
    else:
        k_all = to_nhd(kv_layer[:, 0])
        v_all = to_nhd(kv_layer[:, 1])
    k_flat = k_all.reshape(-1, *k_all.shape[-2:])
    v_flat = v_all.reshape(-1, *v_all.shape[-2:])
    valid = slot_mapping >= 0
    safe_sm = slot_mapping.clamp(min=0)
    keys = to_bf16(k_flat[safe_sm][valid])
    values = to_bf16(v_flat[safe_sm][valid])
    return keys, values


def write_kv_layer_chunk(
    wfile: BinaryIO,
    layer_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Serialize one layer's NHD-shaped K/V to a `KVChunk` on the wire."""
    if keys.dim() == 4:
        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        values = values.reshape(-1, values.shape[-2], values.shape[-1])
    num_tokens = int(keys.shape[0])
    n_heads = int(keys.shape[1])
    head_dim = int(keys.shape[2])
    write_kv_chunk(
        wfile,
        layer_idx=layer_idx,
        num_tokens=num_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype=torch_dtype_to_wire(keys.dtype),
        keys=tensor_to_wire_bytes(keys),
        values=tensor_to_wire_bytes(values),
    )


def arrays_to_blobs(arrays: list[torch.Tensor]) -> list[TensorBlob]:
    """Convert torch tensors (CPU or GPU) to wire-ready `TensorBlob`s."""
    return [
        TensorBlob(
            dtype=torch_dtype_to_wire(arr.dtype),
            shape=tuple(int(d) for d in arr.shape),
            data=tensor_to_wire_bytes(arr),
        )
        for arr in arrays
    ]


def write_layer_arrays_blobs(
    wfile: BinaryIO,
    layer_idx: int,
    blobs: list[TensorBlob],
) -> None:
    write_arrays_state(wfile, layer_idx, blobs)


def write_layer_arrays(
    wfile: BinaryIO,
    layer_idx: int,
    arrays: list[torch.Tensor],
) -> None:
    """Serialize a layer's auxiliary state (SSM/conv) as `ArraysState`."""
    write_layer_arrays_blobs(wfile, layer_idx, arrays_to_blobs(arrays))


def write_prefill_header(
    wfile: BinaryIO,
    *,
    request_id: str,
    model_id: str,
    num_layers: int,
    dtype: DType = "bfloat16",
    start_pos: int = 0,
) -> None:
    write_header(
        wfile,
        Header(
            request_id=request_id,
            model_id=model_id,
            num_layers=num_layers,
            dtype=dtype,
            start_pos=start_pos,
        ),
    )


def write_prefill_done(wfile: BinaryIO, total_tokens: int) -> None:
    write_done(wfile, total_tokens)


def build_layer_to_group(kv_cache_config: object) -> list[int]:
    """Map each layer index (model_runner.kv_caches order) to its kv_cache group.

    vLLM's hybrid models split layers across multiple KV cache groups (e.g.
    full attention vs sliding-window attention). `request_finished_all_groups`
    returns block_ids per group; we need this map to look up the right group
    when reading a layer's blocks.
    """
    group_lookup: dict[str, int] = {}
    for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):  # type: ignore[attr-defined]
        for layer_name in group_spec.layer_names:
            group_lookup[layer_name] = group_idx

    layer_to_group: list[int] = []
    for tensor_spec in kv_cache_config.kv_cache_tensors:  # type: ignore[attr-defined]
        for name in tensor_spec.shared_by:
            layer_to_group.append(group_lookup[name])
    return layer_to_group


def gather_layer_kv_from_blocks(
    layer_kv: torch.Tensor,
    block_ids: list[int],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read K and V for `num_tokens` from a layer's paged block storage.

    Captures APC-cached blocks identically to freshly-computed ones — the
    block pool doesn't distinguish.

    `layer_kv` shapes (NHD, set via `VLLM_KV_CACHE_LAYOUT=NHD`):
        - `[2, num_pool_blocks, block_size, n_kv_heads, head_dim]`, or
        - `[num_pool_blocks, 2, block_size, n_kv_heads, head_dim]`.
    Returns NHD-shaped K and V of shape `[num_tokens, n_kv_heads, head_dim]`,
    on CPU.
    """
    if not block_ids:
        return torch.empty(0), torch.empty(0)
    block_idx_tensor = torch.tensor(
        block_ids, dtype=torch.long, device=layer_kv.device
    )
    if layer_kv.shape[0] == 2:
        # [2, blocks, block, H, D]
        gathered_k = layer_kv[0][block_idx_tensor]
        gathered_v = layer_kv[1][block_idx_tensor]
    else:
        # [blocks, 2, block, H, D]
        gathered = layer_kv[block_idx_tensor]
        gathered_k = gathered[:, 0]
        gathered_v = gathered[:, 1]
    # gathered_k/v: [num_blocks, block_size, H, D]. Concat blocks along seq.
    keys = gathered_k.reshape(-1, *gathered_k.shape[-2:])[:num_tokens]
    values = gathered_v.reshape(-1, *gathered_v.shape[-2:])[:num_tokens]
    return to_bf16(keys).cpu(), to_bf16(values).cpu()
