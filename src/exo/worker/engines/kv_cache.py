# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


@dataclass
class KVLayerState:
    keys: torch.Tensor  # [seq_len, n_heads, head_dim]
    values: torch.Tensor  # [seq_len, n_heads, head_dim]


@dataclass
class RotatingKVLayerState:
    keys: torch.Tensor  # [buffer_len, n_heads, head_dim]
    values: torch.Tensor  # [buffer_len, n_heads, head_dim]
    keep: int
    max_size: int
    offset: int
    idx: int


@dataclass
class ArraysLayerState:
    arrays: list[torch.Tensor | None]


LayerState = KVLayerState | RotatingKVLayerState | ArraysLayerState


def _mx_to_torch(arr: mx.array) -> torch.Tensor:
    mx.eval(arr)
    if arr.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(arr.astype(mx.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(arr))


def _torch_to_mx(t: torch.Tensor) -> mx.array:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.numpy())


def _split_kv(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a vLLM paged KV tensor into K and V views, each [num_blocks, block_size, H, D].

    flash_attn backend: (2, num_blocks, block_size, H, D) — K/V at dim 0
    triton_attn backend: (num_blocks, 2, block_size, H, D) — K/V at dim 1
    """
    if kv.shape[0] == 2 and kv.shape[1] != 2:
        return kv[0], kv[1]
    return kv[:, 0], kv[:, 1]


def _kv_to_nhd(k: mx.array, v: mx.array) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MLX BHSD [1, H, S, D] to NHD [S, H, D] torch tensors."""
    kt = _mx_to_torch(k).squeeze(0).permute(1, 0, 2)  # [H,S,D] -> [S,H,D]
    vt = _mx_to_torch(v).squeeze(0).permute(1, 0, 2)
    return kt, vt


def _nhd_to_bhsd(kt: torch.Tensor, vt: torch.Tensor) -> tuple[mx.array, mx.array]:
    """Convert NHD [S, H, D] torch tensors to MLX BHSD [1, H, S, D]."""
    k_mx = _torch_to_mx(kt.permute(1, 0, 2).unsqueeze(0))  # [S,H,D] -> [1,H,S,D]
    v_mx = _torch_to_mx(vt.permute(1, 0, 2).unsqueeze(0))
    return k_mx, v_mx


class TorchKVCache:
    def __init__(self, layers: list[LayerState]):
        self.layers = layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def layer(self, idx: int) -> LayerState:
        return self.layers[idx]

    def kv_layers(self) -> list[tuple[int, KVLayerState | RotatingKVLayerState]]:
        return [
            (i, layer)
            for i, layer in enumerate(self.layers)
            if isinstance(layer, (KVLayerState, RotatingKVLayerState))
        ]

    def detach_cpu(self) -> TorchKVCache:
        layers: list[LayerState] = []
        for layer in self.layers:
            if isinstance(layer, KVLayerState):
                if not layer.keys.is_cuda:
                    layers.append(layer)
                else:
                    layers.append(KVLayerState(
                        keys=layer.keys.detach().to("cpu", non_blocking=True),
                        values=layer.values.detach().to("cpu", non_blocking=True),
                    ))
            elif isinstance(layer, RotatingKVLayerState):
                layers.append(RotatingKVLayerState(
                    keys=layer.keys.detach().to("cpu", non_blocking=True),
                    values=layer.values.detach().to("cpu", non_blocking=True),
                    keep=layer.keep, max_size=layer.max_size,
                    offset=layer.offset, idx=layer.idx,
                ))
            else:
                layers.append(deepcopy(layer))
        if any(l.keys.is_cuda for l in self.layers if isinstance(l, (KVLayerState, RotatingKVLayerState))):
            torch.cuda.synchronize()
        return TorchKVCache(layers)

    def trim_to(self, num_tokens: int) -> TorchKVCache:
        layers: list[LayerState] = []
        for layer in self.layers:
            if isinstance(layer, KVLayerState):
                layers.append(KVLayerState(
                    keys=layer.keys[:num_tokens],
                    values=layer.values[:num_tokens],
                ))
            else:
                layers.append(deepcopy(layer))
        return TorchKVCache(layers)

    @classmethod
    def from_mlx_cache(
        cls, cache: list[KVCache | RotatingKVCache | ArraysCache],
    ) -> TorchKVCache:
        layers: list[LayerState] = []
        for c in cache:
            if isinstance(c, RotatingKVCache):
                if c.keys is None:
                    layers.append(RotatingKVLayerState(
                        keys=torch.empty(0), values=torch.empty(0),
                        keep=c.keep, max_size=c.max_size, offset=c.offset, idx=c._idx,
                    ))
                else:
                    k, v = c.state
                    kt, vt = _kv_to_nhd(k, v)  # pyright: ignore[reportArgumentType]
                    keep, max_size, offset, idx = (int(x) for x in c.meta_state)
                    layers.append(RotatingKVLayerState(
                        keys=kt, values=vt,
                        keep=keep, max_size=max_size, offset=offset, idx=idx,
                    ))
            elif isinstance(c, ArraysCache):
                arrays: list[torch.Tensor | None] = []
                for arr in c.state:
                    arrays.append(_mx_to_torch(arr) if arr is not None else None)
                layers.append(ArraysLayerState(arrays=arrays))
            else:
                if c.keys is None:  # pyright: ignore[reportUnnecessaryComparison]
                    layers.append(KVLayerState(keys=torch.empty(0), values=torch.empty(0)))
                else:
                    k, v = c.state
                    kt, vt = _kv_to_nhd(k, v)  # pyright: ignore[reportArgumentType]
                    layers.append(KVLayerState(keys=kt, values=vt))
        return cls(layers)

    def to_mlx_cache(self) -> list[KVCache | RotatingKVCache | ArraysCache]:
        result: list[KVCache | RotatingKVCache | ArraysCache] = []
        for layer in self.layers:
            if isinstance(layer, RotatingKVLayerState):
                c = RotatingKVCache(max_size=layer.max_size, keep=layer.keep)
                if layer.keys.numel() > 0:
                    k_mx, v_mx = _nhd_to_bhsd(layer.keys, layer.values)
                    c.state = (k_mx, v_mx)
                    c.meta_state = tuple(
                        str(x) for x in (layer.keep, layer.max_size, layer.offset, layer.idx)
                    )
                result.append(c)
            elif isinstance(layer, ArraysLayerState):
                c = ArraysCache(size=len(layer.arrays))
                c.state = [
                    _torch_to_mx(arr) if arr is not None else None
                    for arr in layer.arrays
                ]
                result.append(c)
            else:
                c = KVCache()
                if layer.keys.numel() > 0:
                    k_mx, v_mx = _nhd_to_bhsd(layer.keys, layer.values)
                    c.state = (k_mx, v_mx)
                result.append(c)
        return result

    @classmethod
    def from_vllm_cache(
        cls,
        kv_caches: list[torch.Tensor],
        block_ids: list[int],
        num_tokens: int,
    ) -> TorchKVCache:
        block_table = torch.tensor(block_ids, dtype=torch.long)

        layers: list[LayerState] = []
        for kv in kv_caches:
            k_all, v_all = _split_kv(kv)
            keys = k_all[block_table].flatten(0, 1)[:num_tokens].to("cpu", non_blocking=True)  # pyright: ignore[reportUnknownMemberType]
            values = v_all[block_table].flatten(0, 1)[:num_tokens].to("cpu", non_blocking=True)  # pyright: ignore[reportUnknownMemberType]
            layers.append(KVLayerState(keys=keys, values=values))

        torch.cuda.synchronize()
        return cls(layers)

    def write_to_vllm_blocks(
        self,
        kv_caches: list[torch.Tensor],
        block_ids: list[int],
    ) -> None:
        block_table = torch.tensor(block_ids, dtype=torch.long)

        device = kv_caches[0].device
        for layer_idx, layer in enumerate(self.layers):
            if not isinstance(layer, KVLayerState):
                continue
            kv = kv_caches[layer_idx]
            k_all, v_all = _split_kv(kv)
            num_tokens = layer.keys.shape[0]
            block_size = k_all.shape[1]
            n_full = num_tokens // block_size
            remainder = num_tokens % block_size
            if n_full > 0:
                k_all[block_table[:n_full]] = layer.keys[:n_full * block_size].view(n_full, block_size, *layer.keys.shape[1:]).to(device, non_blocking=True)
                v_all[block_table[:n_full]] = layer.values[:n_full * block_size].view(n_full, block_size, *layer.values.shape[1:]).to(device, non_blocking=True)
            if remainder > 0:
                k_all[block_table[n_full], :remainder] = layer.keys[n_full * block_size:].to(device, non_blocking=True)
                v_all[block_table[n_full], :remainder] = layer.values[n_full * block_size:].to(device, non_blocking=True)
        torch.cuda.synchronize()

    def __repr__(self) -> str:
        parts: list[str] = [f"TorchKVCache({self.num_layers} layers)"]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, KVLayerState):
                parts.append(f"  [{i}] KV: keys={list(layer.keys.shape)} values={list(layer.values.shape)} {layer.keys.dtype}")
            elif isinstance(layer, RotatingKVLayerState):
                parts.append(f"  [{i}] RotatingKV: keys={list(layer.keys.shape)} keep={layer.keep} max_size={layer.max_size} offset={layer.offset} idx={layer.idx}")
            else:
                shapes = [list(a.shape) if a is not None else None for a in layer.arrays]
                parts.append(f"  [{i}] Arrays: {shapes}")
        return "\n".join(parts)
