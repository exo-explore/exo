from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)


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
        return torch.from_numpy(np.array(arr.astype(mx.float32))).to(torch.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    return torch.from_numpy(np.array(arr))  # pyright: ignore[reportUnknownMemberType]


def _torch_to_mx(t: torch.Tensor) -> mx.array:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.numpy())


def _split_kv(
    kv: torch.Tensor | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv, list):
        return kv[0], kv[1]
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
    def __init__(
        self, layers: list[LayerState], token_offset_per_group: list[int] | None = None
    ):
        self.layers = layers
        self.token_offset_per_group = token_offset_per_group or []
        self._num_tokens: int | None = None

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

    def detach_cpu(self) -> "TorchKVCache":
        layers: list[LayerState] = []
        for layer in self.layers:
            if isinstance(layer, KVLayerState):
                if not layer.keys.is_cuda:
                    layers.append(layer)
                else:
                    layers.append(
                        KVLayerState(
                            keys=layer.keys.detach().to("cpu", non_blocking=True),
                            values=layer.values.detach().to("cpu", non_blocking=True),
                        )
                    )
            elif isinstance(layer, RotatingKVLayerState):
                layers.append(
                    RotatingKVLayerState(
                        keys=layer.keys.detach().to("cpu", non_blocking=True),
                        values=layer.values.detach().to("cpu", non_blocking=True),
                        keep=layer.keep,
                        max_size=layer.max_size,
                        offset=layer.offset,
                        idx=layer.idx,
                    )
                )
            else:
                layers.append(deepcopy(layer))
        if any(
            layer.keys.is_cuda
            for layer in self.layers
            if isinstance(layer, (KVLayerState, RotatingKVLayerState))
        ):
            torch.cuda.synchronize()
        return TorchKVCache(layers, list(self.token_offset_per_group))

    def trim_to(self, num_tokens: int) -> "TorchKVCache":
        trimmed = TorchKVCache(list(self.layers), list(self.token_offset_per_group))
        trimmed._num_tokens = num_tokens
        return trimmed

    @property
    def num_tokens(self) -> int | None:
        return getattr(self, "_num_tokens", None)

    @classmethod
    def from_mlx_cache(
        cls,
        cache: Sequence[
            KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList
        ],
    ) -> "TorchKVCache":
        layers: list[LayerState] = []
        for c in cache:
            if isinstance(c, RotatingKVCache):
                if c.keys is None:
                    layers.append(
                        RotatingKVLayerState(
                            keys=torch.empty(0),
                            values=torch.empty(0),
                            keep=c.keep,
                            max_size=c.max_size,
                            offset=c.offset,
                            idx=c._idx,
                        )
                    )
                else:
                    k, v = c.state
                    kt, vt = _kv_to_nhd(k, v)  # pyright: ignore[reportArgumentType]
                    keep, max_size, offset, idx = (int(x) for x in c.meta_state)
                    layers.append(
                        RotatingKVLayerState(
                            keys=kt,
                            values=vt,
                            keep=keep,
                            max_size=max_size,
                            offset=offset,
                            idx=idx,
                        )
                    )
            elif isinstance(c, ArraysCache):
                arrays: list[torch.Tensor | None] = []
                for arr in c.state:
                    arrays.append(_mx_to_torch(arr) if arr is not None else None)
                layers.append(ArraysLayerState(arrays=arrays))
            else:
                if c.keys is None:
                    layers.append(
                        KVLayerState(keys=torch.empty(0), values=torch.empty(0))
                    )
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
                        str(x)
                        for x in (layer.keep, layer.max_size, layer.offset, layer.idx)
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
        kv_caches: list[torch.Tensor | list[torch.Tensor]],
        block_ids_per_group: list[list[int]],
        layer_to_group: list[int],
        num_tokens: int,
        token_offset_per_group: list[int] | None = None,
        block_sizes_per_group: list[int] | None = None,
    ) -> "TorchKVCache":
        block_tables = [
            torch.tensor(ids, dtype=torch.long) for ids in block_ids_per_group
        ]
        if token_offset_per_group is None:
            token_offset_per_group = [0] * len(block_ids_per_group)

        layers: list[LayerState] = []
        for layer_idx, kv in enumerate(kv_caches):
            gi = layer_to_group[layer_idx]
            bt = block_tables[gi]
            k_all, v_all = _split_kv(kv)

            if len(bt) == 0:
                layers.append(KVLayerState(keys=torch.empty(0), values=torch.empty(0)))
                continue

            if k_all.dim() >= 4 and len(bt) > 0 and block_sizes_per_group is not None:
                page_size = k_all.shape[1]
                sched_block_size = block_sizes_per_group[gi]
                pages_per_block = sched_block_size // page_size
                if pages_per_block > 1:
                    expanded: list[int] = []
                    block_indices = cast(list[int], bt.tolist())  # pyright: ignore[reportUnknownMemberType]
                    for b in block_indices:
                        start_page = b * pages_per_block
                        end_page = min(start_page + pages_per_block, k_all.shape[0])
                        expanded.extend(range(start_page, end_page))
                    bt = torch.tensor(expanded, dtype=torch.long)

            keys = k_all[bt].to("cpu", non_blocking=True)
            values = v_all[bt].to("cpu", non_blocking=True)
            torch.cuda.synchronize()
            layers.append(KVLayerState(keys=keys, values=values))
        return cls(layers, list(token_offset_per_group))

    def write_to_vllm_blocks(
        self,
        kv_caches: list[torch.Tensor | list[torch.Tensor]],
        block_ids_per_group: list[list[int]],
        layer_to_group: list[int],
        token_offset_per_group: list[int] | None = None,
    ) -> None:
        block_tables = [
            torch.tensor(ids, dtype=torch.long) for ids in block_ids_per_group
        ]

        first = kv_caches[0]
        device = first[0].device if isinstance(first, list) else first.device
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, ArraysLayerState):
                gi = layer_to_group[layer_idx]
                bt = block_tables[gi]
                kv = kv_caches[layer_idx]
                if isinstance(kv, list):
                    for stored, target in zip(layer.arrays, kv, strict=True):
                        if stored is not None:
                            n = min(len(bt), stored.shape[0])
                            if n > 0:
                                target[bt[:n]] = stored[:n].to(
                                    device, non_blocking=True
                                )
                continue
            if not isinstance(layer, KVLayerState):
                continue
            gi = layer_to_group[layer_idx]
            bt = block_tables[gi]
            kv = kv_caches[layer_idx]
            k_all, v_all = _split_kv(kv)

            keys = layer.keys
            values = layer.values
            block_size = k_all.shape[-3] if k_all.dim() >= 3 else k_all.shape[1]
            needs_reshape = keys.dim() == 3 and keys.shape[1:] != k_all.shape[1:]
            if needs_reshape:
                offset = token_offset_per_group[gi] if token_offset_per_group else 0
                if offset > 0:
                    keys = keys[offset:]
                    values = values[offset:]
                s, h, d = keys.shape
                pad = (block_size - s % block_size) % block_size
                if pad > 0:
                    keys = torch.nn.functional.pad(keys, (0, 0, 0, 0, 0, pad))
                    values = torch.nn.functional.pad(values, (0, 0, 0, 0, 0, pad))
                keys = keys.reshape(-1, block_size, h, d)
                values = values.reshape(-1, block_size, h, d)

            n_blocks = min(len(bt), keys.shape[0])
            if n_blocks > 0:
                k_all[bt[:n_blocks]] = keys[:n_blocks].to(device, non_blocking=True)
                v_all[bt[:n_blocks]] = values[:n_blocks].to(device, non_blocking=True)
        torch.cuda.synchronize()

    def __iter__(self) -> Iterator[LayerState]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        parts: list[str] = [f"TorchKVCache({self.num_layers} layers)"]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, KVLayerState):
                parts.append(
                    f"  [{i}] KV: keys={list(layer.keys.shape)} values={list(layer.values.shape)} {layer.keys.dtype}"
                )
            elif isinstance(layer, RotatingKVLayerState):
                parts.append(
                    f"  [{i}] RotatingKV: keys={list(layer.keys.shape)} keep={layer.keep} max_size={layer.max_size} offset={layer.offset} idx={layer.idx}"
                )
            else:
                shapes = [
                    list(a.shape) if a is not None else None for a in layer.arrays
                ]
                parts.append(f"  [{i}] Arrays: {shapes}")
        return "\n".join(parts)
