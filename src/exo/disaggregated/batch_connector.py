from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # pyright: ignore[reportMissingImports]
    KVConnectorBase_V1,  # pyright: ignore[reportUnknownVariableType]
    KVConnectorMetadata,  # pyright: ignore[reportUnknownVariableType]
    KVConnectorRole,  # pyright: ignore[reportUnknownVariableType]
    SupportsHMA,  # pyright: ignore[reportUnknownVariableType]
)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")

_shared_captured_layers: dict[int, dict[str, torch.Tensor]] = {}
_shared_captured_arrays: dict[int, list[torch.Tensor]] = {}


def get_shared_captured_layers() -> dict[int, dict[str, torch.Tensor]]:
    return _shared_captured_layers


def get_shared_captured_arrays() -> dict[int, list[torch.Tensor]]:
    return _shared_captured_arrays


def clear_shared_captured_layers() -> None:
    _shared_captured_layers.clear()
    _shared_captured_arrays.clear()


@dataclass
class BatchConnectorMetadata(KVConnectorMetadata):  # pyright: ignore[reportUntypedBaseClass]
    pass


class BatchConnector(KVConnectorBase_V1, SupportsHMA):  # pyright: ignore[reportUntypedBaseClass]
    captured_layers: dict[int, dict[str, torch.Tensor]]

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None) -> None:  # type: ignore
        super().__init__(vllm_config, role, kv_cache_config)  # pyright: ignore[reportUnknownMemberType]
        self.captured_layers = _shared_captured_layers

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:  # pyright: ignore[reportAny]
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:  # pyright: ignore[reportAny]
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)  # pyright: ignore[reportAny]
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:  # pyright: ignore[reportAny]
            return

        m = _LAYER_RE.search(layer_name)
        if m is None:
            return
        layer_idx = int(m.group(1))

        if isinstance(kv_layer, (list, tuple)):
            from exo.disaggregated.streaming_connector import _to_bf16

            _shared_captured_arrays[layer_idx] = [_to_bf16(t).cpu() for t in kv_layer]  # pyright: ignore[reportAny]
            return

        if slot_mapping is not None:
            if kv_layer.shape[0] == 2:  # pyright: ignore[reportAny]
                k_all = kv_layer[0]  # pyright: ignore[reportAny]
                v_all = kv_layer[1]  # pyright: ignore[reportAny]
            else:
                k_all = kv_layer[:, 0]  # pyright: ignore[reportAny]
                v_all = kv_layer[:, 1]  # pyright: ignore[reportAny]
            k_flat = k_all.reshape(-1, *k_all.shape[-2:])  # pyright: ignore[reportAny]
            v_flat = v_all.reshape(-1, *v_all.shape[-2:])  # pyright: ignore[reportAny]
            valid = slot_mapping >= 0  # pyright: ignore[reportAny]
            safe_sm = slot_mapping.clamp(min=0)  # pyright: ignore[reportAny]
            keys = k_flat[safe_sm][valid]  # pyright: ignore[reportAny]
            values = v_flat[safe_sm][valid]  # pyright: ignore[reportAny]
            from exo.disaggregated.streaming_connector import _to_bf16

            keys = _to_bf16(keys)  # pyright: ignore[reportAny]
            values = _to_bf16(values)  # pyright: ignore[reportAny]

            prev = self.captured_layers.get(layer_idx)
            if prev is not None:
                self.captured_layers[layer_idx] = {
                    "keys": torch.cat([prev["keys"], keys.cpu()], dim=0),  # type: ignore
                    "values": torch.cat([prev["values"], values.cpu()], dim=0),  # type: ignore
                }
            else:
                self.captured_layers[layer_idx] = {
                    "keys": keys.cpu(),  # pyright: ignore[reportAny]
                    "values": values.cpu(),  # pyright: ignore[reportAny]
                }

    def wait_for_save(self) -> None:
        pass

    def request_finished_all_groups(self, request: Any, block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:  # pyright: ignore[reportAny]
        return False, None

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int, bool]:  # pyright: ignore[reportAny]
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:  # pyright: ignore[reportAny]
        pass

    def build_connector_meta(self, scheduler_output: Any) -> BatchConnectorMetadata:  # pyright: ignore[reportAny]
        return BatchConnectorMetadata()
