from __future__ import annotations

import queue
import re
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # pyright: ignore[reportMissingImports]
    KVConnectorBase_V1,  # pyright: ignore[reportUnknownVariableType]
    KVConnectorMetadata,  # pyright: ignore[reportUnknownVariableType]
    KVConnectorRole,  # pyright: ignore[reportUnknownVariableType]
)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")

_shared_queue: queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None] = queue.Queue()


def get_shared_queue() -> queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None]:
    return _shared_queue


def reset_shared_queue() -> None:
    while not _shared_queue.empty():
        try:
            _shared_queue.get_nowait()
        except queue.Empty:
            break


@dataclass
class StreamingConnectorMetadata(KVConnectorMetadata):  # pyright: ignore[reportUntypedBaseClass]
    pass


class StreamingConnector(KVConnectorBase_V1):  # pyright: ignore[reportUntypedBaseClass]
    _queue: queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None]

    _save_count: int = 0

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None) -> None:  # type: ignore
        super().__init__(vllm_config, role, kv_cache_config)  # pyright: ignore[reportUnknownMemberType]
        self._queue = _shared_queue

    @property
    def layer_queue(self) -> queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None]:
        return self._queue

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
            return

        if self._save_count < 1:
            import logging
            logging.getLogger("exo").info(f"save_kv_layer: kv_layer.shape={kv_layer.shape} dtype={kv_layer.dtype} slot_mapping.shape={slot_mapping.shape if slot_mapping is not None else None}")  # pyright: ignore[reportAny]
            self._save_count += 1

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
            if keys.dtype not in (torch.bfloat16, torch.float16, torch.float32):  # pyright: ignore[reportAny]
                keys = keys.to(torch.bfloat16)  # pyright: ignore[reportAny]
                values = values.to(torch.bfloat16)  # pyright: ignore[reportAny]
            self._queue.put((layer_idx, keys.cpu(), values.cpu()))  # pyright: ignore[reportAny]
        else:
            self._queue.put((layer_idx, kv_layer.cpu().clone(), kv_layer.cpu().clone()))  # pyright: ignore[reportAny]

    def wait_for_save(self) -> None:
        pass

    def finish(self) -> None:
        self._queue.put(None)

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int, bool]:  # pyright: ignore[reportAny]
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:  # pyright: ignore[reportAny]
        pass

    def build_connector_meta(self, scheduler_output: Any) -> StreamingConnectorMetadata:  # pyright: ignore[reportAny]
        return StreamingConnectorMetadata()
