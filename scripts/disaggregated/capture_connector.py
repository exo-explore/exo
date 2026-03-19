"""Minimal KVConnector that captures per-layer cache data."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)

captured_layers: dict[str, Any] = {}


@dataclass
class CaptureMetadata(KVConnectorMetadata):
    pass


class CaptureConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        import time
        slot_mapping = getattr(attn_metadata, 'slot_mapping', None)
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:
            return
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t_sync = time.perf_counter() - t0
        if isinstance(kv_layer, (list, tuple)):
            captured_layers[layer_name] = [t.cpu().clone() for t in kv_layer]
        else:
            slot_mapping = getattr(attn_metadata, 'slot_mapping', None)
            if slot_mapping is not None:
                if kv_layer.shape[0] == 2:
                    k_all = kv_layer[0]
                    v_all = kv_layer[1]
                else:
                    k_all = kv_layer[:, 0]
                    v_all = kv_layer[:, 1]
                k_flat = k_all.reshape(-1, *k_all.shape[-2:])
                v_flat = v_all.reshape(-1, *v_all.shape[-2:])
                valid = slot_mapping >= 0
                safe_sm = slot_mapping.clamp(min=0)
                keys = k_flat[safe_sm]
                values = v_flat[safe_sm]
                keys[~valid] = 0
                values[~valid] = 0
                prev = captured_layers.get(layer_name)
                if isinstance(prev, dict) and "keys" in prev:
                    t1 = time.perf_counter()
                    captured_layers[layer_name] = {
                        "keys": torch.cat([prev["keys"], keys.cpu()], dim=0),
                        "values": torch.cat([prev["values"], values.cpu()], dim=0),
                    }
                    t_copy = time.perf_counter() - t1
                else:
                    t1 = time.perf_counter()
                    captured_layers[layer_name] = {
                        "keys": keys.cpu(),
                        "values": values.cpu(),
                    }
                    t_copy = time.perf_counter() - t1
                if "layers.3." in layer_name:
                    print(f"    [attn save] sync={t_sync*1000:.1f}ms copy={t_copy*1000:.1f}ms tokens={keys.shape[0]}")
            else:
                captured_layers[layer_name] = kv_layer.cpu().clone()

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return CaptureMetadata()
