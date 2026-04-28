# type: ignore
import contextlib
import queue
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from exo.worker.engines.vllm.disaggregated.adapter import (
    extract_kv_via_slot_mapping,
    to_bf16,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_LOG_PREFIX = "[exo-pd vllm-connector]"


# Module-level shared state. Populated by the connector's hooks (running inside
# vLLM's scheduler/worker, same process since V1 multiprocessing is off);
# drained by the producer engine in `serve_prefill` after the request finishes.
#
# `_kv_queue` is the original streaming-connector path ported into this module.
# We defer prefix reuse to vLLM APC and do not keep a separate TorchKVCache.
_kv_queue: "queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None]" = queue.Queue()
_arrays_queue: "queue.Queue[tuple[int, list[torch.Tensor]] | None]" = queue.Queue()
_captured_layers: dict[int, dict[str, torch.Tensor]] = {}
_captured_arrays: dict[int, list[torch.Tensor]] = {}
# Hybrid-model SSM/conv state captured via causal_conv1d + delta-rule patches.
_gdn_states: dict[int, dict[str, torch.Tensor]] = {}
_gdn_layer_order: list[int] = []
_gdn_call_idx: list[int] = [0]
_ssm_call_idx: list[int] = [0]
# Per-layer save_kv_layer call diagnostics: list of slot_mapping sizes seen.
_save_kv_layer_diag: dict[int, list[int]] = {}


def get_kv_queue() -> "queue.Queue[tuple[int, torch.Tensor, torch.Tensor] | None]":
    return _kv_queue


def get_arrays_queue() -> "queue.Queue[tuple[int, list[torch.Tensor]] | None]":
    return _arrays_queue


def get_gdn_states() -> dict[int, dict[str, torch.Tensor]]:
    return _gdn_states


def get_save_kv_layer_diag() -> dict[int, list[int]]:
    return _save_kv_layer_diag


def get_captured_layers() -> dict[int, dict[str, torch.Tensor]]:
    return _captured_layers


def get_captured_arrays() -> dict[int, list[torch.Tensor]]:
    return _captured_arrays


def reset_capture_state() -> None:
    while not _kv_queue.empty():
        try:
            _kv_queue.get_nowait()
        except queue.Empty:
            break
    while not _arrays_queue.empty():
        try:
            _arrays_queue.get_nowait()
        except queue.Empty:
            break
    _captured_layers.clear()
    _captured_arrays.clear()
    _gdn_states.clear()
    _gdn_call_idx[0] = 0
    _ssm_call_idx[0] = 0
    _save_kv_layer_diag.clear()


@dataclass
class StreamingConnectorMetadata(KVConnectorMetadata):
    pass


@dataclass
class BatchConnectorMetadata(KVConnectorMetadata):
    pass


class StreamingConnector(KVConnectorBase_V1, SupportsHMA):
    """Original streaming producer connector, kept under the new server abstraction."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._save_count = 0

    # =========================================================================
    # Worker-side hooks (the only ones we actually use)
    # =========================================================================

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        m = _LAYER_RE.search(layer_name)
        layer_idx_for_diag = int(m.group(1)) if m else -1
        slot_size = (
            int(slot_mapping.shape[0]) if slot_mapping is not None else -1
        )
        is_list_kv = isinstance(kv_layer, (list, tuple))
        # Tag list/tuple as negative so the diag log distinguishes hybrid from
        # non-hybrid even when slot_size is the same.
        _save_kv_layer_diag.setdefault(layer_idx_for_diag, []).append(
            -slot_size if is_list_kv else slot_size
        )

        # Skip decode-step saves (small slot mapping); we only want prefill.
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:
            return
        if m is None:
            return
        layer_idx = int(m.group(1))

        # Hybrid (Mamba+attention) layers: kv_layer is a list/tuple of state
        # tensors (conv + ssm). Send them straight to the arrays queue —
        # they don't live in the paged KV cache. Stay on GPU; the writer
        # thread does the D2H copy via `tensor_to_wire_bytes`.
        if isinstance(kv_layer, (list, tuple)):
            arrays = [to_bf16(t) for t in kv_layer]
            _arrays_queue.put((layer_idx, arrays))
            return

        # Standard attention layers (full or sliding-window): extract K/V
        # via slot_mapping, which points to where vLLM is *writing* this
        # forward step's tokens. Capturing here, before sliding-window
        # eviction in the block pool, is the only way to ship every prompt
        # token's K/V regardless of attention type.
        if slot_mapping is not None:
            try:
                keys, values = extract_kv_via_slot_mapping(kv_layer, slot_mapping)
            except Exception as exc:
                logger.warning(
                    f"save_kv_layer extract failed layer={layer_idx} "
                    f"kv_layer.shape={getattr(kv_layer, 'shape', None)} "
                    f"slot_mapping.shape={slot_mapping.shape}: {exc!r}"
                )
                return
            _kv_queue.put((layer_idx, keys, values))

    def wait_for_save(self) -> None:
        return

    # =========================================================================
    # Scheduler-side hooks (no-ops; we don't load and don't track allocs)
    # =========================================================================

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> StreamingConnectorMetadata:
        return StreamingConnectorMetadata()

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class BatchConnector(KVConnectorBase_V1, SupportsHMA):
    """Original batch producer connector, ported for parity with the old branch."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:
            return

        m = _LAYER_RE.search(layer_name)
        if m is None:
            return
        layer_idx = int(m.group(1))

        if isinstance(kv_layer, (list, tuple)):
            _captured_arrays[layer_idx] = [to_bf16(t).cpu() for t in kv_layer]
            return

        if slot_mapping is None:
            return
        keys, values = extract_kv_via_slot_mapping(kv_layer, slot_mapping)
        prev = _captured_layers.get(layer_idx)
        if prev is None:
            _captured_layers[layer_idx] = {"keys": keys, "values": values}
        else:
            _captured_layers[layer_idx] = {
                "keys": torch.cat([prev["keys"], keys], dim=0),
                "values": torch.cat([prev["values"], values], dim=0),
            }

    def wait_for_save(self) -> None:
        return

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> BatchConnectorMetadata:
        return BatchConnectorMetadata()


ExoKVProducerConnector = StreamingConnector


# =============================================================================
# Bypass patches — necessary to make our connector usable inside vLLM 1.x.
# Ported from the original branch's prefill_server.py:_patch_vllm_for_connector.
# =============================================================================

_connector_patched = False


def _patch_vllm_for_connector(connector_class: type[Any]) -> None:
    """Three patches that make a custom save-only connector cooperate with vLLM.

    1. Suppress `unify_hybrid_kv_cache_specs` ValueError on hybrid (Mamba +
       attention) models — the unifier complains about mixed cache specs we
       don't need to actually unify for save-only operation.
    2. Override `Scheduler._connector_finished` to short-circuit the
       async-save state machine. We're synchronous on the producer side.
    3. Make `KVConnectorFactory._get_connector_class_with_compat` recognize
       our class name and return our class directly, bypassing vLLM's
       registry of built-in connectors.
    """
    global _connector_patched
    if _connector_patched:
        return
    _connector_patched = True

    from vllm.v1.core import kv_cache_utils

    original_unify = kv_cache_utils.unify_hybrid_kv_cache_specs

    def patched_unify(kv_cache_spec: Any) -> None:
        with contextlib.suppress(ValueError):
            original_unify(kv_cache_spec)

    kv_cache_utils.unify_hybrid_kv_cache_specs = patched_unify

    from vllm.v1.core.sched import scheduler as sched_mod

    def patched_connector_finished(_self: Any, _request: Any) -> tuple[bool, Any]:
        return False, None

    sched_mod.Scheduler._connector_finished = patched_connector_finished

    from vllm.distributed.kv_transfer.kv_connector import factory

    original_get = factory.KVConnectorFactory._get_connector_class_with_compat

    @classmethod
    def patched_get(cls: Any, kv_transfer_config: Any) -> tuple[Any, Any]:
        kv_conn = getattr(kv_transfer_config, "kv_connector", None) or ""
        kv_conn_lower = kv_conn.lower()
        if kv_conn in {
            connector_class.__name__,
            f"{connector_class.__module__}:{connector_class.__name__}",
            "ExoKVProducerConnector",
            f"{__name__}:ExoKVProducerConnector",
            "StreamingConnector",
            f"{__name__}:StreamingConnector",
        } or "streaming_connector" in kv_conn_lower:
            return connector_class, None
        if "batch_connector" in kv_conn_lower:
            return BatchConnector, None
        return original_get.__func__(cls, kv_transfer_config)

    factory.KVConnectorFactory._get_connector_class_with_compat = patched_get
    logger.info("Installed vLLM connector bypass patches")


# =============================================================================
# Hybrid-model GDN state capture (Qwen3.5/3.6 etc.).
# Patches the conv1d kernel + delta-rule fns to grab conv/ssm states per layer.
# =============================================================================

_gdn_patched = False


def _patch_gdn_capture() -> None:
    global _gdn_patched
    if _gdn_patched:
        return
    _gdn_patched = True

    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc_mod
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn as orig_fn,
        )
    except ImportError:
        return

    def patched_fn(*args: Any, conv_states: Any = None, cache_indices: Any = None, **kwargs: Any) -> Any:
        result = orig_fn(*args, conv_states=conv_states, cache_indices=cache_indices, **kwargs)
        if conv_states is not None and cache_indices is not None:
            x = args[0] if args else None
            if x is not None and x.shape[0] <= 100:
                return result
            ci: int = cache_indices[0].item() if cache_indices.numel() > 0 else 0
            idx = _gdn_call_idx[0]
            if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                # `.contiguous()` decouples the slice from the underlying
                # buffer; D2H is deferred to the writer thread.
                conv_at_ci = conv_states[ci : ci + 1].transpose(-1, -2).contiguous()
                _gdn_states.setdefault(layer_idx, {})["conv"] = conv_at_ci
                _gdn_states[layer_idx]["ci"] = ci
            _gdn_call_idx[0] += 1
        return result

    cc_mod.causal_conv1d_fn = patched_fn
    import sys

    for mod in list(sys.modules.values()):
        if mod is None or mod is cc_mod:
            continue
        if hasattr(mod, "causal_conv1d_fn") and mod.causal_conv1d_fn is orig_fn:
            mod.causal_conv1d_fn = patched_fn
    logger.info("Patched causal_conv1d_fn for GDN conv-state capture")

    # The GDN delta-rule functions live in `mamba/gdn_linear_attn` (defined or
    # re-imported there) and may also be re-exported by model modules. Patch
    # all candidate modules + propagate to anywhere they're imported.
    candidate_modules = [
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "vllm.model_executor.models.qwen3_next",
        "vllm.model_executor.models.qwen3_5",
    ]
    fn_names = ("fi_chunk_gated_delta_rule", "fla_chunk_gated_delta_rule")
    patched_targets: list[str] = []

    for mod_path in candidate_modules:
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except ImportError:
            continue
        for fn_name in fn_names:
            orig = getattr(mod, fn_name, None)
            if orig is None:
                continue

            def make_patched(orig_fn_inner: Any) -> Any:
                def patched_chunk(*args: Any, **kwargs: Any) -> Any:
                    result = orig_fn_inner(*args, **kwargs)
                    output_final_state = kwargs.get("output_final_state", False)
                    if (
                        output_final_state
                        and isinstance(result, tuple)
                        and len(result) == 2
                    ):
                        _, ssm_state = result
                        idx = _ssm_call_idx[0]
                        if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                            layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                            _gdn_states.setdefault(layer_idx, {})["ssm"] = (
                                ssm_state
                            )
                        _ssm_call_idx[0] += 1
                    return result

                return patched_chunk

            patched_fn = make_patched(orig)
            setattr(mod, fn_name, patched_fn)
            patched_targets.append(f"{mod_path}.{fn_name}")
            # Propagate to any module that imported the original function.
            import sys as _sys

            for other in list(_sys.modules.values()):
                if other is None or other is mod:
                    continue
                if getattr(other, fn_name, None) is orig:
                    setattr(other, fn_name, patched_fn)
                    patched_targets.append(
                        f"{other.__name__}.{fn_name} (propagated)"
                    )
    if patched_targets:
        logger.info(f"Patched delta-rule fns for SSM capture: {patched_targets}")
    else:
        logger.warning(
            "GDN SSM-capture patch installed no targets — hybrid models may miss ssm state"
        )


def init_gdn_layer_order(kv_caches: Any) -> None:
    """Identify hybrid layers (those with list/tuple kv_cache entries)."""
    _gdn_layer_order.clear()
    for li in range(len(kv_caches)):
        kv = kv_caches[li]
        if isinstance(kv, (list, tuple)) and len(kv) > 1:
            _gdn_layer_order.append(li)
    if _gdn_layer_order:
        logger.info(
            f"GDN layer order: {len(_gdn_layer_order)} hybrid layers detected"
        )
