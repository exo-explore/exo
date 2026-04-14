from __future__ import annotations

import contextlib
import json
import socket
import socketserver
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from exo.disaggregated.protocol import (
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)

if TYPE_CHECKING:
    from vllm.v1.engine.llm_engine import LLMEngine

from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.vllm.kv_cache import KVLayerState, TorchKVCache
from exo.worker.runner.bootstrap import logger

_engine_ref: LLMEngine | None = None
_prefix_cache_ref: KVPrefixCache | None = None
_overlapping: bool = True
_on_status_change: Callable[[bool], None] | None = None
_connector_patched: bool = False
_gdn_patched: bool = False
_gdn_states: dict[int, dict[str, torch.Tensor]] = {}
_gdn_layer_order: list[int] = []
_gdn_call_idx: list[int] = [0]
_ssm_call_idx: list[int] = [0]


def _patch_vllm_for_connector(connector_class: type[Any]) -> None:  # pyright: ignore[reportUnusedFunction]
    global _connector_patched
    if _connector_patched:
        return
    _connector_patched = True

    from vllm.v1.core import kv_cache_utils

    original_unify = kv_cache_utils.unify_hybrid_kv_cache_specs  # type: ignore

    def patched_unify(kv_cache_spec: Any) -> None:  # pyright: ignore[reportAny]
        with contextlib.suppress(ValueError):
            original_unify(kv_cache_spec)

    kv_cache_utils.unify_hybrid_kv_cache_specs = patched_unify  # pyright: ignore[reportAttributeAccessIssue]

    from vllm.v1.core.sched import (  # pyright: ignore[reportMissingImports]
        scheduler as sched_mod,  # pyright: ignore[reportUnknownVariableType]
    )

    def patched_connector_finished(_self: Any, _request: Any) -> tuple[bool, Any]:  # pyright: ignore[reportAny]
        return False, None

    sched_mod.Scheduler._connector_finished = patched_connector_finished  # pyright: ignore[reportUnknownMemberType]

    from vllm.distributed.kv_transfer.kv_connector import (  # pyright: ignore[reportMissingImports]
        factory,  # pyright: ignore[reportUnknownVariableType]
    )

    original_get = factory.KVConnectorFactory._get_connector_class_with_compat  # type: ignore

    @classmethod
    def patched_get(cls: Any, kv_transfer_config: Any) -> tuple[Any, Any]:  # pyright: ignore[reportAny]
        kv_conn = getattr(kv_transfer_config, "kv_connector", None) or ""  # pyright: ignore[reportAny]
        if "streaming_connector" in kv_conn or "batch_connector" in kv_conn:
            return connector_class, None
        return original_get.__func__(cls, kv_transfer_config)  # type: ignore

    factory.KVConnectorFactory._get_connector_class_with_compat = patched_get  # pyright: ignore[reportUnknownMemberType]


def _patch_gdn_capture() -> None:
    global _gdn_patched
    if _gdn_patched:
        return
    _gdn_patched = True

    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc_mod  # type: ignore
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn as orig_fn,  # type: ignore
        )
    except ImportError:
        return

    def patched_fn(*args: Any, conv_states: Any = None, cache_indices: Any = None, **kwargs: Any) -> Any:
        result = orig_fn(*args, conv_states=conv_states, cache_indices=cache_indices, **kwargs)  # type: ignore
        if conv_states is not None and cache_indices is not None:
            x = args[0] if args else None
            if x is not None and x.shape[0] <= 100:  # type: ignore
                return result
            ci: int = cache_indices[0].item() if cache_indices.numel() > 0 else 0  # type: ignore
            idx = _gdn_call_idx[0]
            if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                conv_at_ci = conv_states[ci : ci + 1].transpose(-1, -2).contiguous().cpu()  # type: ignore
                _gdn_states.setdefault(layer_idx, {})["conv"] = conv_at_ci
                _gdn_states[layer_idx]["ci"] = ci  # type: ignore
            _gdn_call_idx[0] += 1
        return result

    cc_mod.causal_conv1d_fn = patched_fn  # type: ignore
    import sys
    for mod in list(sys.modules.values()):
        if mod is None or mod is cc_mod:
            continue
        if hasattr(mod, "causal_conv1d_fn") and mod.causal_conv1d_fn is orig_fn:
            mod.causal_conv1d_fn = patched_fn
    logger.info("Patched causal_conv1d_fn for GDN state capture")

    try:
        from vllm.model_executor.models import qwen3_next as qn_mod  # type: ignore

        orig_chunk = getattr(qn_mod, "fi_chunk_gated_delta_rule", None)  # type: ignore
        if orig_chunk is None:
            return

        def patched_chunk(*args: Any, **kwargs: Any) -> Any:
            result = orig_chunk(*args, **kwargs)
            output_final_state = kwargs.get("output_final_state", False)
            if output_final_state and isinstance(result, tuple) and len(result) == 2:
                _, ssm_state = result
                idx = _ssm_call_idx[0]
                if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                    layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                    _gdn_states.setdefault(layer_idx, {})["ssm"] = ssm_state.cpu()  # type: ignore
                _ssm_call_idx[0] += 1
            return result

        qn_mod.fi_chunk_gated_delta_rule = patched_chunk  # type: ignore

        orig_fla_chunk = getattr(qn_mod, "fla_chunk_gated_delta_rule", None)  # type: ignore
        if orig_fla_chunk is not None:
            def patched_fla_chunk(*args: Any, **kwargs: Any) -> Any:
                result = orig_fla_chunk(*args, **kwargs)
                output_final_state = kwargs.get("output_final_state", False)
                if output_final_state and isinstance(result, tuple) and len(result) == 2:
                    _, ssm_state = result
                    idx = _ssm_call_idx[0]
                    if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                        layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                        _gdn_states.setdefault(layer_idx, {})["ssm"] = ssm_state.cpu()  # type: ignore
                    _ssm_call_idx[0] += 1
                return result
            qn_mod.fla_chunk_gated_delta_rule = patched_fla_chunk  # type: ignore

        logger.info("Patched chunk_gated_delta_rule for SSM state capture")
    except ImportError:
        pass


def _init_gdn_layer_order() -> None:
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    model_runner = get_model_runner()
    if model_runner is None:
        return
    kv_caches = model_runner.kv_caches  # type: ignore
    _gdn_layer_order.clear()
    for li in range(len(kv_caches)):  # type: ignore
        kv = kv_caches[li]  # type: ignore
        if isinstance(kv, (list, tuple)) and len(kv) > 1:
            _gdn_layer_order.append(li)
    if _gdn_layer_order:
        logger.info(f"GDN layer order: {_gdn_layer_order} ({len(_gdn_layer_order)} layers)")


def _get_layer_info(engine: LLMEngine) -> tuple[int, str, list[dict[str, Any]]]:
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    model_runner = get_model_runner()
    assert model_runner is not None
    kv_caches = model_runner.kv_caches
    num_layers: int = len(kv_caches)

    layers_info: list[dict[str, Any]] = []
    for li in range(num_layers):
        kv = kv_caches[li]
        if isinstance(kv, (list, tuple)) and len(kv) > 1:
            layers_info.append({"type": "arrays", "sizes": [2]})
        else:
            sample = kv[0] if isinstance(kv, (list, tuple)) else kv
            n_heads: int = sample.shape[-2]
            head_dim: int = sample.shape[-1]
            layers_info.append({"type": "kv", "n_heads": n_heads, "head_dim": head_dim})

    dtype_str = "bfloat16"
    return num_layers, dtype_str, layers_info


def _run_prefill_overlapping(engine: LLMEngine, token_ids: list[int], start_pos: int, wfile: Any) -> None:  # pyright: ignore[reportAny]
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    model_runner = get_model_runner()
    assert model_runner is not None

    from exo.disaggregated.streaming_connector import (
        get_shared_arrays_queue,
        get_shared_queue,
        reset_shared_queue,
    )

    reset_shared_queue()
    _gdn_states.clear()
    _gdn_call_idx[0] = 0
    _ssm_call_idx[0] = 0
    layer_queue = get_shared_queue()
    arrays_queue = get_shared_arrays_queue()

    server_cached = 0
    cached_data: TorchKVCache | None = None
    if _prefix_cache_ref is not None:
        cached_data, server_cached, _ = _prefix_cache_ref.lookup(token_ids)
        if not isinstance(cached_data, TorchKVCache):
            cached_data = None
            server_cached = 0
    skip_tokens = max(0, start_pos - server_cached)

    num_layers, dtype_str, layers_info = _get_layer_info(engine)
    write_header(wfile, {"num_layers": num_layers, "dtype": dtype_str, "layers": layers_info})  # pyright: ignore[reportAny]

    if cached_data is not None and start_pos < server_cached:
        from exo.worker.engines.vllm.kv_cache import ArraysLayerState
        kv_sent = 0
        arr_sent = 0
        for i, layer in enumerate(cached_data.layers):
            if isinstance(layer, KVLayerState) and layer.keys.numel() > 0:
                keys = layer.keys
                values = layer.values
                if keys.shape != values.shape:
                    logger.warning(f"Skipping layer {i}: keys={list(keys.shape)} != values={list(values.shape)}")
                    continue
                if keys.dim() == 4:
                    keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
                    values = values.reshape(-1, values.shape[-2], values.shape[-1])
                keys = keys[start_pos:server_cached]
                values = values[start_pos:server_cached]
                if keys.numel() > 0:
                    write_kv_chunk(wfile, i, keys, values)  # pyright: ignore[reportAny]
                    kv_sent += 1

            elif isinstance(layer, ArraysLayerState):
                arrays = [a for a in layer.arrays if a is not None]
                if arrays:
                    write_arrays_state(wfile, i, arrays)  # pyright: ignore[reportAny]
                    arr_sent += 1
        logger.info(f"Sent cached: {kv_sent} KV, {arr_sent} arrays for positions {start_pos}-{server_cached}")

    from vllm.sampling_params import (
        SamplingParams,
    )

    prefill_token_ids = token_ids[:-2] if len(token_ids) > 2 else token_ids
    request_id = f"prefill-{time.monotonic_ns()}"
    params = SamplingParams(max_tokens=2, detokenize=False)  # pyright: ignore[reportCallIssue]
    engine.add_request(request_id, {"prompt_token_ids": prefill_token_ids}, params)  # pyright: ignore[reportArgumentType]

    chunks_sent = [0]
    layer_token_counts: dict[int, int] = {}
    all_kv_chunks: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def writer_loop() -> None:
        while True:
            item = layer_queue.get()
            if item is None:
                break
            layer_idx, keys, values = item
            all_kv_chunks.append((layer_idx, keys, values))

            prev = layer_token_counts.get(layer_idx, 0)
            n = keys.shape[0]
            new_total = prev + n
            layer_token_counts[layer_idx] = new_total

            if new_total <= skip_tokens:
                continue
            if prev < skip_tokens:
                trim = skip_tokens - prev
                keys = keys[trim:]
                values = values[trim:]
            if chunks_sent[0] == 0:
                logger.info(f"First KV chunk: layer={layer_idx} keys={keys.shape} keys.dtype={keys.dtype} values.dtype={values.dtype}")
            write_kv_chunk(wfile, layer_idx, keys, values)  # pyright: ignore[reportAny]
            chunks_sent[0] += 1

    writer_thread = threading.Thread(target=writer_loop, daemon=True)
    writer_thread.start()

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.request_id == request_id and output.outputs[0].token_ids:
                engine.abort_request([request_id])  # type: ignore
                break
        else:
            continue
        break

    layer_queue.put(None)
    writer_thread.join()
    actual_per_layer = max(layer_token_counts.values()) if layer_token_counts else 0
    cached_tokens_sent = max(0, server_cached - start_pos) if cached_data is not None and start_pos < server_cached else 0
    tokens_sent = cached_tokens_sent + max(0, actual_per_layer - skip_tokens)
    logger.info(f"Overlapping prefill: sent {chunks_sent[0]} chunks, {tokens_sent} tokens (server_cached={server_cached}, skip={skip_tokens})")

    while not arrays_queue.empty():
        item = arrays_queue.get_nowait()
        if item is not None:
            layer_idx, arrays = item
            write_arrays_state(wfile, layer_idx, arrays)  # pyright: ignore[reportAny]

    gdn_snapshot: list[tuple[int, list[torch.Tensor]]] = []
    for layer_idx in sorted(_gdn_states.keys()):
        state = _gdn_states[layer_idx]
        arrs: list[torch.Tensor] = []
        if "conv" in state:
            arrs.append(state["conv"])
        if "ssm" in state:
            arrs.append(state["ssm"])
        if arrs:
            gdn_snapshot.append((layer_idx, arrs))

    cached_arrays: list[tuple[int, list[torch.Tensor]]] = []
    _stream_gdn_states_and_collect(engine, wfile, num_layers, layers_info, cached_arrays)
    write_done(wfile, tokens_sent)  # pyright: ignore[reportAny]

    connector_cache = _build_torch_cache(all_kv_chunks, gdn_snapshot, num_layers)
    threading.Thread(target=_store_prefix_cache, args=(prefill_token_ids, connector_cache), daemon=True).start()


def _run_prefill_batch(engine: LLMEngine, token_ids: list[int], start_pos: int, wfile: Any) -> None:  # pyright: ignore[reportAny]
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    num_layers, dtype_str, layers_info = _get_layer_info(engine)

    model_runner = get_model_runner()
    assert model_runner is not None

    from exo.disaggregated.batch_connector import (
        clear_shared_captured_layers,
        get_shared_captured_arrays,
        get_shared_captured_layers,
    )

    _gdn_states.clear()
    _gdn_call_idx[0] = 0
    clear_shared_captured_layers()
    captured_layers = get_shared_captured_layers()
    captured_arrays = get_shared_captured_arrays()

    server_cached = 0
    if _prefix_cache_ref is not None:
        _, server_cached, _ = _prefix_cache_ref.lookup(token_ids)
    skip_tokens = max(0, start_pos - server_cached)

    from vllm.sampling_params import (
        SamplingParams,
    )

    prefill_token_ids = token_ids[:-2] if len(token_ids) > 2 else token_ids
    request_id = f"prefill-{time.monotonic_ns()}"
    params = SamplingParams(max_tokens=2, detokenize=False)  # pyright: ignore[reportCallIssue]
    engine.add_request(request_id, {"prompt_token_ids": prefill_token_ids}, params)  # pyright: ignore[reportArgumentType]

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.request_id == request_id and output.outputs[0].token_ids:
                engine.abort_request([request_id])  # type: ignore
                break
        else:
            continue
        break

    write_header(wfile, {"num_layers": num_layers, "dtype": dtype_str, "layers": layers_info})  # pyright: ignore[reportAny]

    all_kv: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for layer_idx in sorted(captured_layers.keys()):
        layer_data = captured_layers[layer_idx]
        keys = layer_data["keys"]
        values = layer_data["values"]
        all_kv.append((layer_idx, keys, values))
        if keys.shape[0] > skip_tokens:
            write_kv_chunk(wfile, layer_idx, keys[skip_tokens:], values[skip_tokens:])  # pyright: ignore[reportAny]

    actual_per_layer = max((k.shape[0] for _, k, _ in all_kv), default=0)
    tokens_sent = max(0, actual_per_layer - skip_tokens)
    logger.info(f"Batch prefill: {len(all_kv)} layers, {tokens_sent} tokens sent (server_cached={server_cached}, skip={skip_tokens}, captured={actual_per_layer})")

    batch_arrays: list[tuple[int, list[torch.Tensor]]] = list(captured_arrays.items())
    for layer_idx, arrs in batch_arrays:
        write_arrays_state(wfile, layer_idx, arrs)  # pyright: ignore[reportAny]
    clear_shared_captured_layers()

    cached_arrays: list[tuple[int, list[torch.Tensor]]] = []
    _stream_gdn_states_and_collect(engine, wfile, num_layers, layers_info, cached_arrays)
    write_done(wfile, tokens_sent)  # pyright: ignore[reportAny]

    connector_cache = _build_torch_cache(all_kv, batch_arrays, num_layers)
    threading.Thread(target=_store_prefix_cache, args=(prefill_token_ids, connector_cache), daemon=True).start()


def _stream_gdn_states_and_collect(
    _engine: LLMEngine,
    wfile: Any,
    num_layers: int,
    layers_info: list[dict[str, Any]],
    out_arrays: list[tuple[int, list[torch.Tensor]]],
) -> None:  # type: ignore
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    if not _gdn_states:
        return

    model_runner = get_model_runner()
    if model_runner is None:
        return

    kv_caches = model_runner.kv_caches  # type: ignore
    torch.cuda.synchronize()

    for layer_idx in sorted(_gdn_states.keys()):
        try:
            state = _gdn_states[layer_idx]
            conv = state.get("conv")
            ssm = state.get("ssm")

            arrays: list[torch.Tensor] = []
            if conv is not None:
                arrays.append(conv)
            if ssm is not None:
                arrays.append(ssm)
            if arrays:
                write_arrays_state(wfile, layer_idx, arrays)  # type: ignore
                out_arrays.append((layer_idx, arrays))
        except Exception:
            logger.opt(exception=True).warning(f"Failed to capture GDN state for layer {layer_idx}")

    _gdn_states.clear()
    _gdn_call_idx[0] = 0


def _build_torch_cache(kv_chunks: list[tuple[int, torch.Tensor, torch.Tensor]], arrays_chunks: list[tuple[int, list[torch.Tensor]]], num_layers: int) -> TorchKVCache:
    from exo.worker.engines.vllm.kv_cache import ArraysLayerState

    layers_by_idx: dict[int, KVLayerState | ArraysLayerState] = {}
    for layer_idx, keys, values in kv_chunks:
        if layer_idx in layers_by_idx:
            prev = layers_by_idx[layer_idx]
            if isinstance(prev, KVLayerState):
                layers_by_idx[layer_idx] = KVLayerState(
                    keys=torch.cat([prev.keys, keys], dim=0),  # type: ignore
                    values=torch.cat([prev.values, values], dim=0),  # type: ignore
                )
        else:
            layers_by_idx[layer_idx] = KVLayerState(keys=keys, values=values)
    for layer_idx, arrays in arrays_chunks:
        layers_by_idx[layer_idx] = ArraysLayerState(arrays=[a if isinstance(a, torch.Tensor) else None for a in arrays])

    ordered: list[KVLayerState | ArraysLayerState] = []
    for i in range(num_layers):
        if i in layers_by_idx:
            ordered.append(layers_by_idx[i])
        else:
            ordered.append(KVLayerState(keys=torch.empty(0), values=torch.empty(0)))
    return TorchKVCache(ordered)


def _extract_vllm_cache(engine: LLMEngine, request_id: str, num_tokens: int) -> TorchKVCache | None:
    try:
        from exo.worker.engines.vllm.growable_cache import get_model_runner
        from exo.worker.engines.vllm.vllm_generator import _build_layer_groups
        model_runner = get_model_runner()
        if model_runner is None:
            return None
        engine_core = engine.engine_core.engine_core  # type: ignore
        coordinator = engine_core.scheduler.kv_cache_manager.coordinator  # type: ignore
        kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config  # type: ignore

        internal_id: str | None = None
        for mgr in coordinator.single_type_managers:  # type: ignore
            for key in mgr.req_to_blocks:  # type: ignore
                if str(key).startswith(request_id):  # type: ignore
                    internal_id = str(key)  # type: ignore
                    break
            if internal_id:
                break
        if internal_id is None:
            return None

        null_block = coordinator.block_pool.null_block  # type: ignore
        block_ids_per_group: list[list[int]] = []
        token_offset_per_group: list[int] = []
        block_sizes_per_group: list[int] = []
        for mgr in coordinator.single_type_managers:  # type: ignore
            blocks = mgr.req_to_blocks.get(internal_id)  # type: ignore
            if not blocks:
                block_ids_per_group.append([])
                token_offset_per_group.append(0)
                block_sizes_per_group.append(0)
                continue
            block_size: int = mgr.block_size  # type: ignore
            block_sizes_per_group.append(block_size)
            num_leading_nulls = 0
            for b in blocks:  # type: ignore
                if b is null_block or b.is_null:  # type: ignore
                    num_leading_nulls += 1
                else:
                    break
            real_blocks = [b for b in blocks if b is not null_block and not b.is_null]  # type: ignore
            block_ids_per_group.append([b.block_id for b in real_blocks])  # type: ignore
            token_offset_per_group.append(num_leading_nulls * block_size)

        layer_to_group = _build_layer_groups(kv_cache_config)
        return TorchKVCache.from_vllm_cache(
            model_runner.kv_caches,  # type: ignore
            block_ids_per_group,
            layer_to_group,
            num_tokens,
            token_offset_per_group,
            block_sizes_per_group,
        )
    except Exception:
        logger.opt(exception=True).warning("Failed to extract vLLM cache")
        return None


def _store_prefix_cache(token_ids: list[int], torch_cache: TorchKVCache) -> None:
    if _prefix_cache_ref is None:
        return
    try:
        before = len(_prefix_cache_ref.prompts)
        _prefix_cache_ref.add_from_torch(token_ids, torch_cache)
        after = len(_prefix_cache_ref.prompts)
        if after > before:
            logger.info(f"Server prefix cache: saved {len(token_ids)} tokens (entries: {before} → {after})")
    except Exception:
        logger.opt(exception=True).warning("Failed to store prefix cache")


def _check_cache(token_ids: list[int]) -> TorchKVCache | None:
    if _prefix_cache_ref is None:
        return None
    import mlx.core as mx

    prompt_arr = mx.array(token_ids)
    best_index: int | None = None
    best_length = 0
    for i, cached_prompt in enumerate(_prefix_cache_ref.prompts):
        prefix_len = min(len(cached_prompt), len(prompt_arr))
        if prefix_len == 0:
            continue
        match_len = int(mx.sum(cached_prompt[:prefix_len] == prompt_arr[:prefix_len]).item())  # pyright: ignore[reportAny]
        if match_len == len(token_ids) and match_len == len(cached_prompt) and match_len > best_length:
            best_index = i
            best_length = match_len

    if best_index is None:
        return None

    cached = _prefix_cache_ref.caches[best_index]
    if isinstance(cached, TorchKVCache):
        return cached
    return None


def _send_cached(torch_cache: TorchKVCache, token_ids: list[int], wfile: Any, engine: LLMEngine) -> None:
    num_layers, dtype_str, layers_info = _get_layer_info(engine)
    write_header(wfile, {"num_layers": num_layers, "dtype": dtype_str, "layers": layers_info})  # type: ignore
    from exo.worker.engines.vllm.kv_cache import ArraysLayerState

    kv_sent = 0
    arr_sent = 0
    for i, layer in enumerate(torch_cache.layers):
        if isinstance(layer, KVLayerState) and layer.keys.numel() > 0:
            write_kv_chunk(wfile, i, layer.keys, layer.values)  # type: ignore
            kv_sent += 1
        elif isinstance(layer, ArraysLayerState):
            arrays = [a for a in layer.arrays if a is not None]
            if arrays:
                write_arrays_state(wfile, i, arrays)  # type: ignore
                arr_sent += 1
    logger.info(f"_send_cached: sent {kv_sent} KV layers, {arr_sent} arrays layers")
    write_done(wfile, len(token_ids))  # type: ignore


class _PrefillHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # type: ignore
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)  # type: ignore

    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            if not line:
                return
            request: dict[str, Any] = json.loads(line.decode("utf-8"))  # pyright: ignore[reportAny]
            token_ids: list[int] = request["token_ids"]  # pyright: ignore[reportAny]
            start_pos: int = request.get("start_pos", 0)  # pyright: ignore[reportAny]

            engine = _engine_ref
            if engine is None:
                error = json.dumps({"error": "No engine loaded"}).encode("utf-8") + b"\n"
                self.wfile.write(error)
                return

            if engine.has_unfinished_requests():
                error = json.dumps({"error": "Engine busy"}).encode("utf-8") + b"\n"
                self.wfile.write(error)
                return

            logger.info(f"Prefill request: {len(token_ids)} tokens, start_pos={start_pos}, overlapping={_overlapping}")
            t0 = time.perf_counter()

            if _on_status_change:
                _on_status_change(True)
            try:
                if _overlapping:
                    _run_prefill_overlapping(engine, token_ids, start_pos, self.wfile)
                else:
                    _run_prefill_batch(engine, token_ids, start_pos, self.wfile)
            finally:
                if _on_status_change:
                    _on_status_change(False)

            elapsed = time.perf_counter() - t0
            logger.info(f"Prefill complete: {len(token_ids)} tokens in {elapsed*1000:.0f}ms ({len(token_ids)/elapsed:.0f} tok/s)")
        except Exception:
            logger.opt(exception=True).error("Prefill handler error")


def start_prefill_server(
    engine: LLMEngine,
    bind_address: str,
    port: int,
    overlapping: bool = True,
    prefix_cache: KVPrefixCache | None = None,
    on_status_change: Callable[[bool], None] | None = None,
) -> socketserver.ThreadingTCPServer:
    global _engine_ref, _overlapping, _prefix_cache_ref, _on_status_change
    _engine_ref = engine
    _overlapping = overlapping
    _prefix_cache_ref = prefix_cache
    _on_status_change = on_status_change

    _patch_gdn_capture()
    _init_gdn_layer_order()

    server = socketserver.ThreadingTCPServer((bind_address, port), _PrefillHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Prefill TCP server started on {bind_address}:{port} (overlapping={overlapping})")
    return server
