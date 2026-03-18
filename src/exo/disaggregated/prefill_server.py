from __future__ import annotations

import contextlib
import json
import queue
import socketserver
import threading
import time
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

from exo.worker.runner.bootstrap import logger

_engine_ref: LLMEngine | None = None
_overlapping: bool = True
_connector_patched: bool = False
_gdn_patched: bool = False
_gdn_states: dict[int, dict[str, torch.Tensor]] = {}
_gdn_layer_order: list[int] = []
_gdn_call_idx: list[int] = [0]


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
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn as orig_fn  # type: ignore
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc_mod  # type: ignore
    except ImportError:
        return

    def patched_fn(*args: Any, conv_states: Any = None, cache_indices: Any = None, **kwargs: Any) -> Any:
        result = orig_fn(*args, conv_states=conv_states, cache_indices=cache_indices, **kwargs)  # type: ignore
        if conv_states is not None and cache_indices is not None:
            x = args[0] if args else None
            if x is not None and x.shape[0] <= 100:  # type: ignore
                return result
            torch.cuda.synchronize()
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
        if hasattr(mod, "causal_conv1d_fn") and getattr(mod, "causal_conv1d_fn") is orig_fn:
            setattr(mod, "causal_conv1d_fn", patched_fn)
    logger.info("Patched causal_conv1d_fn for GDN state capture")


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


def _run_prefill_overlapping(engine: LLMEngine, token_ids: list[int], wfile: Any) -> None:  # pyright: ignore[reportAny]
    from exo.disaggregated.streaming_connector import StreamingConnector
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    model_runner = get_model_runner()
    assert model_runner is not None

    from exo.disaggregated.streaming_connector import get_shared_queue, reset_shared_queue

    reset_shared_queue()
    _gdn_states.clear()
    _gdn_call_idx[0] = 0
    layer_queue = get_shared_queue()

    num_layers, dtype_str, layers_info = _get_layer_info(engine)
    write_header(wfile, {"num_layers": num_layers, "dtype": dtype_str, "layers": layers_info})  # pyright: ignore[reportAny]

    from vllm.sampling_params import (
        SamplingParams,
    )

    request_id = f"prefill-{time.monotonic_ns()}"
    params = SamplingParams(max_tokens=1, detokenize=False)  # pyright: ignore[reportCallIssue]
    engine.add_request(request_id, {"prompt_token_ids": token_ids}, params)  # pyright: ignore[reportArgumentType]

    chunks_sent = [0]

    def writer_loop() -> None:
        while True:
            item = layer_queue.get()
            if item is None:
                break
            layer_idx, keys, values = item
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
    logger.info(f"Overlapping prefill: sent {chunks_sent[0]} KV chunks")

    _stream_gdn_states(engine, wfile, num_layers, layers_info)
    write_done(wfile, len(token_ids))  # pyright: ignore[reportAny]


def _run_prefill_batch(engine: LLMEngine, token_ids: list[int], wfile: Any) -> None:  # pyright: ignore[reportAny]
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    num_layers, dtype_str, layers_info = _get_layer_info(engine)

    model_runner = get_model_runner()
    assert model_runner is not None

    from exo.disaggregated.batch_connector import clear_shared_captured_layers, get_shared_captured_layers

    _gdn_states.clear()
    _gdn_call_idx[0] = 0
    clear_shared_captured_layers()
    captured_layers = get_shared_captured_layers()

    from vllm.sampling_params import (
        SamplingParams,
    )

    request_id = f"prefill-{time.monotonic_ns()}"
    params = SamplingParams(max_tokens=1, detokenize=False)  # pyright: ignore[reportCallIssue]
    engine.add_request(request_id, {"prompt_token_ids": token_ids}, params)  # pyright: ignore[reportArgumentType]

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

    logger.info(f"Batch prefill: streaming {len(captured_layers)} captured layers")
    for layer_idx in sorted(captured_layers.keys()):
        layer_data = captured_layers[layer_idx]
        write_kv_chunk(wfile, layer_idx, layer_data["keys"], layer_data["values"])  # pyright: ignore[reportAny]
    clear_shared_captured_layers()

    _stream_gdn_states(engine, wfile, num_layers, layers_info)
    write_done(wfile, len(token_ids))  # pyright: ignore[reportAny]


def _stream_gdn_states(_engine: LLMEngine, wfile: Any, num_layers: int, layers_info: list[dict[str, Any]]) -> None:  # type: ignore
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
            ci: int = state.get("ci", 0)  # type: ignore
            conv = state.get("conv")
            kv = kv_caches[layer_idx]  # type: ignore
            rec: torch.Tensor | None = None
            if isinstance(kv, (list, tuple)) and len(kv) > 1:
                rec = kv[1][ci : ci + 1].cpu().clone()  # type: ignore

            arrays: list[torch.Tensor] = []
            if conv is not None:
                arrays.append(conv.to(torch.bfloat16))
            if rec is not None:
                arrays.append(rec.to(torch.bfloat16))
            if arrays:
                write_arrays_state(wfile, layer_idx, arrays)  # type: ignore
        except Exception:
            logger.opt(exception=True).warning(f"Failed to capture GDN state for layer {layer_idx}")

    _gdn_states.clear()
    _gdn_call_idx[0] = 0


class _PrefillHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            if not line:
                return
            request: dict[str, Any] = json.loads(line.decode("utf-8"))  # pyright: ignore[reportAny]
            token_ids: list[int] = request["token_ids"]  # pyright: ignore[reportAny]

            engine = _engine_ref
            if engine is None:
                error = json.dumps({"error": "No engine loaded"}).encode("utf-8") + b"\n"
                self.wfile.write(error)
                return

            if engine.has_unfinished_requests():
                error = json.dumps({"error": "Engine busy"}).encode("utf-8") + b"\n"
                self.wfile.write(error)
                return

            logger.info(f"Prefill request: {len(token_ids)} tokens, overlapping={_overlapping}")
            t0 = time.perf_counter()

            if _overlapping:
                _run_prefill_overlapping(engine, token_ids, self.wfile)
            else:
                _run_prefill_batch(engine, token_ids, self.wfile)

            elapsed = time.perf_counter() - t0
            logger.info(f"Prefill complete: {len(token_ids)} tokens in {elapsed*1000:.0f}ms ({len(token_ids)/elapsed:.0f} tok/s)")
        except Exception:
            logger.opt(exception=True).error("Prefill handler error")


def start_prefill_server(
    engine: LLMEngine,
    bind_address: str,
    port: int,
    overlapping: bool = True,
) -> socketserver.ThreadingTCPServer:
    global _engine_ref, _overlapping
    _engine_ref = engine
    _overlapping = overlapping

    _patch_gdn_capture()
    _init_gdn_layer_order()

    server = socketserver.ThreadingTCPServer((bind_address, port), _PrefillHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Prefill TCP server started on {bind_address}:{port} (overlapping={overlapping})")
    return server
