from __future__ import annotations

import contextlib
import json
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

    connector: StreamingConnector | None = None
    try:
        engine_core = engine.engine_core.engine_core  # type: ignore
        scheduler = engine_core.scheduler  # type: ignore
        kv_manager = scheduler.kv_cache_manager  # type: ignore
        connector_obj = getattr(kv_manager, "connector", None) or getattr(scheduler, "connector", None)  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(connector_obj, StreamingConnector):
            connector = connector_obj
    except Exception:
        pass

    if connector is None:
        logger.warning("Could not find StreamingConnector, falling back to non-overlapping")
        _run_prefill_batch(engine, token_ids, wfile)
        return

    num_layers, dtype_str, layers_info = _get_layer_info(engine)
    write_header(wfile, {"num_layers": num_layers, "dtype": dtype_str, "layers": layers_info})  # pyright: ignore[reportAny]

    from vllm.sampling_params import (
        SamplingParams,
    )

    request_id = f"prefill-{time.monotonic_ns()}"
    params = SamplingParams(max_tokens=1, detokenize=False)  # pyright: ignore[reportCallIssue]
    engine.add_request(request_id, {"prompt_token_ids": token_ids}, params)  # pyright: ignore[reportArgumentType]

    prefill_done = threading.Event()

    def engine_loop() -> None:
        while engine.has_unfinished_requests():
            outputs = engine.step()
            for output in outputs:
                if output.request_id == request_id and output.outputs[0].token_ids:
                    engine.abort_request([request_id])  # type: ignore
                    connector.finish()
                    prefill_done.set()
                    return
        connector.finish()
        prefill_done.set()

    engine_thread = threading.Thread(target=engine_loop, daemon=True)
    engine_thread.start()

    layer_queue = connector.layer_queue
    while True:
        item = layer_queue.get()
        if item is None:
            break
        layer_idx, keys, values = item
        write_kv_chunk(wfile, layer_idx, keys, values)  # pyright: ignore[reportAny]

    prefill_done.wait()
    engine_thread.join(timeout=5.0)

    _stream_gdn_states(engine, wfile, num_layers, layers_info)
    write_done(wfile, len(token_ids))  # pyright: ignore[reportAny]


def _run_prefill_batch(engine: LLMEngine, token_ids: list[int], wfile: Any) -> None:  # pyright: ignore[reportAny]
    from exo.disaggregated.batch_connector import BatchConnector
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    num_layers, dtype_str, layers_info = _get_layer_info(engine)

    model_runner = get_model_runner()
    assert model_runner is not None

    connector: BatchConnector | None = None
    try:
        engine_core = engine.engine_core.engine_core  # type: ignore
        scheduler = engine_core.scheduler  # type: ignore
        kv_manager = scheduler.kv_cache_manager  # type: ignore
        connector_obj = getattr(kv_manager, "connector", None) or getattr(scheduler, "connector", None)  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(connector_obj, BatchConnector):
            connector = connector_obj
    except Exception:
        pass

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

    if connector is not None:
        for layer_idx in sorted(connector.captured_layers.keys()):
            layer_data = connector.captured_layers[layer_idx]
            write_kv_chunk(wfile, layer_idx, layer_data["keys"], layer_data["values"])  # pyright: ignore[reportAny]

    _stream_gdn_states(engine, wfile, num_layers, layers_info)
    write_done(wfile, len(token_ids))  # pyright: ignore[reportAny]


def _stream_gdn_states(_engine: LLMEngine, wfile: Any, num_layers: int, layers_info: list[dict[str, Any]]) -> None:  # pyright: ignore[reportAny]
    from exo.worker.engines.vllm.growable_cache import get_model_runner

    model_runner = get_model_runner()
    if model_runner is None:
        return

    kv_caches = model_runner.kv_caches
    for li in range(num_layers):
        if li >= len(layers_info) or layers_info[li]["type"] != "arrays":
            continue
        kv = kv_caches[li]
        if not isinstance(kv, (list, tuple)) or len(kv) < 2:
            continue
        arrays: list[torch.Tensor] = []
        for pool in kv:  # pyright: ignore[reportUnknownVariableType]
            arrays.append(pool[0:1].cpu().clone())  # type: ignore
        write_arrays_state(wfile, li, arrays)  # pyright: ignore[reportAny]


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

    server = socketserver.ThreadingTCPServer((bind_address, port), _PrefillHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Prefill TCP server started on {bind_address}:{port} (overlapping={overlapping})")
    return server
