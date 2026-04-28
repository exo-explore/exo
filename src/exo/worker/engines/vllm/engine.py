import contextlib
import itertools
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.vllm.generator import VllmBatchEngine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.model_output_parsers import (
    apply_all_parsers,
    map_responses_to_chunks,
)
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


class GeneratorQueue[T]:
    def __init__(self) -> None:
        self._q = deque[T]()

    def push(self, t: T) -> None:
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Keep the cheap debug prompt hooks without importing the MLX engine."""
    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class VllmEngine(Engine):
    """Single-node vLLM implementation of the exo Engine interface.

    This intentionally duplicates the local orchestration from the MLX
    BatchGenerator instead of trying to share a batch abstraction too early.
    The vLLM-specific tokenization/sampling/stepping remains inside
    VllmBatchEngine.
    """

    tool_parser: ToolParser | None
    model_id: ModelId
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    _gen: VllmBatchEngine
    max_concurrent_requests: int = EXO_MAX_CONCURRENT_REQUESTS
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active_tasks: dict[
        TaskId,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)

    def warmup(self) -> None:
        self.check_for_cancel_every = self._gen.warmup()

    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._queue.append(task)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        self._collect_cancellations()
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = list(self._apply_cancellations())

        while self._queue and len(self._active_tasks) < self.max_concurrent_requests:
            task = self._queue.popleft()
            if self.should_cancel(task.task_id):
                output.append((task.task_id, CancelledResponse()))
                self._all_tasks.pop(task.task_id, None)
                continue

            try:
                task_id, queue, output_generator = self._start_task(task)
            except Exception as e:
                self._send_error(task, e)
                self._all_tasks.pop(task.task_id, None)
                raise

            self._active_tasks[task_id] = (task, queue, output_generator)

        if not self._gen.has_work:
            return iter(output)

        results = self._gen.step()
        for task_id, response in results:
            if task_id not in self._active_tasks:
                logger.warning(f"{task_id=} not found in active vLLM tasks")
                continue

            task, queue, output_generator = self._active_tasks[task_id]
            queue.push(response)
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            if response.finish_reason is not None:
                output.append((task.task_id, FinishedResponse()))
                del self._active_tasks[task_id]
                self._all_tasks.pop(task.task_id, None)

        return itertools.chain(output, self._apply_cancellations())

    def close(self) -> None:
        self._gen.close()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        import threading
        import time

        import torch
        from vllm import SamplingParams

        from exo.worker.disaggregated.protocol import write_error, write_kv_chunk
        from exo.worker.engines.vllm.disaggregated.adapter import (
            arrays_to_blobs,
            tensor_to_wire_bytes,
            torch_dtype_to_wire,
            write_layer_arrays_blobs,
            write_prefill_done,
            write_prefill_header,
        )
        from exo.worker.engines.vllm.growable_cache import get_model_runner
        from exo.worker.engines.vllm.kv_connector import (
            get_arrays_queue,
            get_gdn_shipped,
            get_gdn_states,
            get_kv_queue,
            get_save_kv_layer_diag,
            init_gdn_layer_order,
            reset_capture_state,
        )

        engine = self._gen.engine
        if engine.has_unfinished_requests():
            logger.warning("serve_prefill: engine busy, refusing prefill request")
            write_error(wfile, code=503, message="engine busy")
            return

        model_runner = get_model_runner()
        if model_runner is None:
            logger.warning("serve_prefill: model runner not initialized")
            write_error(wfile, code=503, message="model runner not initialized")
            return

        init_gdn_layer_order(model_runner.kv_caches)

        prefill_token_ids = (
            request.token_ids[:-2]
            if len(request.token_ids) > 2
            else list(request.token_ids)
        )
        n_layers = len(model_runner.kv_caches)

        reset_capture_state()
        arrays_queue = get_arrays_queue()
        kv_queue = get_kv_queue()

        # vLLM APC is disabled at engine-creation time (enable_prefix_caching=False
        # in generator.py). Cross-request prefix reuse on the consumer (MLX) side
        # is handled by its own kv_prefix_cache; the vLLM-side server here is
        # stateless across requests by design. We still skip the trailing 2
        # tokens because the consumer warm-starts decode from them.
        sp = SamplingParams(max_tokens=2, temperature=0.0, detokenize=False)
        engine.add_request(
            request.request_id,
            {"prompt_token_ids": prefill_token_ids},
            sp,
        )

        write_prefill_header(
            wfile,
            request_id=request.request_id,
            model_id=request.model_id,
            num_layers=n_layers,
            start_pos=request.start_pos,
        )

        # Consumer already has KV for [0, request.start_pos); drop those tokens
        # from the post-forward stream so we only ship the suffix it needs.
        skip_tokens = request.start_pos
        chunks_sent = [0]
        arrays_streamed = [0]
        layer_token_counts: dict[int, int] = {}
        # Both writer threads serialize through this lock — BufferedWriter
        # is not thread-safe and we don't want partial frames interleaved.
        wfile_lock = threading.Lock()
        # Diag for end-of-request bandwidth report.
        writer_stats = {
            "bytes_shipped": 0,
            "wait_event_secs": 0.0,
            "socket_secs": 0.0,
            "first_byte_t": 0.0,
            "last_byte_t": 0.0,
            "started_t": 0.0,
        }

        def writer_loop() -> None:
            writer_stats["started_t"] = time.perf_counter()
            try:
                while True:
                    item = kv_queue.get()
                    if item is None:
                        break
                    layer_idx, keys, values, copy_event = item
                    # Wait for the side-stream D2H to finish populating the
                    # pinned host buffers. CPU-side wait, doesn't block GPU.
                    t_wait = time.perf_counter()
                    copy_event.synchronize()
                    writer_stats["wait_event_secs"] += time.perf_counter() - t_wait
                    previous = layer_token_counts.get(layer_idx, 0)
                    count = int(keys.shape[0])
                    new_total = previous + count
                    layer_token_counts[layer_idx] = new_total

                    if new_total <= skip_tokens:
                        continue
                    if previous < skip_tokens:
                        trim = skip_tokens - previous
                        keys = keys[trim:]
                        values = values[trim:]
                    if keys.dim() == 4:
                        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
                        values = values.reshape(
                            -1, values.shape[-2], values.shape[-1]
                        )
                    num_tokens = int(keys.shape[0])
                    n_heads = int(keys.shape[1])
                    head_dim = int(keys.shape[2])
                    dtype_w = torch_dtype_to_wire(keys.dtype)
                    keys_bytes = tensor_to_wire_bytes(keys)
                    values_bytes = tensor_to_wire_bytes(values)
                    payload_bytes = len(keys_bytes) + len(values_bytes)
                    if chunks_sent[0] == 0:
                        writer_stats["first_byte_t"] = time.perf_counter()
                        logger.info(
                            f"First KV chunk: layer={layer_idx} keys={keys.shape} "
                            f"keys.dtype={keys.dtype} values.dtype={values.dtype}"
                        )
                    t_sock = time.perf_counter()
                    with wfile_lock:
                        write_kv_chunk(
                            wfile,
                            layer_idx=layer_idx,
                            num_tokens=num_tokens,
                            n_heads=n_heads,
                            head_dim=head_dim,
                            dtype=dtype_w,
                            keys=keys_bytes,
                            values=values_bytes,
                        )
                    writer_stats["socket_secs"] += time.perf_counter() - t_sock
                    writer_stats["bytes_shipped"] += payload_bytes
                    writer_stats["last_byte_t"] = time.perf_counter()
                    chunks_sent[0] += 1
            except Exception:
                logger.opt(exception=True).warning(
                    "serve_prefill writer thread crashed"
                )

        def arrays_writer_loop() -> None:
            try:
                while True:
                    item = arrays_queue.get()
                    if item is None:
                        break
                    layer_idx, arrays, copy_event = item
                    if copy_event is not None:
                        copy_event.synchronize()
                    with wfile_lock:
                        write_layer_arrays_blobs(
                            wfile, layer_idx, arrays_to_blobs(arrays)
                        )
                    arrays_streamed[0] += 1
            except Exception:
                logger.opt(exception=True).warning(
                    "serve_prefill arrays writer thread crashed"
                )

        writer_thread = threading.Thread(target=writer_loop, daemon=True)
        writer_thread.start()
        arrays_writer_thread = threading.Thread(
            target=arrays_writer_loop, daemon=True
        )
        arrays_writer_thread.start()

        t0 = time.perf_counter()
        forward_error: Exception | None = None
        try:
            while engine.has_unfinished_requests():
                outputs = engine.step()
                for output in outputs:
                    if (
                        getattr(output, "request_id", None) == request.request_id
                        and output.outputs[0].token_ids
                    ):
                        engine.abort_request([request.request_id])
                        break
                else:
                    continue
                break
            while engine.has_unfinished_requests():
                _ = engine.step()
        except Exception as exc:
            forward_error = exc
            with contextlib.suppress(Exception):
                engine.abort_request([request.request_id])
        finally:
            logger.info(
                f"serve_prefill {request.request_id}: "
                f"kv_queue={kv_queue.qsize()} arrays_queue={arrays_queue.qsize()}"
            )
            kv_queue.put(None)
            arrays_queue.put(None)
            writer_thread.join(timeout=30)
            arrays_writer_thread.join(timeout=30)
            if writer_thread.is_alive():
                logger.warning("serve_prefill: kv writer thread did not exit")
            if arrays_writer_thread.is_alive():
                logger.warning("serve_prefill: arrays writer thread did not exit")

        if forward_error is not None:
            logger.opt(exception=forward_error).error(
                f"serve_prefill {request.request_id}: engine.step() raised"
            )
            with contextlib.suppress(Exception):
                write_error(wfile, code=500, message=f"engine.step: {forward_error!r}")
            return

        # The K/V writer and arrays writer both drained their queues during
        # forward (see writer_loop / arrays_writer_loop above). What remains
        # here is the fallback for any GDN layer whose conv+ssm pair never
        # reached `_try_ship_gdn` — e.g., ssm captured but not conv. We skip
        # layers already shipped by the streaming path.
        gdn = get_gdn_states()
        gdn_shipped = get_gdn_shipped()
        unshipped = [li for li in sorted(gdn.keys()) if li not in gdn_shipped]
        arrays_layers = arrays_streamed[0]
        if unshipped:
            torch.cuda.synchronize()
            for layer_idx in unshipped:
                state = gdn[layer_idx]
                arrs: list[torch.Tensor] = []
                if "conv" in state:
                    arrs.append(state["conv"])
                if "ssm" in state:
                    arrs.append(state["ssm"])
                if arrs:
                    write_layer_arrays_blobs(
                        wfile, layer_idx, arrays_to_blobs(arrs)
                    )
                    arrays_layers += 1

        forwarded_per_layer = max(layer_token_counts.values(), default=0)
        tokens_sent = max(0, forwarded_per_layer - skip_tokens)
        write_prefill_done(wfile, tokens_sent)
        elapsed = time.perf_counter() - t0
        _ = get_save_kv_layer_diag  # (kept for ad-hoc debug; not logged here)
        # Bandwidth + per-stage breakdown for the writer thread.
        bytes_shipped = writer_stats["bytes_shipped"]
        wait_secs = writer_stats["wait_event_secs"]
        sock_secs = writer_stats["socket_secs"]
        first_byte_dt = (
            writer_stats["first_byte_t"] - t0
            if writer_stats["first_byte_t"]
            else 0.0
        )
        ship_secs = (
            writer_stats["last_byte_t"] - writer_stats["first_byte_t"]
            if writer_stats["last_byte_t"]
            else 0.0
        )
        eff_bw_mbps = (bytes_shipped / 1e6 / ship_secs) if ship_secs > 0 else 0.0
        peak_bw_mbps = (bytes_shipped / 1e6 / sock_secs) if sock_secs > 0 else 0.0
        logger.info(
            f"serve_prefill {request.request_id}: "
            f"streamed_chunks={chunks_sent[0]} arrays_layers={arrays_layers} "
            f"tokens={tokens_sent} elapsed_ms={elapsed * 1000:.0f} "
            f"bytes={bytes_shipped / 1e6:.0f}MB ttfb_ms={first_byte_dt * 1000:.0f} "
            f"ship_ms={ship_secs * 1000:.0f} "
            f"wait_event_ms={wait_secs * 1000:.0f} sock_ms={sock_secs * 1000:.0f} "
            f"eff_bw={eff_bw_mbps:.0f}MB/s peak_bw={peak_bw_mbps:.0f}MB/s"
        )

    def _start_task(
        self, task: TextGeneration
    ) -> tuple[
        TaskId,
        GeneratorQueue[GenerationResponse],
        Iterator[GenerationChunk | None],
    ]:
        from exo.worker.engines.vllm.prompt_format import format_vllm_prompt

        _check_for_debug_prompts(task.task_params)

        token_ids, prompt_text, _ = format_vllm_prompt(
            self._gen.engine, task.task_params
        )

        queue = GeneratorQueue[GenerationResponse]()
        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            output_generator = apply_all_parsers(
                queue.gen(),
                prompt_text,
                self.tool_parser,
                TokenizerWrapper(self._gen.engine.get_tokenizer()),
                self.model_id,
                task.task_params.tools,
            )

        check_for_cancel_every = max(self.check_for_cancel_every, 1)
        tokens_since_cancel_check = check_for_cancel_every

        def on_prefill_progress(processed: int, total: int) -> None:
            self._collect_cancellations()
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.add(task.task_id)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=PrefillProgressChunk(
                        model=self.model_id,
                        processed_tokens=processed,
                        total_tokens=total,
                    ),
                )
            )

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= check_for_cancel_every:
                tokens_since_cancel_check = 0
                self._collect_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

        task_id = self._gen.submit(
            task_id=task.task_id,
            task_params=task.task_params,
            on_prefill_progress=on_prefill_progress,
            on_generation_token=on_generation_token,
            token_ids=token_ids,
        )
        return task_id, queue, output_generator

    def _collect_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            elif task_id in self._all_tasks:
                self._cancelled_tasks.add(task_id)

    def _apply_cancellations(self) -> Iterator[tuple[TaskId, CancelledResponse]]:
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks
        results: list[tuple[TaskId, CancelledResponse]] = []
        task_ids_to_abort: list[TaskId] = []

        for task_id, (task, _, _) in list(self._active_tasks.items()):
            if cancel_all or task.task_id in self._cancelled_tasks:
                task_ids_to_abort.append(task_id)
                results.append((task.task_id, CancelledResponse()))
                del self._active_tasks[task_id]
                self._all_tasks.pop(task.task_id, None)

        if self._queue:
            kept_queue: deque[TextGeneration] = deque()
            for task in self._queue:
                if cancel_all or task.task_id in self._cancelled_tasks:
                    results.append((task.task_id, CancelledResponse()))
                    self._all_tasks.pop(task.task_id, None)
                else:
                    kept_queue.append(task)
            self._queue = kept_queue

        if task_ids_to_abort:
            self._gen.cancel(task_ids_to_abort)

        already_cancelled = {task_id for task_id, _ in results}
        for task_id in self._cancelled_tasks:
            if (
                task_id != CANCEL_ALL_TASKS
                and task_id in self._all_tasks
                and task_id not in already_cancelled
            ):
                results.append((task_id, CancelledResponse()))
                self._all_tasks.pop(task_id, None)

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=ErrorChunk(
                    model=self.model_id,
                    finish_reason="error",
                    error_message=str(e),
                ),
            )
        )
