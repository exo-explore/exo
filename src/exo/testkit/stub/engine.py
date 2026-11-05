"""A Builder and Engine that stand in for MLX: exo's runner drives them exactly as it drives MLX's.

What is real: the runner process and its state machine (exo.worker.runner.runner), the
supervisor around it, the worker's planning, the phases (connect, load, warm up) with their
statuses, the ranks agreeing on every step which requests to run and cancel (as the real
generators do), and the chunks the API streams. What is replaced: the model. Tokens come from
`exo.testkit.stub.directives`, and the ranks exchange only those agreements over a TCP ring.

A rank that loses its ring fails as MLX's ring backend does, with "[ring] connection to a peer
was lost" (see exo.testkit.stub.ring).
"""

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import BinaryIO, NoReturn, cast, final

from exo.api.types import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.chunks import Chunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.testkit.stub.config import (
    Phase,
    StubConfig,
    claim_fault,
    read_config,
    ring_proxy_port,
)
from exo.testkit.stub.directives import (
    Directives,
    FaultKind,
    parse_directives,
    reply_length,
    request_key,
    token_text,
)
from exo.testkit.stub.ring import Ring
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine

# The most of a late step's backlog the stub makes up for (see _Request.tokens_due): far
# above any sleep's slack, well below the shortest pause a test makes on purpose
_MAX_CATCH_UP_SECONDS = 1.0


def _hang_forever() -> NoReturn:
    while True:
        threading.Event().wait()


def fail(kind: FaultKind, where: str, stall_seconds: float) -> None:
    """Make this runner fail the way `kind` says."""
    match kind:
        case "none":
            return
        case "crash":
            print(f"stub: crashing {where}", file=sys.stderr, flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        case "raise":
            raise RuntimeError(f"stub fault: raise {where}")
        case "hang":
            print(f"stub: hanging {where}", file=sys.stderr, flush=True)
            _hang_forever()
        case "stall":
            print(
                f"stub: stalling {stall_seconds:g}s {where}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(stall_seconds)


def _on_every_rank(gathered: list[dict[str, object]], key: str) -> set[str]:
    """The ids every rank listed under `key`: what the ranks agree on this step."""
    each_rank = [set(cast(list[str], rank[key])) for rank in gathered]
    agreed = each_rank[0]
    for ids in each_rank[1:]:
        agreed &= ids
    return agreed


def prompt_of(task: TextGeneration) -> str:
    """The last user message, which the reply is a function of."""
    for message in reversed(task.task_params.input):
        if message.role == "user":
            return str(message.content)
    return ""


def _phase(config: StubConfig, phase: Phase, seconds: float, rank: int) -> None:
    if (
        config.fault_phase == phase
        and config.fault_rank in (None, rank)
        and claim_fault(config)
    ):
        fail(config.fault, f"in {phase}", stall_seconds=seconds)
    time.sleep(seconds)


@final
@dataclass(eq=False)
class _Request:
    task: TextGeneration
    directives: Directives
    key: str
    length: int
    prompt_tokens: int
    next_index: int = 0
    due: float = 0.0
    started: float = 0.0
    fault_done: bool = False

    def tokens_due(self, now: float) -> int:
        """How many tokens this request owes by now. Sleeps run late (macOS stretches a
        20 ms sleep past 100 ms under load), so a step makes every token that came due
        since the last one, and the rate holds. A backlog longer than a pause in a real
        model's output (this runner was stopped, or stalled on purpose) is dropped instead:
        the reply carries on at its rate rather than in a burst."""
        if now < self.due:
            return 0
        if now - self.due > _MAX_CATCH_UP_SECONDS:
            self.due = now
        owed = 1 + int((now - self.due) * self.directives.rate)
        return min(owed, self.length - self.next_index)


@final
@dataclass(eq=False)
class StubEngine(Engine):
    model_id: ModelId
    device_rank: int
    world_size: int
    ring: Ring | None
    cancel_receiver: MpReceiver[TaskId]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: set[TaskId] = field(default_factory=set, init=False)
    _active: dict[TaskId, _Request] = field(default_factory=dict, init=False)

    def warmup(self) -> None:
        config = read_config()
        _phase(config, "warmup", config.warmup_seconds, self.device_rank)
        self._gather({})

    def submit(self, task: GenerationTask) -> None:
        if not isinstance(task, TextGeneration):
            raise NotImplementedError("the stub engine only generates text")
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._maybe_queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        now = time.monotonic()
        cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                cancel_all = True
            else:
                self._maybe_cancel.add(task_id)
        # Rank 0 decides how many tokens each request makes this step, so every rank finishes
        # each request at the same step and the ranks never fall out of step with each other
        due = {
            str(task_id): count
            for task_id, request in self._active.items()
            if self.device_rank == 0 and (count := request.tokens_due(now))
        }
        gathered = self._gather(
            {
                "tasks": [str(task.task_id) for task in self._maybe_queue],
                "cancel": sorted(str(task_id) for task_id in self._maybe_cancel),
                "cancel_all": cancel_all,
                "due": due,
            }
        )
        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []

        if any(bool(rank["cancel_all"]) for rank in gathered):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)
        agreed_cancels = _on_every_rank(gathered, "cancel")
        for task_id in sorted(agreed_cancels):
            self._maybe_cancel.discard(TaskId(task_id))
            self._cancelled_tasks.add(TaskId(task_id))
        for task_id in list(self._active):
            if self.should_cancel(task_id):
                del self._active[task_id]
                output.append((task_id, CancelledResponse()))

        due_now = cast(dict[str, int], gathered[0]["due"])
        for task_id, count in due_now.items():
            for _ in range(count):
                request = self._active.get(TaskId(task_id))
                if request is None:
                    break
                output.extend(self._next_token(request, now))

        agreed = _on_every_rank(gathered, "tasks")
        for task in sorted(
            (task for task in self._maybe_queue if str(task.task_id) in agreed),
            key=lambda task: str(task.task_id),
        ):
            self._maybe_queue.remove(task)
            self._start(task, now)
        for task in list(self._maybe_queue):
            if self.should_cancel(task.task_id):
                self._maybe_queue.remove(task)
                output.append((task.task_id, CancelledResponse()))

        if self.device_rank == 0 and self._active and not output:
            # Nothing was due: wait for the next token instead of spinning
            next_due = min(request.due for request in self._active.values())
            time.sleep(max(0.0, min(next_due - time.monotonic(), 0.05)))
        return output

    def close(self) -> None:
        if self.ring is not None:
            self.ring.close()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError("the stub engine doesn't serve disaggregated prefill")

    def _start(self, task: TextGeneration, now: float) -> None:
        prompt = prompt_of(task)
        directives = parse_directives(prompt)
        self._active[task.task_id] = _Request(
            task=task,
            directives=directives,
            key=request_key(prompt),
            length=reply_length(directives, task.task_params.max_output_tokens),
            prompt_tokens=max(1, len(prompt.split())),
            due=now + directives.prefill_seconds,
            started=now,
        )

    def _next_token(
        self, request: _Request, now: float
    ) -> list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        directives = request.directives
        index = request.next_index
        if (
            not request.fault_done
            and directives.fault != "none"
            and directives.fault_rank == self.device_rank
            and directives.fault_at == index
        ):
            request.fault_done = True
            fail(
                directives.fault,
                f"before token {index} of {request.key}",
                directives.stall_seconds,
            )
        request.next_index += 1
        request.due += 1.0 / directives.rate
        task_id = request.task.task_id
        last = request.next_index >= request.length
        output: list[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]] = []
        if self.device_rank == 0:
            output.append((task_id, self._token(request, index, last, now)))
        if last:
            del self._active[task_id]
            output.append((task_id, FinishedResponse()))
        return output

    def _token(
        self, request: _Request, index: int, last: bool, now: float
    ) -> TokenChunk:
        if not last:
            return TokenChunk(
                model=self.model_id,
                text=token_text(request.key, index),
                token_id=index,
                usage=None,
            )
        elapsed = max(now - request.started, 1e-6)
        usage = Usage(
            prompt_tokens=request.prompt_tokens,
            completion_tokens=request.length,
            total_tokens=request.prompt_tokens + request.length,
            prompt_tokens_details=PromptTokensDetails(),
            completion_tokens_details=CompletionTokensDetails(),
        )
        return TokenChunk(
            model=self.model_id,
            text=token_text(request.key, index),
            token_id=index,
            usage=usage,
            finish_reason="length"
            if request.length < request.directives.tokens
            else "stop",
            stats=GenerationStats(
                prompt_tps=request.prompt_tokens / elapsed,
                generation_tps=request.length / elapsed,
                prompt_tokens=request.prompt_tokens,
                generation_tokens=request.length,
                peak_memory_usage=Memory.from_bytes(0),
            ),
        )

    def _gather(self, payload: dict[str, object]) -> list[dict[str, object]]:
        if self.ring is None:
            return [payload]
        # A lost neighbour raises RingError, which ends this runner as MLX's exception does
        gathered = self.ring.all_gather(json.dumps(payload).encode())
        return [cast(dict[str, object], json.loads(item)) for item in gathered]


@final
@dataclass(eq=False)
class StubBuilder(Builder):
    """Built by exo's runner bootstrap in place of MlxBuilder, with the same arguments."""

    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    ring: Ring | None = None
    device_rank: int = 0
    world_size: int = 1

    def connect(self, bound_instance: BoundInstance) -> None:
        config = read_config()
        _phase(
            config,
            "connect",
            config.connect_seconds,
            bound_instance.bound_shard.device_rank,
        )
        self.ring = Ring.connect(bound_instance, ring_proxy_port())

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        config = read_config()
        shard = bound_instance.bound_shard
        self.device_rank, self.world_size = shard.device_rank, shard.world_size
        layers = max(1, shard.end_layer - shard.start_layer)
        if (
            config.fault_phase == "load"
            and config.fault_rank in (None, shard.device_rank)
            and claim_fault(config)
        ):
            fail(config.fault, "in load", stall_seconds=config.load_seconds)
        for layer in range(layers):
            time.sleep(config.load_seconds / layers)
            yield ModelLoadingResponse(layers_loaded=layer + 1, total=layers)
        if self.ring is not None:
            # Loading ends with the ranks waiting for each other, as sharded loading does
            self.ring.barrier()

    def build(self) -> Engine:
        return StubEngine(
            model_id=self.model_id,
            device_rank=self.device_rank,
            world_size=self.world_size,
            ring=self.ring,
            cancel_receiver=self.cancel_receiver,
        )

    def close(self) -> None:
        if self.ring is not None:
            self.ring.close()
            self.ring = None
