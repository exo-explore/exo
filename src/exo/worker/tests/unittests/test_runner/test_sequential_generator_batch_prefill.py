# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Integration tests for :meth:`SequentialGenerator._admit_queued_tasks`.

These tests verify the routing decisions in the batched-prefill path:
which queued tasks get co-prefilled in a single forward, which fall
back to per-slot, and how the env-var gate / eligibility predicate
combine. The actual numerical correctness of :func:`batched_prefill`
is covered by ``tests/test_mlx/test_batched_prefill.py`` against a
real (random-weight) model; these tests stub the prefill function
itself and assert on the SequentialGenerator's branching only.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Generator
from typing import Any, cast

import mlx.core as mx
import pytest

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.utils.channels import MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.llm_inference import batch_generator as bg_mod
from exo.worker.runner.llm_inference.batch_generator import (
    EXO_BATCH_PREFILL,
    BatchedPrefillUnsupportedError,
    SequentialGenerator,
)


class _FakeEventSender:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _make_text_task(
    text: str,
    *,
    images: list[str] | None = None,
    prefill_endpoint: str | None = None,
    bench: bool = True,
) -> TextGeneration:
    extra_kwargs: dict[str, object] = {}
    if images is not None:
        extra_kwargs["images"] = images
    if prefill_endpoint is not None:
        extra_kwargs["prefill_endpoint"] = prefill_endpoint
    return TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId(f"cmd-{text}"),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[InputMessage(role="user", content=InputMessageContent(text))],
            bench=bench,
            **extra_kwargs,
        ),
    )


def _bare_seq_generator(
    sender: _FakeEventSender,
    initial_queue: deque[TextGeneration],
    *,
    draft_model: object | None = None,
    group: object | None = None,
    max_concurrent_tasks: int = 4,
) -> SequentialGenerator:
    """Construct a SequentialGenerator without invoking dataclass init.

    The dataclass __init__ wants a real MLX model + tokenizer. We bypass
    it and stub only the attributes the admit/start path reads.
    """
    g = object.__new__(SequentialGenerator)
    g.model = cast(Any, object())
    g.tokenizer = cast(Any, object())
    g.model_id = ModelId("mlx-community/test-model")
    g.device_rank = 0
    g.event_sender = cast(MpSender[Event], cast(object, sender))
    g.group = cast(Any, group)
    g.kv_prefix_cache = cast(KVPrefixCache | None, None)
    g.tool_parser = None
    g.vision_processor = None
    g.draft_model = cast(Any, draft_model)
    g.drafter_kv_prefix_cache = None
    g.draft_model_id = None
    g.num_draft_tokens = None
    g.drafter_min_output_tokens = None
    g.adaptive_draft_tokens = False
    g.drafter_rank_in_parent = None
    g.remote_drafter_transport = None
    g.check_for_cancel_every = 50
    g._cancelled_tasks = set()
    g._maybe_queue = []
    g._maybe_cancel = []
    g._all_tasks = {task.task_id: task for task in initial_queue}
    g._queue = initial_queue
    g._active_tasks = OrderedDict()
    g._pending_failed = []
    g._recent_acceptance = deque()
    g.max_concurrent_tasks = max_concurrent_tasks
    return g


@pytest.fixture(autouse=True)
def _clear_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default to enabled so each test sets the env explicitly when needed."""
    monkeypatch.delenv(EXO_BATCH_PREFILL, raising=False)


def _stub_prep_to_eligible(
    monkeypatch: pytest.MonkeyPatch,
    eligible_ids: set[str],
) -> None:
    """Stub ``_prepare_for_batch_prefill`` to mark ``eligible_ids`` as eligible.

    The stub returns a tuple shaped like the production helper for
    eligible tasks (with a length-3 mx.array prompt and an empty cache
    list as a placeholder); ineligible tasks return ``None`` so the
    caller routes them to the per-slot path.
    """

    def fake_prep(
        _self: SequentialGenerator, task: TextGeneration
    ) -> tuple[TextGeneration, mx.array, KVCacheType] | None:
        if str(task.command_id) in eligible_ids:
            return (task, mx.array([1, 2, 3]), cast(KVCacheType, []))
        return None

    monkeypatch.setattr(SequentialGenerator, "_prepare_for_batch_prefill", fake_prep)


def _stub_start_one(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Stub ``_start_one`` to record (command_id, used_precomputed_cache) calls."""
    calls: list[tuple[str, bool]] = []

    def fake_start_one(
        gen: SequentialGenerator,
        task: TextGeneration,
        *,
        precomputed_target_cache: KVCacheType | None = None,
    ) -> None:
        calls.append((str(task.command_id), precomputed_target_cache is not None))
        gen._active_tasks[task.task_id] = (
            task,
            cast(Generator[GenerationResponse], iter(())),
            cast(Any, object()),
            cast(Any, iter(())),
        )

    monkeypatch.setattr(SequentialGenerator, "_start_one", fake_start_one)
    return calls


def _stub_batched_prefill(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: BaseException | None = None,
) -> list[int]:
    """Stub :func:`batched_prefill`. Returns the list of batch sizes seen.

    When ``side_effect`` is provided the stub raises it instead of
    returning success — used to test the fallback paths.
    """
    seen_batch_sizes: list[int] = []

    def fake_batched(
        *,
        model: object,
        prompt_tokens_list: list[mx.array],
        caches_list: list[KVCacheType],
        **_: object,
    ) -> tuple[float, int]:
        del model, caches_list
        seen_batch_sizes.append(len(prompt_tokens_list))
        if side_effect is not None:
            raise side_effect
        return 100.0, sum(int(p.size) - 1 for p in prompt_tokens_list)

    monkeypatch.setattr(bg_mod, "batched_prefill", fake_batched)
    return seen_batch_sizes


def test_two_eligible_tasks_use_batched_prefill_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two batch-eligible tasks must share one ``batched_prefill`` call."""
    sender = _FakeEventSender()
    tasks = [_make_text_task(f"t{i}") for i in range(2)]
    g = _bare_seq_generator(sender, deque(tasks))

    _stub_prep_to_eligible(monkeypatch, {f"cmd-t{i}" for i in range(2)})
    calls = _stub_start_one(monkeypatch)
    sizes = _stub_batched_prefill(monkeypatch)

    g._admit_queued_tasks()

    assert sizes == [2], "exactly one batched_prefill call with B=2"
    assert [c[0] for c in calls] == ["cmd-t0", "cmd-t1"]
    assert all(used for _, used in calls), (
        "every eligible task must receive a precomputed_target_cache"
    )


def test_single_eligible_task_falls_back_to_per_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1-eligible admit cycle skips batched_prefill (no parallelism win)."""
    sender = _FakeEventSender()
    tasks = [_make_text_task("only")]
    g = _bare_seq_generator(sender, deque(tasks))

    _stub_prep_to_eligible(monkeypatch, {"cmd-only"})
    calls = _stub_start_one(monkeypatch)
    sizes = _stub_batched_prefill(monkeypatch)

    g._admit_queued_tasks()

    assert sizes == [], "batched_prefill must not be called for a single slot"
    assert calls == [("cmd-only", False)]


def test_mixed_eligibility_routes_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible + ineligible tasks split: batched for the eligible 2, per-slot for the rest."""
    sender = _FakeEventSender()
    tasks = [_make_text_task(f"t{i}") for i in range(4)]
    g = _bare_seq_generator(sender, deque(tasks))

    _stub_prep_to_eligible(monkeypatch, {"cmd-t0", "cmd-t2"})
    calls = _stub_start_one(monkeypatch)
    sizes = _stub_batched_prefill(monkeypatch)

    g._admit_queued_tasks()

    assert sizes == [2]
    by_id = {cid: used for cid, used in calls}
    assert by_id["cmd-t0"] is True
    assert by_id["cmd-t2"] is True
    assert by_id["cmd-t1"] is False
    assert by_id["cmd-t3"] is False


def test_env_var_disables_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EXO_BATCH_PREFILL=0`` must skip batched_prefill entirely."""
    monkeypatch.setenv(EXO_BATCH_PREFILL, "0")
    sender = _FakeEventSender()
    tasks = [_make_text_task(f"t{i}") for i in range(3)]
    g = _bare_seq_generator(sender, deque(tasks))

    _stub_prep_to_eligible(monkeypatch, {f"cmd-t{i}" for i in range(3)})
    calls = _stub_start_one(monkeypatch)
    sizes = _stub_batched_prefill(monkeypatch)

    g._admit_queued_tasks()

    assert sizes == []
    assert all(not used for _, used in calls)
    assert {cid for cid, _ in calls} == {f"cmd-t{i}" for i in range(3)}


def test_unsupported_cache_falls_back_to_per_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:class:`BatchedPrefillUnsupportedError` must demote every candidate to per-slot.

    This is the runner-liveness contract: a model whose cache layers
    do not implement ``merge``/``extract`` (e.g. ``DeepseekV4Cache``)
    surfaces the unsupported error from inside the helper; the
    SequentialGenerator must catch it and continue with the per-slot
    prefill path instead of crashing the runner subprocess.
    """
    sender = _FakeEventSender()
    tasks = [_make_text_task(f"t{i}") for i in range(2)]
    g = _bare_seq_generator(sender, deque(tasks))

    _stub_prep_to_eligible(monkeypatch, {f"cmd-t{i}" for i in range(2)})
    calls = _stub_start_one(monkeypatch)
    _stub_batched_prefill(
        monkeypatch,
        side_effect=BatchedPrefillUnsupportedError("test: unsupported cache layer"),
    )

    g._admit_queued_tasks()

    assert calls == [("cmd-t0", False), ("cmd-t1", False)]


def test_distributed_group_disqualifies_batching() -> None:
    """Multi-rank target must not batch; pipeline_parallel_prefill owns the driver loop."""
    sender = _FakeEventSender()
    task = _make_text_task("only")

    class _FakeGroup:
        def size(self) -> int:
            return 4

    g = _bare_seq_generator(sender, deque([task]), group=_FakeGroup())
    assert g._batch_eligible_for_prefill(task) is False


def test_vision_request_disqualifies_batching() -> None:
    """Vision prep needs per-task embed-table patching; never batch."""
    sender = _FakeEventSender()
    task = _make_text_task("img-task", images=["data:image/png;base64,..."])
    g = _bare_seq_generator(sender, deque([task]))
    assert g._batch_eligible_for_prefill(task) is False


def test_remote_prefill_disqualifies_batching() -> None:
    """Remote prefill ships the cache off-target; the local batched forward is moot."""
    sender = _FakeEventSender()
    task = _make_text_task("rem", prefill_endpoint="http://prefill:8000")
    g = _bare_seq_generator(sender, deque([task]))
    assert g._batch_eligible_for_prefill(task) is False


def test_inprocess_drafter_disqualifies_batching() -> None:
    """In-process model drafter needs paired drafter prefill; V1 only batches the asymmetric (no draft_model) path."""
    sender = _FakeEventSender()
    task = _make_text_task("draft")
    g = _bare_seq_generator(sender, deque([task]), draft_model=object())
    assert g._batch_eligible_for_prefill(task) is False


def test_asymmetric_drafter_target_qualifies_for_batching() -> None:
    """Asymmetric drafter target rank has ``draft_model=None`` so it batches.

    Drafter prefill happens out-of-band over the wire (per-session
    ``OP_PREFILL``) so the target-side batching is independent of
    drafter alignment.
    """
    sender = _FakeEventSender()
    task = _make_text_task("asym")
    g = _bare_seq_generator(sender, deque([task]), draft_model=None)
    assert g._batch_eligible_for_prefill(task) is True
