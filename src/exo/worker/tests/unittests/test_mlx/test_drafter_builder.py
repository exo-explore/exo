"""Tests for MlxBuilder selecting the right Engine based on drafter presence.

These tests stub out the heavy MLX paths (model load, KVPrefixCache,
tokenizer probing) and just exercise the routing logic in ``MlxBuilder.build``:

- No drafter, batching enabled (default): ``BatchGenerator``.
- No drafter, ``EXO_NO_BATCH`` set: ``SequentialGenerator``.
- Drafter loaded, batching enabled: ``SequentialGenerator`` is forced because
  upstream ``BatchGenerator`` does not support speculative decoding.
- Drafter is threaded through into the SequentialGenerator's ``draft_model``
  field so ``mlx_generate`` can pass it to ``stream_generate``.
"""

from typing import cast
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)


def _build_mlx_builder(
    *,
    draft_model: Model | None,
    draft_model_id: ModelId | None = None,
    group: mx.distributed.Group | None = None,
) -> MlxBuilder:
    fake_tokenizer = MagicMock(spec=TokenizerWrapper)
    fake_tokenizer.has_tool_calling = False
    fake_tokenizer.tool_call_start = None
    fake_tokenizer.tool_call_end = None
    fake_tokenizer.tool_parser = None

    return MlxBuilder(
        model_id=ModelId("mlx-community/test-target"),
        event_sender=cast(MpSender[Event], MagicMock()),
        cancel_receiver=cast(MpReceiver[TaskId], MagicMock()),
        inference_model=cast(Model, MagicMock()),
        tokenizer=fake_tokenizer,
        group=group,
        vision_processor=None,
        draft_model=draft_model,
        draft_model_id=draft_model_id,
    )


def _fake_group(size: int, rank: int = 0) -> mx.distributed.Group:
    fake = MagicMock(spec=mx.distributed.Group)
    fake.size = MagicMock(return_value=size)
    fake.rank = MagicMock(return_value=rank)
    return cast(mx.distributed.Group, fake)


def test_mlx_builder_uses_batch_generator_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    builder = _build_mlx_builder(draft_model=None)
    engine = builder.build()
    assert isinstance(engine, BatchGenerator)


def test_mlx_builder_uses_sequential_when_no_batch_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    builder = _build_mlx_builder(draft_model=None)
    engine = builder.build()
    assert isinstance(engine, SequentialGenerator)
    assert engine.draft_model is None


def test_mlx_builder_forces_sequential_when_drafter_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a drafter model is present, BatchGenerator can't use it, so we must
    fall back to SequentialGenerator regardless of EXO_NO_BATCH."""
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.delenv("EXO_NUM_DRAFT_TOKENS", raising=False)
    monkeypatch.delenv("EXO_DRAFTER_MIN_OUTPUT_TOKENS", raising=False)
    fake_drafter = cast(Model, MagicMock())
    drafter_id = ModelId("mlx-community/test-drafter")
    builder = _build_mlx_builder(draft_model=fake_drafter, draft_model_id=drafter_id)

    engine = builder.build()

    assert isinstance(engine, SequentialGenerator)
    assert engine.draft_model is fake_drafter
    assert engine.draft_model_id == drafter_id
    # Defaults should be applied so dashboards see the actual K in use.
    assert engine.num_draft_tokens is not None and engine.num_draft_tokens >= 2
    assert (
        engine.drafter_min_output_tokens is not None
        and engine.drafter_min_output_tokens > 0
    )


def test_mlx_builder_honours_env_overrides_for_drafter_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_NUM_DRAFT_TOKENS", "7")
    monkeypatch.setenv("EXO_DRAFTER_MIN_OUTPUT_TOKENS", "32")
    fake_drafter = cast(Model, MagicMock())
    builder = _build_mlx_builder(
        draft_model=fake_drafter,
        draft_model_id=ModelId("mlx-community/test-drafter"),
    )

    engine = builder.build()

    assert isinstance(engine, SequentialGenerator)
    assert engine.num_draft_tokens == 7
    assert engine.drafter_min_output_tokens == 32


def test_mlx_builder_routes_to_sequential_when_request_drafting_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2 (PR #19 round 2): when ``EXO_ALLOW_REQUEST_DRAFTING`` is
    set, the builder must route to SequentialGenerator even when no
    drafter model is loaded and ``EXO_DRAFT_MODE`` is unset.
    BatchGenerator silently ignores per-request ``draft_mode``
    overrides because it has no spec-decoding hook, so honoring
    request-level ngram drafting requires the sequential path.
    """
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
    monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", "1")

    builder = _build_mlx_builder(draft_model=None)
    engine = builder.build()

    assert isinstance(engine, SequentialGenerator), (
        "EXO_ALLOW_REQUEST_DRAFTING must force SequentialGenerator so "
        "per-request draft_mode overrides actually take effect; got "
        f"{type(engine).__name__}"
    )
    # No drafter model loaded -> the engine accepts ngram requests but
    # doesn't have a model drafter to fall back to.
    assert engine.draft_model is None


def test_mlx_builder_request_drafting_flag_accepts_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthy spellings (``1``, ``true``, ``yes``) all enable
    request-level drafting; the empty string and ``0`` do not.
    """
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)

    for truthy in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", truthy)
        builder = _build_mlx_builder(draft_model=None)
        engine = builder.build()
        assert isinstance(engine, SequentialGenerator), (
            f"EXO_ALLOW_REQUEST_DRAFTING={truthy!r} must enable request "
            f"drafting; got {type(engine).__name__}"
        )

    for falsey in ("", "0", "no", "false"):
        monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", falsey)
        builder = _build_mlx_builder(draft_model=None)
        engine = builder.build()
        assert isinstance(engine, BatchGenerator), (
            f"EXO_ALLOW_REQUEST_DRAFTING={falsey!r} must NOT trigger "
            f"sequential routing; got {type(engine).__name__}"
        )


class TestMultiDeviceDraftingFallback:
    """Codex P1 (PR #19 round-(N+3), builder.py:136): forcing
    ``SequentialGenerator`` for ``EXO_DRAFT_MODE=ngram`` /
    ``EXO_ALLOW_REQUEST_DRAFTING`` on multi-device runners loses
    batching with no benefit, because ``mlx_generate`` demotes
    ``draft_mode`` to ``"none"`` whenever ``group`` is set. The
    builder must keep ``BatchGenerator`` when the runner is
    multi-device so concurrent traffic preserves throughput.
    """

    def test_multi_device_runner_keeps_batch_generator_under_ngram_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "ngram")
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)

        builder = _build_mlx_builder(draft_model=None, group=_fake_group(size=2))
        engine = builder.build()

        assert isinstance(engine, BatchGenerator), (
            "multi-device runner with EXO_DRAFT_MODE=ngram must stay on "
            "BatchGenerator (mlx_generate demotes draft_mode='none' for "
            f"distributed anyway); got {type(engine).__name__}"
        )

    def test_multi_device_runner_keeps_batch_generator_under_request_drafting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
        monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", "1")

        builder = _build_mlx_builder(draft_model=None, group=_fake_group(size=4))
        engine = builder.build()

        assert isinstance(engine, BatchGenerator), (
            "multi-device runner with EXO_ALLOW_REQUEST_DRAFTING must stay "
            "on BatchGenerator; got {type(engine).__name__}"
        )

    def test_multi_device_runner_keeps_batch_generator_with_loaded_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even when a drafter is loaded, distributed mlx_generate
        # demotes draft_mode='none', so SequentialGenerator buys
        # nothing. Keep batching for throughput.
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)

        builder = _build_mlx_builder(
            draft_model=cast(Model, MagicMock()),
            draft_model_id=ModelId("mlx-community/test-drafter"),
            group=_fake_group(size=2),
        )
        engine = builder.build()

        assert isinstance(engine, BatchGenerator), (
            "multi-device runner with a loaded drafter must stay on "
            "BatchGenerator until mlx_generate gains a multi-device drafting "
            f"path; got {type(engine).__name__}"
        )

    def test_single_device_group_still_routes_to_sequential_for_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``size==1`` is single-device-via-group; drafting is
        # available, so SequentialGenerator is correct.
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)
        fake_drafter = cast(Model, MagicMock())

        builder = _build_mlx_builder(
            draft_model=fake_drafter,
            draft_model_id=ModelId("mlx-community/test-drafter"),
            group=_fake_group(size=1),
        )
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "single-device-via-group runner with a drafter must use "
            f"SequentialGenerator; got {type(engine).__name__}"
        )
        assert engine.draft_model is fake_drafter

    def test_multi_device_with_exo_no_batch_still_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``EXO_NO_BATCH`` disables batching entirely (different
        # operator intent); sequential is correct regardless of
        # device count.
        monkeypatch.setenv("EXO_NO_BATCH", "1")
        monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)

        builder = _build_mlx_builder(draft_model=None, group=_fake_group(size=4))
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "EXO_NO_BATCH must always route to SequentialGenerator, even on "
            f"multi-device runners; got {type(engine).__name__}"
        )


class TestNgramEnvDoesNotForceSequentialAlone:
    """Codex P1 (PR #19 round-(N+6), builder.py:151).

    ``mlx_generate`` demotes ``draft_mode="ngram"`` to ``"none"`` for
    non-greedy requests (the runner default sampler uses
    ``temperature=0.7``), so forcing ``SequentialGenerator`` whenever
    ``EXO_DRAFT_MODE=ngram`` is set silently disables batching for
    the entire worker -- a strict throughput regression for mixed
    traffic where most requests are non-greedy and never speculate.

    The builder must keep ``BatchGenerator`` when ``EXO_DRAFT_MODE=ngram``
    is the *only* drafting trigger; operators who explicitly want
    n-gram acceleration on greedy requests must also opt into
    sequential mode (``EXO_NO_BATCH=1``) or per-request control
    (``EXO_ALLOW_REQUEST_DRAFTING=1``).
    """

    def test_single_device_ngram_env_alone_keeps_batch_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "ngram")
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)

        # Capture loguru warnings via a sink we install for the duration
        # of the test; ``loguru`` doesn't pipe to pytest's stdlib
        # ``caplog`` so we attach a custom sink instead.
        from exo.worker.runner.bootstrap import logger as bootstrap_logger

        captured_warnings: list[str] = []
        sink_id = bootstrap_logger.add(
            lambda message: captured_warnings.append(str(message)),
            level="WARNING",
        )
        try:
            builder = _build_mlx_builder(draft_model=None)
            engine = builder.build()
        finally:
            bootstrap_logger.remove(sink_id)

        assert isinstance(engine, BatchGenerator), (
            "EXO_DRAFT_MODE=ngram alone must NOT force SequentialGenerator; "
            "n-gram is best-effort under non-greedy sampling and would be "
            "demoted to 'none' anyway. Forcing sequential here loses batching "
            f"for the entire worker. Got {type(engine).__name__}"
        )
        # Operator-facing warning must explain that the n-gram env is a
        # no-op without EXO_NO_BATCH or EXO_ALLOW_REQUEST_DRAFTING.
        assert any(
            "EXO_DRAFT_MODE='ngram' set" in msg and "no-op" in msg
            for msg in captured_warnings
        ), f"Expected n-gram no-op warning; captured: {captured_warnings}"

    def test_single_device_ngram_with_exo_no_batch_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``EXO_NO_BATCH`` is the operator-side opt-in that makes
        # ``EXO_DRAFT_MODE=ngram`` actually run for greedy requests.
        monkeypatch.setenv("EXO_NO_BATCH", "1")
        monkeypatch.setenv("EXO_DRAFT_MODE", "ngram")
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)

        builder = _build_mlx_builder(draft_model=None)
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "EXO_DRAFT_MODE=ngram + EXO_NO_BATCH=1 must use "
            f"SequentialGenerator; got {type(engine).__name__}"
        )

    def test_single_device_ngram_with_request_drafting_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Per-request override path: SequentialGenerator is required so
        # ``draft_mode="ngram"`` from a request body actually applies.
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "ngram")
        monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", "1")

        builder = _build_mlx_builder(draft_model=None)
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "EXO_ALLOW_REQUEST_DRAFTING=1 must force SequentialGenerator "
            f"so request-level overrides apply; got {type(engine).__name__}"
        )

    def test_single_device_with_loaded_drafter_still_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Loaded ``draft_model`` (not just ngram env) is the strong
        # signal for sequential mode; this path is unchanged.
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.delenv("EXO_DRAFT_MODE", raising=False)
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)
        fake_drafter = cast(Model, MagicMock())

        builder = _build_mlx_builder(
            draft_model=fake_drafter,
            draft_model_id=ModelId("mlx-community/test-drafter"),
        )
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "loaded drafter must still force SequentialGenerator regardless "
            f"of EXO_DRAFT_MODE; got {type(engine).__name__}"
        )


class TestExoDraftModeNoneRespectsBatching:
    """Codex P1 (PR #19 round-(N+8), builder.py:169): when the
    operator explicitly sets ``EXO_DRAFT_MODE=none`` while a drafter
    model is loaded, the worker MUST keep ``BatchGenerator``.

    ``mlx_generate`` resolves ``draft_mode="none"`` for every request
    in this configuration (the env var overrides the default
    ``"model"`` that a loaded drafter would imply), so forcing
    ``SequentialGenerator`` would lose batching with zero
    spec-decode benefit -- a strict throughput regression for the
    common 'load drafter weights but disable speculation for this
    workload' pattern.
    """

    def test_loaded_drafter_with_explicit_none_keeps_batch_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "none")
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)
        fake_drafter = cast(Model, MagicMock())

        builder = _build_mlx_builder(
            draft_model=fake_drafter,
            draft_model_id=ModelId("mlx-community/test-drafter"),
        )
        engine = builder.build()

        assert isinstance(engine, BatchGenerator), (
            "EXO_DRAFT_MODE='none' must keep BatchGenerator even with a "
            "loaded drafter model -- mlx_generate would resolve "
            "draft_mode='none' for every request anyway, so "
            "SequentialGenerator buys nothing and only loses batching. "
            f"Got {type(engine).__name__}"
        )

    def test_loaded_drafter_with_explicit_none_plus_request_drafting_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-request opt-in must still force SequentialGenerator
        because requests can legitimately raise ``draft_mode`` above
        ``"none"`` via ``use_drafter=true`` (see
        ``resolve_draft_mode``)."""
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "none")
        monkeypatch.setenv("EXO_ALLOW_REQUEST_DRAFTING", "1")
        fake_drafter = cast(Model, MagicMock())

        builder = _build_mlx_builder(
            draft_model=fake_drafter,
            draft_model_id=ModelId("mlx-community/test-drafter"),
        )
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "EXO_ALLOW_REQUEST_DRAFTING=1 still forces SequentialGenerator "
            "so per-request use_drafter=true overrides apply; "
            f"got {type(engine).__name__}"
        )

    def test_loaded_drafter_with_model_mode_still_uses_sequential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity check the non-regression path: explicit
        ``EXO_DRAFT_MODE='model'`` with a loaded drafter must still
        route to ``SequentialGenerator``."""
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
        monkeypatch.setenv("EXO_DRAFT_MODE", "model")
        monkeypatch.delenv("EXO_ALLOW_REQUEST_DRAFTING", raising=False)
        fake_drafter = cast(Model, MagicMock())

        builder = _build_mlx_builder(
            draft_model=fake_drafter,
            draft_model_id=ModelId("mlx-community/test-drafter"),
        )
        engine = builder.build()

        assert isinstance(engine, SequentialGenerator), (
            "EXO_DRAFT_MODE='model' with loaded drafter must use "
            f"SequentialGenerator; got {type(engine).__name__}"
        )
