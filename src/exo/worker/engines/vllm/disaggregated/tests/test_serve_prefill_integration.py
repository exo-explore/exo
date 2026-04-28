"""End-to-end test for VllmEngine.serve_prefill.

Boots a real vLLM engine with a small model, calls serve_prefill twice in a
row against an in-memory wire buffer, and verifies both runs produce a
well-formed stream (header -> KV chunks -> Done).

The second run is the regression case: with vLLM APC enabled this would hit
the chunked-prefill + APC + custom kv-connector CUDA assert
(`vectorized_gather_kernel: ind >= ind_dim_size`) and the server would close
the socket without a Done frame. With APC disabled at engine creation time
each request runs a full forward pass and the stream is well-formed.

Run on Spark (gx10-de89):
    cd /home/larry/exo
    uv run pytest -q -s -m "" \\
        src/exo/worker/engines/vllm/disaggregated/tests/test_serve_prefill_integration.py \\
        --model-id Qwen/Qwen3-0.6B

The test is gated on `--model-id` being passed; the model must already be
present at `EXO_DEFAULT_MODELS_DIR/<id-with-/-as--->` (the standard exo
download layout). On machines without CUDA / vLLM the test is skipped.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from typing import cast

import pytest

from exo.shared.types.common import ModelId
from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    ErrorMessage,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest


def _has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _make_token_ids(n: int) -> list[int]:
    # Deterministic synthetic tokens. Vocab >= ~30k for the Qwen tokenizers we
    # care about, so 100..30099 is safe.
    return [(i * 1009 + 17) % 30000 + 100 for i in range(n)]


def _decode_stream(
    payload: bytes,
) -> tuple[list[KVChunk], list[ArraysState], Done | None, ErrorMessage | None]:
    buf = io.BytesIO(payload)
    _ = read_header(buf)
    chunks: list[KVChunk] = []
    arrays: list[ArraysState] = []
    done: Done | None = None
    error: ErrorMessage | None = None
    while True:
        msg = read_message(buf)
        if msg is None:
            break
        if isinstance(msg, KVChunk):
            chunks.append(msg)
        elif isinstance(msg, ArraysState):
            arrays.append(msg)
        elif isinstance(msg, Done):
            done = msg
            break
        elif isinstance(msg, ErrorMessage):
            error = msg
            break
    return chunks, arrays, done, error


@pytest.fixture(scope="module")
def vllm_engine(request: pytest.FixtureRequest) -> Iterator[object]:
    """Build a real VllmEngine pointed at a downloaded HF model."""
    if not _has_cuda():
        pytest.skip("CUDA not available")
    model_id_str = request.config.getoption("--model-id")
    if not model_id_str:
        pytest.skip("pass --model-id <hf-id> to run this test")

    model_id = ModelId(cast(str, model_id_str))

    from exo.download.download_utils import build_model_path

    if not build_model_path(model_id).exists():
        pytest.skip(f"model {model_id} not downloaded locally")

    from exo.worker.engines.vllm.engine import VllmEngine
    from exo.worker.engines.vllm.generator import (
        VllmBatchEngine,
        load_vllm_engine,
    )
    from exo.worker.engines.vllm.kv_connector import (
        ExoKVProducerConnector,
        _patch_gdn_capture,
        _patch_vllm_for_connector,
    )

    # Mirror VllmBuilder.load() — patches must run before LLMEngine init.
    _patch_vllm_for_connector(ExoKVProducerConnector)
    _patch_gdn_capture()

    llm_engine, tool_parser = load_vllm_engine(
        model_id=model_id,
        trust_remote_code=False,
        n_layers=1,
        kv_connector_cls=ExoKVProducerConnector,
    )
    gen = VllmBatchEngine(engine=llm_engine, model_id=model_id)

    # serve_prefill only touches self._gen.engine; the channel fields exist
    # for the (unused-here) generation path.
    class _DummySender:
        def send(self, _: object) -> None: ...

    class _DummyReceiver:
        def collect(self) -> list[object]:
            return []

    engine = VllmEngine(
        tool_parser=tool_parser,
        model_id=model_id,
        cancel_receiver=cast("object", _DummyReceiver()),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", _DummySender()),  # pyright: ignore[reportArgumentType]
        _gen=gen,
        max_concurrent_requests=1,
    )
    try:
        yield engine
    finally:
        with contextlib.suppress(Exception):
            engine.close()


def _run_one(engine: object, n_tokens: int, label: str) -> Done:
    request = PrefillRequest(
        request_id=f"itest-{label}",
        model_id="ignored",
        token_ids=_make_token_ids(n_tokens),
        start_pos=0,
        use_prefix_cache=True,
    )
    buf = io.BytesIO()
    engine.serve_prefill(request, buf)  # pyright: ignore[reportAttributeAccessIssue]

    payload = buf.getvalue()
    assert payload, f"{label}: server wrote nothing"

    chunks, arrays, done, error = _decode_stream(payload)
    if error is not None:
        pytest.fail(f"{label}: server returned ErrorMessage [{error.code}]: {error.message}")
    assert done is not None, (
        f"{label}: stream did not end with Done "
        f"(received {len(chunks)} kv chunks, {len(arrays)} arrays)"
    )
    expected = max(0, n_tokens - 2)  # serve_prefill drops the last 2 tokens
    assert done.total_tokens > 0, f"{label}: Done reported 0 tokens"
    assert done.total_tokens >= expected - 64, (
        f"{label}: got {done.total_tokens} tokens, expected ~{expected}"
    )
    assert chunks, f"{label}: no KV chunks shipped"
    return done


def test_serve_prefill_two_runs_no_apc_assert(vllm_engine: object) -> None:
    """Two consecutive prefills against the same engine must both succeed.

    Before the fix, the second call hit a CUDA assert (vLLM APC + chunked
    prefill + custom kv-connector). With APC off, each request runs a full
    forward and the stream is well-formed both times.
    """
    first = _run_one(vllm_engine, n_tokens=512, label="run1")
    second = _run_one(vllm_engine, n_tokens=512, label="run2-same-prompt")
    assert second.total_tokens == first.total_tokens, (
        f"run2 returned {second.total_tokens}, run1 returned {first.total_tokens}"
    )


def test_serve_prefill_different_lengths(vllm_engine: object) -> None:
    """A second prefill with a different prompt length still succeeds."""
    a = _run_one(vllm_engine, n_tokens=256, label="run-256")
    b = _run_one(vllm_engine, n_tokens=768, label="run-768")
    assert b.total_tokens > a.total_tokens, (
        f"longer prompt should ship more tokens: 256->{a.total_tokens} 768->{b.total_tokens}"
    )
