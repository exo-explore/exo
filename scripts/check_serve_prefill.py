#!/usr/bin/env python
"""Standalone smoke test for VllmEngine.serve_prefill.

Loads a real vLLM engine, runs serve_prefill against an in-memory buffer
twice in a row with the same prompt, and verifies both runs produce a
well-formed wire stream (header -> KV chunks -> Done).

The second run is the regression guard: with vLLM APC enabled this would
trip the chunked-prefill + APC + custom kv-connector CUDA assert
(`vectorized_gather_kernel: ind >= ind_dim_size`) and the server would
close the socket before the Done frame.

Usage on the Spark (gx10-de89):

    cd /home/larry/exo
    /nix/store/2b82iz9ac0pxqafrgxmgdkq8sr2hwlx6-exo-cuda-13-venv/bin/python \\
        scripts/check_serve_prefill.py Qwen/Qwen3-0.6B

Exits 0 on success, non-zero with a diagnostic on failure.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from pathlib import Path
from typing import cast


def _ensure_repo_on_path() -> None:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_repo_on_path()

from exo.shared.types.common import ModelId  # noqa: E402
from exo.worker.disaggregated.protocol import (  # noqa: E402
    ArraysState,
    Done,
    ErrorMessage,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest  # noqa: E402


def _make_token_ids(n: int) -> list[int]:
    return [(i * 1009 + 17) % 30000 + 100 for i in range(n)]


def _decode(
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


def _build_engine(model_id: ModelId) -> object:
    from exo.worker.engines.vllm.engine import VllmEngine
    from exo.worker.engines.vllm.generator import VllmBatchEngine, load_vllm_engine
    from exo.worker.engines.vllm.kv_connector import (
        ExoKVProducerConnector,
        _patch_gdn_capture,
        _patch_vllm_for_connector,
    )

    _patch_vllm_for_connector(ExoKVProducerConnector)
    _patch_gdn_capture()

    llm_engine, tool_parser = load_vllm_engine(
        model_id=model_id,
        trust_remote_code=False,
        n_layers=1,
        kv_connector_cls=ExoKVProducerConnector,
    )
    gen = VllmBatchEngine(engine=llm_engine, model_id=model_id)

    class _S:
        def send(self, _: object) -> None: ...

    class _R:
        def collect(self) -> list[object]:
            return []

    return VllmEngine(
        tool_parser=tool_parser,
        model_id=model_id,
        cancel_receiver=cast("object", _R()),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", _S()),  # pyright: ignore[reportArgumentType]
        _gen=gen,
        max_concurrent_requests=1,
    )


def _run_one(engine: object, n_tokens: int, label: str) -> int:
    request = PrefillRequest(
        request_id=f"check-{label}-{os.getpid()}",
        model_id="ignored",
        token_ids=_make_token_ids(n_tokens),
        start_pos=0,
        use_prefix_cache=True,
    )
    buf = io.BytesIO()
    engine.serve_prefill(request, buf)  # pyright: ignore[reportAttributeAccessIssue]
    payload = buf.getvalue()
    if not payload:
        raise AssertionError(f"{label}: server wrote nothing")

    chunks, arrays, done, error = _decode(payload)
    if error is not None:
        raise AssertionError(
            f"{label}: server returned ErrorMessage [{error.code}]: {error.message}"
        )
    if done is None:
        raise AssertionError(
            f"{label}: stream did not end with Done "
            f"({len(chunks)} kv chunks, {len(arrays)} arrays)"
        )
    if done.total_tokens <= 0:
        raise AssertionError(f"{label}: Done reported {done.total_tokens} tokens")
    if not chunks:
        raise AssertionError(f"{label}: no KV chunks shipped")

    expected = max(0, n_tokens - 2)
    if done.total_tokens < expected - 64:
        raise AssertionError(
            f"{label}: got {done.total_tokens} tokens, expected ~{expected}"
        )
    print(
        f"  [{label}] OK: tokens={done.total_tokens} "
        f"kv_chunks={len(chunks)} arrays={len(arrays)}"
    )
    return done.total_tokens


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    model_id = ModelId(argv[1])

    from exo.download.download_utils import build_model_path

    model_path = build_model_path(model_id)
    if not model_path.exists():
        print(f"FAIL: model {model_id} not found at {model_path}")
        return 1
    print(f"Loading vLLM engine for {model_id} ({model_path}) ...")

    engine = _build_engine(model_id)
    failures: list[str] = []
    try:
        try:
            t1 = _run_one(engine, n_tokens=512, label="run1-fresh")
        except AssertionError as e:
            failures.append(f"run1: {e}")
            t1 = 0
        try:
            t2 = _run_one(engine, n_tokens=512, label="run2-same-prompt")
        except AssertionError as e:
            failures.append(f"run2: {e}")
            t2 = 0
        if t1 and t2 and t1 != t2:
            failures.append(
                f"run1 returned {t1} tokens but run2 returned {t2} (should match)"
            )
        try:
            ta = _run_one(engine, n_tokens=256, label="run3-shorter")
            tb = _run_one(engine, n_tokens=768, label="run4-longer")
            if ta and tb and tb <= ta:
                failures.append(
                    f"longer prompt should produce more tokens: 256->{ta} 768->{tb}"
                )
        except AssertionError as e:
            failures.append(f"length-variation: {e}")
    finally:
        with contextlib.suppress(Exception):
            engine.close()  # pyright: ignore[reportAttributeAccessIssue]

    if failures:
        print()
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
