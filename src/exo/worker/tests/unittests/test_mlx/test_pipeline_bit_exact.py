# type: ignore
"""Pipeline bit-exact test: single-process vs 2-rank pipeline logits must match.

Adapted from test_tp_bit_exact.py's harness for PIPELINE sharding — the mode
with no existing bit-exact coverage. Reproduces the Qwen3.6-35B (qwen3_5_moe)
2-node corruption as a deterministic synthetic red/green test.

Run: EXO_DASHBOARD_DIR=exo-dashboard-stub .venv/bin/python -m pytest \
  src/exo/worker/tests/unittests/test_mlx/test_pipeline_bit_exact.py -v -m ""
"""

import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.models.model_cards import ModelCard, ModelTask

_PROMPT = [[1, 23, 45, 67, 89, 12, 34, 56]]

MODEL_CONFIGS = {
    # Hybrid linear/full attention MoE — the family that garbles under 2-node
    # pipeline sharding (Qwen3.6-35B-A3B arch)
    "qwen3_5_moe": dict(
        module="mlx_lm.models.qwen3_5_moe",
        args=dict(
            model_type="qwen3_5_moe",
            text_config=dict(
                model_type="qwen3_5_moe",
                vocab_size=512,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=32,
                max_position_embeddings=128,
                rms_norm_eps=1e-6,
                tie_word_embeddings=False,
                attention_bias=False,
                full_attention_interval=2,
                linear_num_value_heads=32,
                linear_num_key_heads=16,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                num_experts=16,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                shared_expert_intermediate_size=256,
                moe_intermediate_size=256,
                norm_topk_prob=True,
                rope_parameters={
                    "type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                },
            ),
        ),
    ),
}


def _build(name):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path

    import exo.worker.engines.mlx.auto_parallel  # noqa: F401

    cfg = MODEL_CONFIGS[name]
    module = __import__(cfg["module"], fromlist=["Model", "ModelArgs"])
    mx.random.seed(0)
    args = module.ModelArgs(**cfg["args"])
    m = module.Model(args)

    def _to_bf16(_p, v):
        if hasattr(v, "dtype") and v.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return v.astype(mx.bfloat16)
        return v

    m.update(tree_map_with_path(_to_bf16, m.parameters()))
    mx.eval(m.parameters())
    return mx, m


def _run_pipeline(name, out_path, rank: int, world_size: int,
                  layer_splits: list[tuple[int, int]], steps: int):
    import mlx.core as mx
    import exo.worker.engines.mlx.auto_parallel as ap

    g = mx.distributed.init(backend="ring", strict=True)
    mx_, m = _build(name)

    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.shared.models.model_cards import ModelCard

    start_layer, end_layer = layer_splits[rank]
    total = layer_splits[-1][1]
    shard_meta = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test/" + name),
            storage_size=Memory.from_gb(1),
            n_layers=total,
            hidden_size=512,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=total,
    )

    gen = ap.pipeline_auto_parallel(m, g, shard_meta)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        m = stop.value

    inputs = mx_.array(_PROMPT, dtype=mx_.int32)
    if steps <= 1:
        logits = m(inputs)
        mx_.eval(logits)
        np.savez(out_path, logits=np.asarray(logits.astype(mx_.float32)))
    else:
        # decode-style: prefill forward with cache, then step-by-step
        cache = m.make_cache()  # type: ignore
        logits = m(inputs, cache=cache)
        mx_.eval(logits)
        # greedy next-token stepping like exo's generator
        token = mx_.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        outs = [token]
        for _ in range(steps - 1):
            logits = m(token, cache=cache)
            mx_.eval(logits)
            token = mx_.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            outs.append(token)
        seq = mx_.concatenate(outs, axis=-1)
        np.savez(out_path, steps=np.asarray(seq.tolist()))


def _pipeline_worker(name, out_path, rank, world_size, layer_splits, steps, q):
    try:
        _run_pipeline(name, out_path, rank, world_size, layer_splits, steps)
        q.put((rank, True, "ok"))
    except Exception:
        import traceback
        q.put((rank, False, traceback.format_exc()[-800:]))


def _ref_worker(name, out_path, steps, q):
    try:
        mx_, m = _build(name)
        inputs = mx_.array(_PROMPT, dtype=mx_.int32)
        if steps <= 1:
            logits = m(inputs)
            mx_.eval(logits)
            np.savez(out_path, logits=np.asarray(logits.astype(mx_.float32)))
        else:
            cache = m.make_cache()  # type: ignore
            logits = m(inputs, cache=cache)
            mx_.eval(logits)
            token = mx_.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            outs = [token]
            for _ in range(steps - 1):
                logits = m(token, cache=cache)
                mx_.eval(logits)
                token = mx_.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                outs.append(token)
            seq = mx_.concatenate(outs, axis=-1)
            np.savez(out_path, steps=np.asarray(seq.tolist()))
        q.put((-1, True, "ok"))
    except Exception:
        import traceback
        q.put((-1, False, traceback.format_exc()[-800:]))


def _hostfile(world_size: int, base_port: int) -> str:
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
    return f.name


def _run_compare(name: str, world_size: int, port: int, steps: int):
    ctx = mp.get_context("spawn")
    hostfile = _hostfile(world_size, port)
    os.environ["MLX_HOSTFILE"] = hostfile

    total_layers = 4
    per = total_layers // world_size
    layer_splits = [(r * per, (r + 1) * per) for r in range(world_size)]

    ref_path = f"/tmp/pipeline_ref_{name}_{steps}.npz"
    tp_path = f"/tmp/pipeline_shard_{name}_{steps}.npz"

    q: Any = ctx.Queue()
    ref = ctx.Process(target=_ref_worker, args=(name, ref_path, steps, q))
    ref.start()
    ref.join(120)
    ref_ok = q.get(timeout=300)

    # rank workers need MLX_RANK set per process BEFORE mlx import
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_pipeline_worker,
            args=(name, tp_path, rank, world_size, layer_splits, steps, q),
        )
        # MLX_RANK must be set inside the child before mx.distributed.init
        procs.append(p)
    # patch: set env per-child via wrapper
    def _with_rank(rank, fn):
        def inner():
            os.environ["MLX_RANK"] = str(rank)
            fn()
        return inner

    # simpler: run workers sequentially in own processes with env set by target
    for rank, p in enumerate(procs):
        p.start()
        p.join(240)
    results = [q.get(timeout=300) for _ in range(world_size + 1)]

    for rank, ok, payload in results:
        if not ok:
            pytest.fail(f"[{name} rank {rank}] FAIL: {payload}")

    ref = np.load(ref_path)
    shard = np.load(tp_path)
    if steps <= 1:
        diff = np.abs(ref["logits"] - shard["logits"])
        max_diff = float(diff.max())
        assert max_diff == 0.0, (
            f"[{name} PIPELINE={world_size} steps={steps}] not bit-exact: "
            f"max={max_diff} mean={float(diff.mean())}"
        )
    else:
        assert list(ref["steps"][0]) == list(shard["steps"][0]), (
            f"[{name} PIPELINE={world_size}] decode divergence: "
            f"ref={list(ref['steps'][0])} shard={list(shard['steps'][0])}"
        )


@pytest.mark.slow
@pytest.mark.skipif(os.sys.platform != "darwin", reason="MLX distributed requires Metal")
@pytest.mark.parametrize("steps", [1, 8])
def test_pipeline_bit_exact_qwen3_5_moe(steps):
    _run_compare("qwen3_5_moe", world_size=2, port=32400 + steps, steps=steps)
