# type: ignore
"""uv run pytest -v -m "" src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py"""

import importlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback

import numpy as np
import pytest

MODEL_CONFIGS = {
    "llama": dict(
        module="mlx_lm.models.llama",
        args=dict(
            model_type="llama",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=512,
            max_position_embeddings=128,
            head_dim=32,
            rope_theta=10000.0,
        ),
    ),
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
    "qwen3_next": dict(
        module="mlx_lm.models.qwen3_next",
        args=dict(
            model_type="qwen3_next",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=512,
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
            mlp_only_layers=[],
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
        ),
    ),
    "deepseek_v3": dict(
        module="mlx_lm.models.deepseek_v3",
        args=dict(
            model_type="deepseek_v3",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            rope_theta=10000.0,
            rope_scaling={},
            attention_bias=False,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
        ),
    ),
    "deepseek_v3_q4": dict(
        module="mlx_lm.models.deepseek_v3",
        quantize=dict(group_size=32, bits=4, mode="affine"),
        args=dict(
            model_type="deepseek_v3",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            q_lora_rank=None,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=32,
            rope_theta=10000.0,
            rope_scaling={},
            attention_bias=False,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
        ),
    ),
    "glm4_moe_lite": dict(
        module="mlx_lm.models.glm4_moe_lite",
        args=dict(
            model_type="glm4_moe_lite",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            rope_theta=10000.0,
            attention_bias=False,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=32,
        ),
    ),
    "minimax": dict(
        module="mlx_lm.models.minimax",
        args=dict(
            model_type="minimax",
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=128,
            num_experts_per_tok=2,
            num_local_experts=8,
            shared_intermediate_size=256,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rotary_dim=32,
            vocab_size=512,
        ),
    ),
    "gpt_oss": dict(
        module="mlx_lm.models.gpt_oss",
        args=dict(
            model_type="gpt_oss",
            hidden_size=512,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=512,
            head_dim=32,
            rms_norm_eps=1e-6,
            num_local_experts=8,
            num_experts_per_tok=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=64,
            rope_theta=10000.0,
        ),
    ),
    "gemma4": dict(
        module="mlx_lm.models.gemma4",
        args=dict(
            model_type="gemma4",
            vocab_size=512,
            text_config=dict(
                vocab_size=512,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=32,
                global_head_dim=32,
                num_kv_shared_layers=0,
                vocab_size_per_layer_input=512,
                hidden_size_per_layer_input=512,
                rms_norm_eps=1e-6,
                max_position_embeddings=128,
                sliding_window=64,
                sliding_window_pattern=2,
                layer_types=[
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                enable_moe_block=True,
                num_experts=8,
                top_k_experts=2,
                moe_intermediate_size=256,
            ),
        ),
    ),
}

_PROMPT = [[1, 23, 45, 67, 89, 12, 34, 56]]


def _build(name):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path

    import exo.worker.engines.mlx.auto_parallel  # noqa: F401

    cfg = MODEL_CONFIGS[name]
    module = importlib.import_module(cfg["module"])
    model_cls = module.Model
    model_args_cls = module.ModelArgs

    mx.random.seed(0)
    args = model_args_cls(**cfg["args"])
    m = model_cls(args)

    def _to_bf16(_p, v):
        if hasattr(v, "dtype") and v.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return v.astype(mx.bfloat16)
        return v

    m.update(tree_map_with_path(_to_bf16, m.parameters()))
    if "quantize" in cfg:
        nn.quantize(m, **cfg["quantize"])
    mx.eval(m.parameters())
    return mx, m


def _run(name, out_path, shard):
    import mlx.core as mx

    if shard:
        g = mx.distributed.init(backend="ring", strict=True)
    mx_, m = _build(name)
    if shard:
        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        m = tensor_auto_parallel(m, g, on_layer_loaded=None)
        mx_.eval(m.parameters())
    inputs = mx_.array(_PROMPT, dtype=mx_.int32)
    logits = m(inputs)
    mx_.eval(logits)
    np.savez(out_path, logits=np.asarray(logits.astype(mx_.float32)))


def _ref_worker(name, out_path, q):
    try:
        _run(name, out_path, shard=False)
        q.put(True)
    except BaseException as e:
        q.put(f"{e}\n{traceback.format_exc()}")


def _tp_worker(name, rank, hf, out_path, q):
    os.environ["MLX_HOSTFILE"] = hf
    os.environ["MLX_RANK"] = str(rank)
    try:
        path = out_path if rank == 0 else out_path + f".r{rank}"
        _run(name, path, shard=True)
        q.put((rank, True, None))
    except BaseException as e:
        q.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def _run_compare(name, world_size, port_base):
    d = tempfile.mkdtemp()
    ref_path = f"{d}/ref.npz"
    tp_path = f"{d}/tp.npz"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    p = ctx.Process(target=_ref_worker, args=(name, ref_path, q))
    p.start()
    p.join(300)
    r = q.get(timeout=10)
    if r is not True:
        pytest.fail(f"[{name}] ref FAIL: {str(r)[:500]}")

    hosts = [f"127.0.0.1:{port_base + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hf = f.name
    ps = [
        ctx.Process(target=_tp_worker, args=(name, rank, hf, tp_path, q))
        for rank in range(world_size)
    ]
    for pp in ps:
        pp.start()
    results = [q.get(timeout=300) for _ in range(world_size)]
    for pp in ps:
        pp.join(60)
    for rank, ok, payload in results:
        if not ok:
            pytest.fail(f"[{name}] rank {rank} FAIL: {payload[:500]}")

    ref = np.load(ref_path)["logits"]
    tp = np.load(tp_path)["logits"]
    diff = np.abs(ref - tp)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    assert max_diff == 0.0, (
        f"[{name} TP={world_size}] not bit-exact: max={max_diff} mean={mean_diff}"
    )


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin", reason="MLX distributed requires Metal"
    ),
]


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("name", list(MODEL_CONFIGS))
def test_tp_bit_exact(name, world_size):
    name_idx = list(MODEL_CONFIGS).index(name)
    port = 32000 + name_idx * 20 + world_size
    _run_compare(name, world_size, port)
