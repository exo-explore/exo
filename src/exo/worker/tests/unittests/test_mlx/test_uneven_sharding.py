# type: ignore
import importlib
import itertools
import json
import multiprocessing as mp
import os
import tempfile
import traceback

import mlx.core as mx
import numpy as np
import pytest
from mlx.nn.layers.distributed import compute_shard_sizes

from exo.shared.types.worker.shards import TensorShardMode

RANDOM_SEED = 42
INPUT_TOKENS = [1, 100, 200, 300]

REDUCED_CONFIGS = {
    "llama": {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    },
    "gpt_oss": {
        "model_type": "gpt_oss",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "vocab_size": 1024,
        "intermediate_size": 256,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "sliding_window": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-5,
    },
    "deepseek_v3": {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_layer_freq": 2,
        "qk_rope_head_dim": 32,
        "qk_nope_head_dim": 32,
        "v_head_dim": 64,
        "q_lora_rank": None,
        "kv_lora_rank": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
    "step3p5": {
        "model_type": "step3p5",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_attention_groups": 4,
        "head_dim": 32,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_intermediate_size": 128,
        "share_expert_dim": 128,
        "sliding_window": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    },
    "minimax": {
        "model_type": "minimax",
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 256,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rotary_dim": 32,
        # QK norm's all_gather pattern requires equal shard sizes across ranks,
        # incompatible with uneven tp — needs separate fix for padded all_gather
        "use_qk_norm": False,
    },
    "qwen3_moe": {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [],
        "moe_intermediate_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "max_position_embeddings": 1024,
        "norm_topk_prob": True,
    },
    "qwen3_5": {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 1024,
            "intermediate_size": 512,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 256,
            "moe_intermediate_size": 128,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "max_position_embeddings": 1024,
            "head_dim": 32,
        },
    },
    "glm4_moe": {
        "model_type": "glm4_moe",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 1.0,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_qk_norm": False,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "partial_rotary_factor": 1.0,
        "norm_topk_prob": True,
    },
    "nemotron_h": {
        "model_type": "nemotron_h",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "max_position_embeddings": 1024,
        "attention_bias": False,
        "mamba_num_heads": 8,
        "mamba_head_dim": 32,
        "mamba_proj_bias": False,
        "ssm_state_size": 16,
        "conv_kernel": 4,
        "n_groups": 4,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "head_dim": 32,
        "hybrid_override_pattern": "M-*E",
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    },
    "glm4_moe_lite": {
        "model_type": "glm4_moe_lite",
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 1024,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_layer_freq": 2,
        "qk_rope_head_dim": 32,
        "qk_nope_head_dim": 32,
        "v_head_dim": 64,
        "q_lora_rank": None,
        "kv_lora_rank": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
}


def _build_model(config):
    mx.random.seed(RANDOM_SEED)
    mod = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    args = mod.ModelArgs.from_dict(config)
    model = mod.Model(args)
    mx.eval(model.parameters())
    return model


def _forward(model, tokens):
    x = mx.array([tokens])
    logits = model(x)
    mx.eval(logits)
    return np.array(logits[0, -1, :])


def _create_hostfile(world_size, base_port):
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
    return f.name


def _run_single_device(config, result_queue):
    try:
        model = _build_model(config)
        logits = _forward(model, INPUT_TOKENS)
        result_queue.put((0, True, logits))
    except Exception as e:
        result_queue.put((0, False, f"{e}\n{traceback.format_exc()}"))


def _run_tensor_device(
    rank,
    world_size,
    hostfile_path,
    config,
    result_queue,
    shard_weights=None,
    shard_mode=None,
):
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        model = _build_model(config)

        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        model = tensor_auto_parallel(
            model,
            group,
            timeout_seconds=60.0,
            on_timeout=None,
            on_layer_loaded=None,
            shard_weights=shard_weights,
            shard_mode=shard_mode,
        )

        logits = _forward(model, INPUT_TOKENS)
        result_queue.put((rank, True, logits))
    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def _run_single(config):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_run_single_device, args=(config, result_queue))
    p.start()
    p.join(timeout=60)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        raise TimeoutError("Single device timed out")
    rank, success, value = result_queue.get()
    assert success, f"Single device failed: {value}"
    return value


def _run_tensor(config, world_size, base_port, shard_weights=None, shard_mode=None):
    ctx = mp.get_context("spawn")
    hostfile_path = _create_hostfile(world_size, base_port)
    try:
        result_queue = ctx.Queue()
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_run_tensor_device,
                args=(
                    rank,
                    world_size,
                    hostfile_path,
                    config,
                    result_queue,
                    shard_weights,
                    shard_mode,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=120)

        timed_out = any(p.is_alive() for p in processes)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        assert not timed_out, "Tensor parallel timed out"

        results = {}
        while not result_queue.empty():
            rank, success, value = result_queue.get()
            results[rank] = (success, value)

        assert len(results) == world_size, (
            f"Missing results: got {list(results.keys())}"
        )
        for rank, (success, value) in results.items():
            assert success, f"Rank {rank} failed: {value}"

        return results[0][1]
    finally:
        os.unlink(hostfile_path)


class TestComputeShardSizes:
    def test_even_division(self):
        assert compute_shard_sizes(64, 2) == [32, 32]
        assert compute_shard_sizes(64, 4) == [16, 16, 16, 16]

    def test_uneven_division(self):
        assert compute_shard_sizes(8, 3) == [3, 3, 2]
        assert compute_shard_sizes(64, 3) == [22, 21, 21]
        assert compute_shard_sizes(10, 3) == [4, 3, 3]

    def test_sum_invariant(self):
        for total in [7, 8, 64, 100, 255, 2880]:
            for n in [2, 3, 5, 7]:
                sizes = compute_shard_sizes(total, n)
                assert sum(sizes) == total, f"sum({sizes}) != {total}"


class TestWeightSplitMath:
    def test_all_to_sharded_unquantized(self):
        mx.random.seed(RANDOM_SEED)
        weight = mx.random.normal((64, 256))
        x = mx.random.normal((1, 4, 256))

        full_output = x @ weight.T
        mx.eval(full_output)

        for n in [2, 3, 5, 7]:
            sizes = compute_shard_sizes(64, n)
            indices = list(itertools.accumulate(sizes[:-1]))
            shards = mx.split(weight, indices, axis=0)

            reconstructed = mx.concatenate([x @ s.T for s in shards], axis=-1)
            mx.eval(reconstructed)

            diff = float(mx.max(mx.abs(full_output - reconstructed)))
            assert diff < 1e-5, f"all-to-sharded N={n}: diff={diff}"

    def test_sharded_to_all_unquantized(self):
        mx.random.seed(RANDOM_SEED)
        weight = mx.random.normal((128, 256))
        x = mx.random.normal((1, 4, 256))

        full_output = x @ weight.T
        mx.eval(full_output)

        for n in [2, 3, 5, 7]:
            sizes = compute_shard_sizes(256, n)
            w_indices = list(itertools.accumulate(sizes[:-1]))
            x_indices = list(itertools.accumulate(sizes[:-1]))

            w_shards = mx.split(weight, w_indices, axis=-1)
            x_shards = mx.split(x, x_indices, axis=-1)

            partial_outputs = [
                xs @ ws.T for xs, ws in zip(x_shards, w_shards, strict=True)
            ]
            reconstructed = sum(partial_outputs)
            mx.eval(reconstructed)

            diff = float(mx.max(mx.abs(full_output - reconstructed)))
            assert diff < 5e-5, f"sharded-to-all N={n}: diff={diff}"

    def test_all_to_sharded_quantized(self):
        mx.random.seed(RANDOM_SEED)
        weight = mx.random.normal((64, 256))
        group_size = 32
        bits = 4
        qw, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        x = mx.random.normal((1, 4, 256))

        full_output = mx.quantized_matmul(
            x,
            qw,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(full_output)

        for n in [2, 3]:
            sizes = compute_shard_sizes(64, n)
            indices = list(itertools.accumulate(sizes[:-1]))

            qw_shards = mx.split(qw, indices, axis=0)
            scales_shards = mx.split(scales, indices, axis=0)
            biases_shards = mx.split(biases, indices, axis=0)

            partial = [
                mx.quantized_matmul(
                    x,
                    qw_s,
                    scales=sc_s,
                    biases=bi_s,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                )
                for qw_s, sc_s, bi_s in zip(
                    qw_shards, scales_shards, biases_shards, strict=True
                )
            ]
            reconstructed = mx.concatenate(partial, axis=-1)
            mx.eval(reconstructed)

            diff = float(mx.max(mx.abs(full_output - reconstructed)))
            assert diff < 1e-5, f"quantized all-to-sharded N={n}: diff={diff}"

    def test_sharded_to_all_quantized(self):
        mx.random.seed(RANDOM_SEED)
        weight = mx.random.normal((128, 256))
        group_size = 32
        bits = 4
        qw, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        x = mx.random.normal((1, 4, 256))

        full_output = mx.quantized_matmul(
            x,
            qw,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(full_output)

        num_quant_groups = scales.shape[-1]
        for n in [2]:
            # Split in quantization-group space (same as _shard_quantized_s2a)
            group_counts = compute_shard_sizes(num_quant_groups, n)
            weight_ppg = group_size * bits // 32

            packed_sizes = [gc * weight_ppg for gc in group_counts]
            packed_indices = list(itertools.accumulate(packed_sizes[:-1]))
            qw_shards = mx.split(qw, packed_indices, axis=-1)

            scale_indices = list(itertools.accumulate(group_counts[:-1]))
            scales_shards = mx.split(scales, scale_indices, axis=-1)
            biases_shards = mx.split(biases, scale_indices, axis=-1)

            logical_sizes = [gc * group_size for gc in group_counts]
            x_indices = list(itertools.accumulate(logical_sizes[:-1]))
            x_shards = mx.split(x, x_indices, axis=-1)

            partial = [
                mx.quantized_matmul(
                    xs,
                    qw_s,
                    scales=sc_s,
                    biases=bi_s,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                )
                for xs, qw_s, sc_s, bi_s in zip(
                    x_shards, qw_shards, scales_shards, biases_shards, strict=True
                )
            ]
            reconstructed = sum(partial)
            mx.eval(reconstructed)

            diff = float(mx.max(mx.abs(full_output - reconstructed)))
            assert diff < 1e-4, f"quantized sharded-to-all N={n}: diff={diff}"


# Port allocation: 31200-31999 (non-colliding with conftest 29600-29800 and qwen35 29950-31100)
_BASE_PORT = 40000
_port_counter = 0


def _next_port_block():
    global _port_counter
    port = _BASE_PORT + _port_counter * 10
    _port_counter += 1
    return port


@pytest.mark.slow
class TestTensorParallelTP2:
    @pytest.mark.parametrize("model_name", list(REDUCED_CONFIGS.keys()))
    def test_tp2_matches_single(self, model_name):
        config = REDUCED_CONFIGS[model_name]
        single_logits = _run_single(config)
        tp2_logits = _run_tensor(config, world_size=2, base_port=_next_port_block())

        diff = float(np.max(np.abs(single_logits - tp2_logits)))
        assert diff < 3e-6, f"{model_name} tp=2 logit diff: {diff}"


@pytest.mark.slow
class TestTensorParallelTP3:
    @pytest.mark.parametrize("model_name", list(REDUCED_CONFIGS.keys()))
    def test_tp3_matches_single(self, model_name):
        config = REDUCED_CONFIGS[model_name]
        single_logits = _run_single(config)
        tp3_logits = _run_tensor(config, world_size=3, base_port=_next_port_block())

        diff = float(np.max(np.abs(single_logits - tp3_logits)))
        assert diff < 3e-6, f"{model_name} tp=3 logit diff: {diff}"


@pytest.mark.slow
class TestWeightedShardingTP2:
    @pytest.mark.parametrize("model_name", list(REDUCED_CONFIGS.keys()))
    def test_weighted_tp2_matches_single(self, model_name):
        config = REDUCED_CONFIGS[model_name]
        single_logits = _run_single(config)
        tp2_logits = _run_tensor(
            config, world_size=2, base_port=_next_port_block(), shard_weights=[2.0, 1.0]
        )

        diff = float(np.max(np.abs(single_logits - tp2_logits)))
        assert diff < 3e-6, f"{model_name} weighted tp=2 logit diff: {diff}"


@pytest.mark.slow
class TestWeightedShardingTP3:
    @pytest.mark.parametrize("model_name", list(REDUCED_CONFIGS.keys()))
    def test_weighted_tp3_matches_single(self, model_name):
        config = REDUCED_CONFIGS[model_name]
        single_logits = _run_single(config)
        tp3_logits = _run_tensor(
            config,
            world_size=3,
            base_port=_next_port_block(),
            shard_weights=[3.0, 2.0, 1.0],
        )

        diff = float(np.max(np.abs(single_logits - tp3_logits)))
        assert diff < 3e-6, f"{model_name} weighted tp=3 logit diff: {diff}"


@pytest.mark.slow
class TestGreedyShardingTP2:
    @pytest.mark.parametrize("model_name", list(REDUCED_CONFIGS.keys()))
    def test_greedy_tp2_matches_single(self, model_name):
        config = REDUCED_CONFIGS[model_name]
        single_logits = _run_single(config)
        tp2_logits = _run_tensor(
            config,
            world_size=2,
            base_port=_next_port_block(),
            shard_weights=[2.0, 1.0],
            shard_mode=TensorShardMode.Greedy,
        )

        diff = float(np.max(np.abs(single_logits - tp2_logits)))
        assert diff < 3e-6, f"{model_name} greedy tp=2 logit diff: {diff}"
