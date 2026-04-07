# type: ignore
import importlib
import json
import multiprocessing as mp
import os
import tempfile
import traceback
from typing import Any

import mlx.core as mx
import pytest

RANDOM_SEED = 42
INPUT_TOKENS = [1, 100, 200, 300, 400, 500]
MAX_GEN_TOKENS = 50


def _dense_no_kv_shared_config() -> dict[str, Any]:
    return {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "vocab_size_per_layer_input": 1024,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "sliding_window": 32,
        "sliding_window_pattern": 3,
        "max_position_embeddings": 2048,
        "attention_k_eq_v": False,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "tie_word_embeddings": True,
    }


def _moe_config() -> dict[str, Any]:
    return {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "vocab_size_per_layer_input": 1024,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "sliding_window": 32,
        "sliding_window_pattern": 3,
        "max_position_embeddings": 2048,
        "attention_k_eq_v": True,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": True,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 128,
        "tie_word_embeddings": True,
    }


def _kv_shared_config() -> dict[str, Any]:
    return {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "vocab_size_per_layer_input": 1024,
        "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 32,
        "sliding_window": 32,
        "sliding_window_pattern": 3,
        "max_position_embeddings": 2048,
        "attention_k_eq_v": False,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "tie_word_embeddings": True,
    }


def _kv_shared_sources_split_config() -> dict[str, Any]:
    """Config where sources land on layers 1 and 2 (one on each of the first
    two 2-layer ranks). pattern=2 → layer_types =
    [sliding, full, sliding, full, sliding, full]; num_kv_shared_layers=3 →
    non-shared layers are 0,1,2 and the last occurrence of each type in that
    window gives sources = {sliding:2, full:1}. previous_kvs =
    [0, 1, 2, 1, 2, 1]."""
    return {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "vocab_size_per_layer_input": 1024,
        "num_kv_shared_layers": 3,
        "hidden_size_per_layer_input": 32,
        "sliding_window": 32,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 2048,
        "attention_k_eq_v": False,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "tie_word_embeddings": True,
    }


def _kv_shared_sources_first_config() -> dict[str, Any]:
    """Config where both kv-share sources live in layers 0 and 1, so every
    shared layer must read from rank 0. pattern=2 → layer_types =
    [sliding, full, sliding, full, sliding, full]; num_kv_shared_layers=4 →
    non-shared layers are only 0,1 and therefore are the unique sources
    (sliding→0, full→1). previous_kvs = [0, 1, 0, 1, 0, 1]."""
    return {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "vocab_size_per_layer_input": 1024,
        "num_kv_shared_layers": 4,
        "hidden_size_per_layer_input": 32,
        "sliding_window": 32,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 2048,
        "attention_k_eq_v": False,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "tie_word_embeddings": True,
    }


def _build_gemma4_model(text_config: dict[str, Any]):
    mx.random.seed(RANDOM_SEED)
    gemma4 = importlib.import_module("mlx_lm.models.gemma4")
    args = gemma4.ModelArgs.from_dict(
        {
            "model_type": "gemma4",
            "text_config": text_config,
            "vocab_size": text_config["vocab_size"],
        }
    )
    return gemma4.Model(args)


def _create_hostfile(world_size: int, base_port: int) -> str:
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        return f.name


def _greedy_generate(model, prompt_tokens: list[int], n_steps: int) -> list[int]:
    cache = model.make_cache()
    inputs = mx.array([prompt_tokens])
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    next_id = int(mx.argmax(logits[0, -1]).item())
    generated = [next_id]
    for _ in range(n_steps - 1):
        next_inputs = mx.array([[next_id]])
        logits = model(next_inputs, cache=cache)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[0, -1]).item())
        generated.append(next_id)
    return generated


def _run_single(text_config: dict[str, Any], result_queue: Any) -> None:
    try:
        model = _build_gemma4_model(text_config)
        cache = model.make_cache()
        inputs = mx.array([INPUT_TOKENS])
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        last_logits = logits[0, -1]
        tokens = _greedy_generate(
            _build_gemma4_model(text_config), INPUT_TOKENS, MAX_GEN_TOKENS
        )
        result_queue.put(("ok", last_logits.tolist(), tokens))
    except Exception as e:
        result_queue.put(("err", f"{e}\n{traceback.format_exc()}", None))


def _run_tensor(
    rank: int,
    world_size: int,
    hostfile: str,
    text_config: dict[str, Any],
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        group = mx.distributed.init(backend="ring", strict=True)
        model = _build_gemma4_model(text_config)
        model = tensor_auto_parallel(
            model, group, timeout_seconds=60.0, on_timeout=None, on_layer_loaded=None
        )

        cache = model.make_cache()
        inputs = mx.array([INPUT_TOKENS])
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        last_logits = logits[0, -1]

        gen_model = _build_gemma4_model(text_config)
        gen_model = tensor_auto_parallel(
            gen_model,
            group,
            timeout_seconds=60.0,
            on_timeout=None,
            on_layer_loaded=None,
        )
        tokens = _greedy_generate(gen_model, INPUT_TOKENS, MAX_GEN_TOKENS)

        result_queue.put((rank, "ok", last_logits.tolist(), tokens))
    except Exception as e:
        result_queue.put((rank, "err", f"{e}\n{traceback.format_exc()}", None))


def _run_pipeline(
    rank: int,
    world_size: int,
    hostfile: str,
    splits: list[tuple[int, int]],
    text_config: dict[str, Any],
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        from exo.shared.models.model_cards import ModelCard, ModelTask
        from exo.shared.types.common import ModelId
        from exo.shared.types.memory import Memory
        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel

        group = mx.distributed.init(backend="ring", strict=True)
        start, end = splits[rank]

        n_layers = text_config["num_hidden_layers"]
        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("test/gemma4-test"),
                storage_size=Memory.from_gb(1),
                n_layers=n_layers,
                hidden_size=text_config["hidden_size"],
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start,
            end_layer=end,
            n_layers=n_layers,
        )

        def _build():
            model = _build_gemma4_model(text_config)
            return pipeline_auto_parallel(
                model, group, shard_meta, on_layer_loaded=None
            )

        model = _build()
        cache = model.make_cache()
        inputs = mx.array([INPUT_TOKENS])
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        last_logits = logits[0, -1]

        gen_model = _build()
        tokens = _greedy_generate(gen_model, INPUT_TOKENS, MAX_GEN_TOKENS)

        result_queue.put((rank, "ok", last_logits.tolist(), tokens))
    except Exception as e:
        result_queue.put((rank, "err", f"{e}\n{traceback.format_exc()}", None))


def _spawn_single(text_config: dict[str, Any]) -> tuple[list[float], list[int]]:
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    p = ctx.Process(target=_run_single, args=(text_config, queue))
    p.start()
    p.join(timeout=120)
    status, payload, tokens = queue.get(timeout=5)
    if status != "ok":
        raise RuntimeError(f"single device failed: {payload}")
    return payload, tokens


def _spawn_distributed(
    target,
    args_per_rank: list[tuple],
    timeout: float = 240.0,
) -> dict[int, tuple[list[float], list[int]]]:
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    procs = []
    for rank_args in args_per_rank:
        p = ctx.Process(target=target, args=(*rank_args, queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout)

    results: dict[int, tuple[list[float], list[int]]] = {}
    errors: dict[int, str] = {}
    while not queue.empty():
        rank, status, payload, tokens = queue.get()
        if status == "ok":
            results[rank] = (payload, tokens)
        else:
            errors[rank] = payload
    if errors:
        raise RuntimeError(f"distributed run errors: {errors}")
    return results


@pytest.mark.slow
def test_dense_tensor_matches_single() -> None:
    text_config = _dense_no_kv_shared_config()
    base_port = 31200

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_tensor,
            [(0, 2, hostfile, text_config), (1, 2, hostfile, text_config)],
        )
    finally:
        os.unlink(hostfile)

    for rank in range(2):
        rank_logits, rank_tokens = results[rank]
        diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
        assert diff < 5e-5, f"rank {rank} logit diff {diff}"
        assert rank_tokens == single_tokens, (
            f"rank {rank} tokens {rank_tokens} != single {single_tokens}"
        )


@pytest.mark.slow
def test_dense_pipeline_matches_single() -> None:
    text_config = _dense_no_kv_shared_config()
    base_port = 31220
    n_layers = text_config["num_hidden_layers"]
    splits = [(0, n_layers // 2), (n_layers // 2, n_layers)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[1]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"pipeline rank 1 logit diff {diff}"
    assert rank_tokens == single_tokens, (
        f"pipeline tokens {rank_tokens} != single {single_tokens}"
    )


@pytest.mark.slow
def test_dense_pipeline_asymmetric() -> None:
    text_config = _dense_no_kv_shared_config()
    base_port = 31240
    splits = [(0, 2), (2, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[1]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"asymmetric pipeline logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_moe_tensor_matches_single() -> None:
    text_config = _moe_config()
    base_port = 31260

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_tensor,
            [(0, 2, hostfile, text_config), (1, 2, hostfile, text_config)],
        )
    finally:
        os.unlink(hostfile)

    for rank in range(2):
        rank_logits, rank_tokens = results[rank]
        diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
        assert diff < 5e-5, f"moe tensor rank {rank} logit diff {diff}"
        assert rank_tokens == single_tokens


@pytest.mark.slow
def test_moe_pipeline_matches_single() -> None:
    text_config = _moe_config()
    base_port = 31280
    n_layers = text_config["num_hidden_layers"]
    splits = [(0, n_layers // 2), (n_layers // 2, n_layers)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[1]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"moe pipeline logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_valid_split() -> None:
    """KV-shared layers stay with their source — split (0,2),(2,6) keeps both shared sources on rank 1."""
    text_config = _kv_shared_config()
    base_port = 31300
    splits = [(0, 2), (2, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[1]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"kv_shared pipeline logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_split_separates_source_from_shared() -> None:
    """Split (0,3),(3,6) puts source layer 2 on rank 0 but shared layer 5 (which
    reads layer 2) on rank 1. Cross-rank kv transfer should make this work."""
    text_config = _kv_shared_config()
    base_port = 31320
    splits = [(0, 3), (3, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[1]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"kv_shared cross-rank logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_3node_both_sources_remote() -> None:
    """3-rank split (0,2),(2,4),(4,6). Both sources (2 and 3) live on rank 1;
    rank 2's shared layers (4 and 5) need to receive both via cross-rank kv
    transfer."""
    text_config = _kv_shared_config()
    base_port = 31340
    splits = [(0, 2), (2, 4), (4, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(3, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 3, hostfile, splits, text_config),
                (1, 3, hostfile, splits, text_config),
                (2, 3, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[2]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"kv_shared 3-rank logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_3node_multi_hop() -> None:
    """3-rank split (0,3),(3,4),(4,6). Source layer 2 lives on rank 0, source
    layer 3 lives on rank 1. Rank 2's shared layers need both, so source 2 must
    be forwarded through rank 1 to rank 2 (multi-hop forwarding via the
    received-bundle pass-through)."""
    text_config = _kv_shared_config()
    base_port = 31360
    splits = [(0, 3), (3, 4), (4, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(3, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 3, hostfile, splits, text_config),
                (1, 3, hostfile, splits, text_config),
                (2, 3, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[2]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"kv_shared multi-hop logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_sources_rank0_consumed_by_all() -> None:
    """3-rank split (0,2),(2,4),(4,6). Sources (layers 0, 1) live entirely on
    rank 0. Rank 1 (layers 2, 3) and rank 2 (layers 4, 5) are ALL shared
    layers, and every shared layer reads from one of rank 0's sources."""
    text_config = _kv_shared_sources_first_config()
    base_port = 31380
    splits = [(0, 2), (2, 4), (4, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(3, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 3, hostfile, splits, text_config),
                (1, 3, hostfile, splits, text_config),
                (2, 3, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[2]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"sources-rank0 logit diff {diff}"
    assert rank_tokens == single_tokens


@pytest.mark.slow
def test_kv_shared_pipeline_sources_split_across_first_two_ranks() -> None:
    """3-rank split (0,2),(2,4),(4,6). previous_kvs = [0, 1, 2, 1, 2, 1] puts
    the two sources at layers 1 (rank 0) and 2 (rank 1)."""
    text_config = _kv_shared_sources_split_config()
    base_port = 31400
    splits = [(0, 2), (2, 4), (4, 6)]

    single_logits, single_tokens = _spawn_single(text_config)

    hostfile = _create_hostfile(3, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline,
            [
                (0, 3, hostfile, splits, text_config),
                (1, 3, hostfile, splits, text_config),
                (2, 3, hostfile, splits, text_config),
            ],
        )
    finally:
        os.unlink(hostfile)

    rank_logits, rank_tokens = results[2]
    diff = max(abs(a - b) for a, b in zip(single_logits, rank_logits, strict=True))
    assert diff < 5e-5, f"sources-split logit diff {diff}"
    assert rank_tokens == single_tokens


def _e2b_like_config() -> dict[str, Any]:
    """Exact gemma-4-e2b text_config (30 layers, 10 kv-shared,
    hidden_size_per_layer_input=256, sliding_window=512, explicit layer_types)."""
    pattern = ["sliding_attention"] * 4 + ["full_attention"]
    layer_types = pattern * 6
    return {
        "model_type": "gemma4_text",
        "hidden_size": 1536,
        "num_hidden_layers": 30,
        "intermediate_size": 8192,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "num_kv_shared_layers": 10,
        "hidden_size_per_layer_input": 256,
        "sliding_window": 512,
        "sliding_window_pattern": 5,
        "layer_types": layer_types,
        "max_position_embeddings": 32768,
        "attention_k_eq_v": False,
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "enable_moe_block": False,
        "tie_word_embeddings": True,
    }


def _load_gemma4_tokenizer():  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("mlx-community/gemma-4-e2b-it-4bit")


def _run_pipeline_via_prefill(
    rank: int,
    world_size: int,
    hostfile: str,
    splits: list[tuple[int, int]],
    text_config: dict[str, Any],
    result_queue: Any,
) -> None:
    """End-to-end reproducer for the warmup hang: load the real
    mlx-community/gemma-4-e2b-it-4bit model + tokenizer, shard it across a
    2-rank pipeline, and drive it through exo's `warmup_inference` which
    is the *exact same entrypoint* runner.main calls before serving."""
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        from mlx_lm.utils import load

        from exo.shared.models.model_cards import ModelCard, ModelTask
        from exo.shared.types.common import ModelId
        from exo.shared.types.memory import Memory
        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel
        from exo.worker.engines.mlx.generator.generate import warmup_inference

        group = mx.distributed.init(backend="ring", strict=True)
        start, end = splits[rank]

        model, tokenizer = load("mlx-community/gemma-4-e2b-it-4bit")
        n_layers = len(model.layers)
        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("mlx-community/gemma-4-e2b-it-4bit"),
                storage_size=Memory.from_gb(3),
                n_layers=n_layers,
                hidden_size=1536,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start,
            end_layer=end,
            n_layers=n_layers,
        )
        model = pipeline_auto_parallel(model, group, shard_meta, on_layer_loaded=None)

        tokens_generated = warmup_inference(
            model=model,
            tokenizer=tokenizer,
            group=group,
            model_id=ModelId("mlx-community/gemma-4-e2b-it-4bit"),
        )

        result_queue.put((rank, "ok", [], [tokens_generated]))
    except Exception as e:
        result_queue.put((rank, "err", f"{e}\n{traceback.format_exc()}", None))


@pytest.mark.slow
def test_e2b_like_pipeline_warmup_path() -> None:
    """Runs exo's real prefill+decode path on an e2b-shaped config in a
    2-rank pipeline split. Catches bugs that only show up when the cache
    goes through trim/snapshot-restore, per_layer_inputs is active, and
    the shard boundary crosses mixed cache types."""
    text_config = _e2b_like_config()
    base_port = 31420
    n_layers = text_config["num_hidden_layers"]
    splits = [(0, n_layers // 2), (n_layers // 2, n_layers)]

    hostfile = _create_hostfile(2, base_port)
    try:
        results = _spawn_distributed(
            _run_pipeline_via_prefill,
            [
                (0, 2, hostfile, splits, text_config),
                (1, 2, hostfile, splits, text_config),
            ],
            timeout=120.0,
        )
    finally:
        os.unlink(hostfile)

    # Reaching this point means both ranks completed `warmup_inference`
    # end-to-end (prefill + decode) without hanging or raising — that's the
    # only thing this test is here to prove.
    assert 0 in results and 1 in results, f"missing rank results: {list(results)}"
