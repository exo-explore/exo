# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
# pyright: reportUnusedImport=false
"""Test B=1 vs B=2 equivalence for batch generation.

Verifies that running two requests concurrently in a batch (B=2) produces
identical token selections to running them sequentially (B=1).
Uses random weights — no model download required.
"""

from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

# Import batch_generate to activate the right-padding BatchKVCache patch
import mlx_engine.generator.batch_generate  # noqa: F401
import pytest
from mlx_engine.cache import encode_prompt, make_kv_cache
from mlx_engine.generator.generate import prefill
from mlx_engine.types import Model
from mlx_lm.generate import _merge_caches
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

NUM_STEPS = 20


def _init_random(model: nn.Module) -> None:
    """Initialize all model parameters with random values."""
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())


def _run_b1_vs_b2(
    model: Model,
    tokenizer: TokenizerWrapper,
    tokens_a: mx.array,
    tokens_b: mx.array,
) -> tuple[float, int]:
    """Run B=1 sequential and B=2 batched, return (max_diff, mismatches)."""
    sampler = make_sampler(temp=0.0)

    # B=1 sequential
    cache_a1 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_a[:-1], cache_a1, None, None, None)
    merged_a1 = _merge_caches([[c for c in cache_a1]])
    for c in merged_a1:
        c.prepare(lengths=[1], right_padding=[0])
    model(mx.array([[tokens_a[-2].item()]]), cache=merged_a1)
    mx.eval([c.state for c in merged_a1])
    for c in merged_a1:
        c.finalize()

    cache_b1 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_b[:-1], cache_b1, None, None, None)
    merged_b1 = _merge_caches([[c for c in cache_b1]])
    for c in merged_b1:
        c.prepare(lengths=[1], right_padding=[0])
    model(mx.array([[tokens_b[-2].item()]]), cache=merged_b1)
    mx.eval([c.state for c in merged_b1])
    for c in merged_b1:
        c.finalize()

    b1_logits_a: list[mx.array] = []
    b1_logits_b: list[mx.array] = []
    next_a, next_b = tokens_a[-1].item(), tokens_b[-1].item()
    for _ in range(NUM_STEPS):
        la = model(mx.array([[next_a]]), cache=merged_a1)
        mx.eval(la)
        b1_logits_a.append(la[0, -1])
        next_a = int(mx.argmax(la[0, -1]).item())
        lb = model(mx.array([[next_b]]), cache=merged_b1)
        mx.eval(lb)
        b1_logits_b.append(lb[0, -1])
        next_b = int(mx.argmax(lb[0, -1]).item())

    # B=2 batched
    cache_a2 = make_kv_cache(model)
    cache_b2 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_a[:-1], cache_a2, None, None, None)
    prefill(model, tokenizer, sampler, tokens_b[:-1], cache_b2, None, None, None)
    merged_b2 = _merge_caches([list(cache_a2), list(cache_b2)])
    for c in merged_b2:
        c.prepare(lengths=[1, 1], right_padding=[0, 0])
    model(
        mx.array([[tokens_a[-2].item()], [tokens_b[-2].item()]]),
        cache=merged_b2,
    )
    mx.eval([c.state for c in merged_b2])
    for c in merged_b2:
        c.finalize()

    b2_logits_a: list[mx.array] = []
    b2_logits_b: list[mx.array] = []
    next_a2, next_b2 = tokens_a[-1].item(), tokens_b[-1].item()
    for _ in range(NUM_STEPS):
        l2 = model(mx.array([[next_a2], [next_b2]]), cache=merged_b2)
        mx.eval(l2)
        b2_logits_a.append(l2[0, -1])
        b2_logits_b.append(l2[1, -1])
        next_a2 = int(mx.argmax(l2[0, -1]).item())
        next_b2 = int(mx.argmax(l2[1, -1]).item())

    # Compare
    max_diff = 0.0
    mismatches = 0
    for step in range(NUM_STEPS):
        diff_a = float(
            mx.max(
                mx.abs(
                    b1_logits_a[step].astype(mx.float32)
                    - b2_logits_a[step].astype(mx.float32)
                )
            ).item()
        )
        diff_b = float(
            mx.max(
                mx.abs(
                    b1_logits_b[step].astype(mx.float32)
                    - b2_logits_b[step].astype(mx.float32)
                )
            ).item()
        )
        max_diff = max(max_diff, diff_a, diff_b)
        if int(mx.argmax(b1_logits_a[step]).item()) != int(
            mx.argmax(b2_logits_a[step]).item()
        ):
            mismatches += 1
        if int(mx.argmax(b1_logits_b[step]).item()) != int(
            mx.argmax(b2_logits_b[step]).item()
        ):
            mismatches += 1

    return max_diff, mismatches


def _make_tokenizer() -> TokenizerWrapper:
    """Load the Qwen tokenizer (tiny download, shared across Qwen models)."""
    from huggingface_hub import snapshot_download

    model_path = Path(
        snapshot_download(
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            allow_patterns=["tokenizer*", "*.jinja"],
        )
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TokenizerWrapper(hf_tokenizer)


@pytest.mark.slow
def test_batch_b2_llama() -> None:
    """Llama-style model (KVCache only) must produce bit-exact logits in B=2.

    Right-padded BatchKVCache keeps data at position 0 for all sequences,
    so flash attention sees identical data layout as B=1 → bit-exact output.
    """
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution briefly.")

    max_diff, mismatches = _run_b1_vs_b2(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    assert mismatches == 0, f"Llama B=2 token mismatches: {mismatches}/{NUM_STEPS * 2}"
    assert max_diff < 0.002, f"Llama B=2 max logit diff: {max_diff}"


@pytest.mark.slow
def test_batch_b2_qwen35_moe() -> None:
    """Qwen3.5 MoE model (hybrid SSM+attention+MoE) must produce bit-exact logits in B=2.

    Right-padded BatchKVCache keeps data at position 0 for all sequences,
    so flash attention sees identical data layout as B=1 → bit-exact output.
    """
    from mlx_lm.models.qwen3_5_moe import Model as Qwen35MoeModel
    from mlx_lm.models.qwen3_5_moe import ModelArgs

    mx.random.seed(42)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 8,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "head_dim": 64,
            "max_position_embeddings": 4096,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 64,
            "mamba_ssm_dtype": "float32",
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
            },
            "attention_bias": False,
            "attn_output_gate": True,
        },
    }
    args = ModelArgs.from_dict(config)
    model = Qwen35MoeModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution briefly.")

    max_diff, mismatches = _run_b1_vs_b2(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    assert mismatches == 0, (
        f"Qwen3.5 MoE B=2 token mismatches: {mismatches}/{NUM_STEPS * 2}"
    )
    assert max_diff < 0.002, f"Qwen3.5 MoE B=2 max logit diff: {max_diff}"
