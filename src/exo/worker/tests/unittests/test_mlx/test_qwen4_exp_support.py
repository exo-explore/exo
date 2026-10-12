import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as mlx_vlm_utils
import pytest
from mlx_vlm.models.cache import ArraysCache as MLXVLMArrayCache
from mlx_vlm.models.qwen4_exp.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel, QSAKVCache

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    has_non_kv_caches,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.types import Model


class _LanguageModel(nn.Module):
    pass


class _ConditionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()


def _tiny_qwen4_exp() -> utils_mlx.MLXVLMLogitsModel:
    config = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        hc_count=4,
        hc_lowrank=8,
        head_dim=16,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "qwen_sparse_attention",
        ],
        ple_layer_ids=[],
        ngram_size=3,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=16,
        indexer_compress_ratio=4,
        mtp_num_hidden_layers=0,
        eos_token_id=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 2, 4],
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.5,
        },
    )
    return utils_mlx.MLXVLMLogitsModel(
        LanguageModel(
            config,
            ModelConfig(
                text_config=config,
                vision_config=VisionConfig(),
                model_type="qwen4_exp",
            ),
        ),
    )


def _write_config(path: Path, model_type: str) -> dict[str, Any]:
    config = {"model_type": model_type}
    (path / "config.json").write_text(json.dumps(config))
    return config


def test_qwen4_exp_uses_mlx_vlm_language_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_config = _write_config(tmp_path, "qwen4_exp")
    conditional_model = _ConditionalModel()
    seen: dict[str, object] = {}

    def fake_load_model(
        model_path: Path, lazy: bool = False, **kwargs: object
    ) -> nn.Module:
        seen.update(model_path=model_path, lazy=lazy, kwargs=kwargs)
        return conditional_model

    monkeypatch.setattr(mlx_vlm_utils, "load_model", fake_load_model)

    model, config = utils_mlx.load_language_model(tmp_path, lazy=True, strict=False)

    assert isinstance(model, utils_mlx.MLXVLMLogitsModel)
    assert model.language_model is conditional_model.language_model
    assert model.model_path == tmp_path  # type: ignore[attr-defined]
    assert config == expected_config
    assert seen == {
        "model_path": tmp_path,
        "lazy": True,
        "kwargs": {"strict": False},
    }


def test_existing_architectures_stay_on_mlx_lm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_config = _write_config(tmp_path, "llama")
    expected_model = _LanguageModel()
    seen: dict[str, object] = {}

    def fake_load_model(
        model_path: Path, *, lazy: bool, strict: bool
    ) -> tuple[nn.Module, dict[str, Any]]:
        seen.update(model_path=model_path, lazy=lazy, strict=strict)
        return expected_model, expected_config

    monkeypatch.setattr(utils_mlx, "load_mlx_lm_model", fake_load_model)

    model, config = utils_mlx.load_language_model(tmp_path, lazy=False, strict=True)

    assert model is expected_model
    assert config == expected_config
    assert seen == {"model_path": tmp_path, "lazy": False, "strict": True}


def test_qwen4_exp_tiny_prefill_and_decode_maintain_qsa_cache() -> None:
    model = _tiny_qwen4_exp()
    cache = model.make_cache()

    prefill = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(prefill)

    assert prefill.shape == (1, 3, 128)
    qsa_cache = cast(QSAKVCache, cache[-1])
    assert cast(int, qsa_cache.offset) == 3
    assert cast(mx.array | None, qsa_cache.index_keys) is not None

    decode = model(mx.array([[4]]), cache=cache)
    mx.eval(decode)

    assert decode.shape == (1, 1, 128)
    assert cast(int, qsa_cache.offset) == 4
    index_keys = cast(mx.array | None, qsa_cache.index_keys)
    assert index_keys is not None
    assert index_keys.shape[1] == 4


def test_qwen4_exp_cache_snapshot_restores_gdn_and_trims_qsa() -> None:
    model = _tiny_qwen4_exp()
    cache = model.make_cache()

    logits = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(logits)
    assert has_non_kv_caches(cache)

    snapshot = snapshot_ssm_states(cache)
    assert snapshot.token_count == 3
    saved_gdn = cast(MLXVLMArrayCache, snapshot.states[0])

    logits = model(mx.array([[4]]), cache=cache)
    mx.eval(logits)
    trim_cache(cache, 1, snapshot)

    restored_gdn = cast(MLXVLMArrayCache, cache[0])
    for restored, saved in zip(restored_gdn.cache, saved_gdn.cache, strict=True):  # type: ignore[reportUnknownMemberType]
        if saved is not None:
            assert restored is not None
            assert mx.array_equal(restored, saved)
    assert isinstance(cache[-1], QSAKVCache)
    qsa_cache = cache[-1]
    assert cast(int, qsa_cache.offset) == 3
    index_keys = cast(mx.array | None, qsa_cache.index_keys)
    assert index_keys is not None
    assert index_keys.shape[1] == 3


def test_qwen4_exp_prefix_cache_restores_recurrent_and_qsa_state() -> None:
    model = _tiny_qwen4_exp()
    prompt = mx.array([1, 2, 3])
    cache = model.make_cache()

    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    snapshot = snapshot_ssm_states(cache)

    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(prompt, cache, [snapshot])

    restored, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
        cast(Model, cast(object, model)),
        mx.array([1, 2, 3, 4]),
    )

    assert matched_index == 0
    # EXO treats an all-but-final-token match as exact so generation always
    # retains one token to run through the model.
    assert is_exact is True
    assert remaining.tolist() == [4]
    restored_qsa = cast(QSAKVCache, restored[-1])
    assert cast(int, restored_qsa.offset) == 3

    logits = model(remaining[None, :], cache=restored)
    mx.eval(logits)
    assert logits.shape == (1, 1, 128)
    assert cast(int, restored_qsa.offset) == 4


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3.8-Flash-Next",
        "mlx-community/Qwen-3.8-Flash-Next-4bit",
        "local/qwen4_exp-flash-next",
        "local/qwen4-exp-flash-next",
    ],
)
def test_qwen38_eos_tokens(model_id: str) -> None:
    assert utils_mlx.get_eos_token_ids_for_model(ModelId(model_id)) == [
        248046,
        248044,
    ]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "xhigh"),
        ("xhigh", "xhigh"),
    ],
)
def test_qwen38_normalizes_reasoning_effort(requested: str, expected: str) -> None:
    task = TextGenerationTaskParams.model_validate(
        {
            "model": "sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit",
            "input": [{"role": "user", "content": "hello"}],
            "reasoning_effort": requested,
        }
    )

    assert utils_mlx.normalize_chat_template_reasoning_effort(task) == expected
