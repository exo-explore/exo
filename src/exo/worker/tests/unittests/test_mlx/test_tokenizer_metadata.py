import json
from pathlib import Path

from pytest import MonkeyPatch

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.utils_mlx import (
    get_eos_token_ids_for_model,
    resolve_local_model_metadata,
)


def _write_config(model_path: Path, value: dict[str, object]) -> None:
    (model_path / "config.json").write_text(json.dumps(value))


def test_resolves_renamed_kimi_from_outer_metadata(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "eos_token_id": 163586,
            "text_config": {
                "model_type": "kimi_k2",
                "architectures": ["DeepseekV3ForCausalLM"],
            },
        },
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_kimi.TikTokenTokenizer",
                        None,
                    ]
                },
            }
        )
    )

    metadata = resolve_local_model_metadata(tmp_path)

    assert metadata.is_kimi
    assert metadata.outer_model_type == "kimi_k25"
    assert metadata.text_model_type == "kimi_k2"
    assert metadata.uses_kimi_slow_tokenizer
    assert get_eos_token_ids_for_model(ModelId("org/renamed-model"), tmp_path) == [
        163586
    ]


def test_repo_name_cannot_override_non_kimi_metadata(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "eos_token_id": [248046, 248044],
        },
    )

    metadata = resolve_local_model_metadata(tmp_path)

    assert not metadata.is_kimi
    assert get_eos_token_ids_for_model(
        ModelId("org/kimi-k2-distill-qwen"), tmp_path
    ) == [248046, 248044]


def test_generation_and_model_eos_ids_are_deduplicated(tmp_path: Path) -> None:
    _write_config(tmp_path, {"model_type": "glm_moe_dsa", "eos_token_id": [2, 3]})
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [3, 4]})
    )

    assert get_eos_token_ids_for_model(ModelId("org/anything"), tmp_path) == [
        3,
        4,
        2,
    ]


def test_qwen38_eos_ids_come_from_metadata_without_name_allowlist(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "qwen3_8",
            "architectures": ["Qwen3_8ForCausalLM"],
        },
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [248046, 248044]})
    )

    assert get_eos_token_ids_for_model(ModelId("org/renamed-latest-qwen"), tmp_path) == [
        248046,
        248044,
    ]


def test_gemma_metadata_adds_turn_terminators(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "eos_token_id": 1,
        },
    )

    assert get_eos_token_ids_for_model(ModelId("org/renamed"), tmp_path) == [1, 106, 50]


def test_missing_kimi_slow_tokenizer_uses_fast_path(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "eos_token_id": 163586,
        },
    )
    sentinel = object()
    calls: list[tuple[Path, dict[str, object], list[int] | None]] = []

    def fake_load_tokenizer(
        model_path: Path,
        *,
        tokenizer_config_extra: dict[str, object],
        eos_token_ids: list[int] | None,
    ):
        calls.append((model_path, tokenizer_config_extra, eos_token_ids))
        return sentinel

    monkeypatch.setattr(utils_mlx, "load_tokenizer", fake_load_tokenizer)

    result = utils_mlx.load_tokenizer_for_model_id(
        ModelId("org/renamed-kimi"), tmp_path, trust_remote_code=False
    )

    assert result is sentinel
    assert calls == [(tmp_path, {"trust_remote_code": False}, [163586])]


def test_repo_metadata_cannot_enable_remote_tokenizer_code(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_config(
        tmp_path,
        {
            "model_type": "kimi_evil",
            "architectures": ["KimiEvilForCausalLM"],
            "eos_token_id": 7,
        },
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_kimi.TikTokenTokenizer",
                        None,
                    ]
                },
            }
        )
    )
    (tmp_path / "tokenization_kimi.py").write_text(
        "raise AssertionError('must not execute')"
    )
    sentinel = object()

    def fake_load_tokenizer(
        _model_path: Path,
        *,
        tokenizer_config_extra: dict[str, object],
        eos_token_ids: list[int] | None,
    ) -> object:
        del tokenizer_config_extra, eos_token_ids
        return sentinel

    monkeypatch.setattr(utils_mlx, "load_tokenizer", fake_load_tokenizer)

    result = utils_mlx.load_tokenizer_for_model_id(
        ModelId("org/renamed-model"), tmp_path, trust_remote_code=False
    )

    assert result is sentinel


def test_malformed_existing_config_does_not_reactivate_name_heuristics(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{")

    assert (
        get_eos_token_ids_for_model(ModelId("org/kimi-k2-misleading"), tmp_path) is None
    )
