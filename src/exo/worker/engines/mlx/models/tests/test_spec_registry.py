"""The family registry's master-side answers equal the code they replaced.

Each `_main_*` function below is copied verbatim from the code that the registry replaced
(exo at 871caad4: model_cards.ConfigData.supports_tensor, master/placement.py,
utils_mlx.get_eos_token_ids_for_model, vision._MLX_VLM_MODEL_TYPE_ALIASES), so these tests
prove the registry kept their behaviour.
"""

import itertools
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ConfigData, ModelCard, ModelId
from exo.worker.engines.mlx.models.spec_registry import (
    FAMILY_SPECS,
    eos_token_ids_for_model_id,
    family_spec_for_model_type,
    mlx_vlm_model_type_aliases,
    placement_rules_for_model_type,
    tensor_parallel_architectures,
)

_MAIN_TENSOR_ARCHITECTURES = [
    ["Glm4MoeLiteForCausalLM"],
    ["GlmMoeDsaForCausalLM"],
    ["DeepseekV4ForCausalLM"],
    ["DeepseekV32ForCausalLM"],
    ["DeepseekV3ForCausalLM"],
    ["Qwen3NextForCausalLM"],
    ["Qwen3MoeForCausalLM"],
    ["Qwen3_5MoeForConditionalGeneration"],
    ["Qwen3_5ForConditionalGeneration"],
    ["Qwen3VLForConditionalGeneration"],
    ["MiniMaxM2ForCausalLM"],
    ["LlamaForCausalLM"],
    ["GptOssForCausalLM"],
    ["Step3p5ForCausalLM"],
    ["NemotronHForCausalLM"],
    ["Gemma4ForConditionalGeneration"],
    # Changed on purpose: Inkling is new.
    ["InklingForConditionalGeneration"],
]

_MAIN_MLX_VLM_MODEL_TYPE_ALIASES: dict[str, str] = {
    "kimi_k25": "kimi_vl",
    "kimi_k26": "kimi_vl",
}


def _main_supports_tensor(architectures: list[str] | None) -> bool:
    return architectures in _MAIN_TENSOR_ARCHITECTURES


def _main_eos_token_ids(model_id: str) -> list[int] | None:
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # For GLM-4.7 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # For Qwen3.5 / Qwen3.6: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    # Changed on purpose: Inkling is new.
    elif "inkling" in model_id_lower:
        return [200006]
    return None


def _main_skips_kv_head_check(card: ModelCard) -> bool:
    return card.base_model.startswith("DeepSeek V4")


def _main_pipeline_error(card: ModelCard) -> str | None:
    if card.model_id == ModelId("mlx-community/DeepSeek-V3.1-8bit"):
        return "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
    return None


def _main_multi_node_pipeline_error(card: ModelCard) -> str | None:
    if card.base_model.startswith("Gemma 4"):
        return "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
    return None


def _builtin_cards() -> list[tuple[Path, ModelCard]]:
    paths = sorted((Path(RESOURCES_DIR) / "inference_model_cards").rglob("*.toml"))
    return [(p, ModelCard.model_validate(tomllib.loads(p.read_text()))) for p in paths]


_CARD_FILES = _builtin_cards()
_CARDS = [card for _, card in _CARD_FILES]


def test_there_are_builtin_cards() -> None:
    assert len(_CARDS) == 125


@pytest.mark.parametrize(
    ("path", "card"), _CARD_FILES, ids=[str(c.model_id) for _, c in _CARD_FILES]
)
def test_every_builtin_card_belongs_to_a_family(path: Path, card: ModelCard) -> None:
    spec = family_spec_for_model_type(card.model_type)
    assert spec is not None, f"{card.model_id}: model_type {card.model_type!r}"
    assert path.parent.name == spec.name


@pytest.mark.parametrize("card", _CARDS, ids=[str(c.model_id) for c in _CARDS])
def test_placement_rules_are_unchanged_for_every_builtin_card(card: ModelCard) -> None:
    rules = placement_rules_for_model_type(card.model_type)

    assert (not rules.tensor_parallel_splits_kv_heads) == _main_skips_kv_head_check(
        card
    )
    pipeline_errors = [
        error
        for model_id, error in rules.pipeline_unsupported_model_ids
        if card.model_id == ModelId(model_id)
    ]
    assert (pipeline_errors[0] if pipeline_errors else None) == _main_pipeline_error(
        card
    )
    assert rules.multi_node_pipeline_error == _main_multi_node_pipeline_error(card)


def test_cards_without_a_model_type_get_no_family_rules() -> None:
    # Custom cards fetched from Hugging Face have no model_type, as they had no base_model.
    assert family_spec_for_model_type(None) is None
    rules = placement_rules_for_model_type(None)
    assert rules.tensor_parallel_splits_kv_heads
    assert rules.multi_node_pipeline_error is None
    assert rules.pipeline_unsupported_model_ids == ()


def test_tensor_parallel_architectures_are_unchanged() -> None:
    assert tensor_parallel_architectures() == frozenset(
        architecture for (architecture,) in _MAIN_TENSOR_ARCHITECTURES
    )


@pytest.mark.parametrize(
    "architectures",
    [
        None,
        [],
        ["LlamaForCausalLM", "MistralForCausalLM"],
        ["Qwen3ForCausalLM"],
        ["Glm4MoeForCausalLM"],
        ["KimiK25ForConditionalGeneration"],
        *_MAIN_TENSOR_ARCHITECTURES,
    ],
)
def test_supports_tensor_is_unchanged(architectures: list[str] | None) -> None:
    config = ConfigData.model_validate(
        {"architectures": architectures, "num_hidden_layers": 2},
        context={"model_id": "test/model"},
    )
    assert config.supports_tensor == _main_supports_tensor(architectures)


_EOS_FRAGMENTS = [
    "kimi-k2",
    "glm-5",
    "glm",
    "gpt-oss",
    "qwen3.5",
    "qwen-3.5",
    "qwen3.6",
    "qwen-3.6",
    "gemma-4",
    "gemma-3",
    "deepseek-v4",
    "llama",
    "inkling",
]
# Every card's id, plus ids that combine two families' rules, to check their precedence.
_EOS_MODEL_IDS = (
    [str(c.model_id) for c in _CARDS]
    + [f"someone/{a.upper()}-{b}" for a, b in itertools.permutations(_EOS_FRAGMENTS, 2)]
    + ["someone/Unrelated-Model", "GLM-4-9B-0414", "google/gemma-3-27b-it"]
)


@pytest.mark.parametrize("model_id", _EOS_MODEL_IDS)
def test_eos_token_ids_are_unchanged(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) == _main_eos_token_ids(model_id)


def test_mlx_vlm_aliases_are_unchanged() -> None:
    assert mlx_vlm_model_type_aliases() == _MAIN_MLX_VLM_MODEL_TYPE_ALIASES


def test_family_names_are_their_folders() -> None:
    families_dir = Path(__file__).resolve().parent.parent
    for spec in FAMILY_SPECS:
        assert (families_dir / spec.name / "spec.py").is_file(), spec.name


def test_master_side_imports_do_not_need_mlx() -> None:
    """The master runs without MLX: the registry's master side must not import it."""
    script = """
import importlib.abc
import sys


class _NoMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("mlx", "mlx_lm", "mlx_vlm"):
            raise ImportError(f"{name} imported on the master side")
        return None


sys.meta_path.insert(0, _NoMlx())
import exo.master.placement
import exo.shared.models.model_cards
import exo.worker.engines.mlx.models.spec_registry
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
