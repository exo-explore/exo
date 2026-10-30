"""The worker-side family registry picks what the code it replaced picked.

The `_main_*` functions are copied from exo at 871caad4 (auto_parallel.tensor_auto_parallel,
auto_parallel._set_layers, model_output_parsers.apply_all_parsers and the model-id checks in
utils_mlx), with each branch returning what it chose instead of doing it. Every mlx-lm model
module is checked against them, so a model type the old isinstance chains handled through
subclassing can't be lost either.
"""

import importlib
import pkgutil
import tomllib
import types
from pathlib import Path
from typing import cast

import mlx.nn as nn
import mlx_lm.models
import pytest
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gemma4 import Model as Gemma4Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import GptOssMoeModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import NemotronHModel as NemotronHInnerModel
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import Qwen3NextModel as Qwen3NextInnerModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel

from exo.shared.constants import RESOURCES_DIR
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_deepseek_v32
from exo.worker.engines.mlx.models.deepseek_v4.model import Model as DeepseekV4Model
from exo.worker.engines.mlx.models.deepseek_v4.model import (
    ModelArgs as DeepseekV4ModelArgs,
)
from exo.worker.engines.mlx.models.deepseek_v4.parser import parse_deepseek_v4
from exo.worker.engines.mlx.models.family import ModelClasses
from exo.worker.engines.mlx.models.gpt_oss.parser import parse_gpt_oss
from exo.worker.engines.mlx.models.gpt_oss.sharding import (
    fix_pipeline_shard as gpt_oss_fixup,
)
from exo.worker.engines.mlx.models.hy_v3.model import Model as HyV3Model
from exo.worker.engines.mlx.models.nemotron_h.sharding import (
    fix_pipeline_shard as nemotron_h_fixup,
)
from exo.worker.engines.mlx.models.qwen3.sharding import (
    fix_pipeline_shard as qwen3_fixup,
)
from exo.worker.engines.mlx.models.spec_registry import FAMILY_SPECS
from exo.worker.engines.mlx.models.step.sharding import fix_pipeline_shard as step_fixup
from exo.worker.engines.mlx.sharding.pipeline import PipelineShardFixup


def _main_tensor_strategy(model_class: type) -> str | None:
    if issubclass(model_class, (LlamaModel, Ministral3Model)):
        return "LlamaShardingStrategy"
    elif issubclass(model_class, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        return "DeepSeekShardingStrategy"
    elif issubclass(model_class, DeepseekV4Model):
        return "DeepseekV4ShardingStrategy"
    elif issubclass(model_class, MiniMaxModel):
        return "MiniMaxShardingStrategy"
    elif issubclass(model_class, GLM4MoeLiteModel):
        return "GLM4MoeLiteShardingStrategy"
    elif issubclass(model_class, Glm4MoeModel):
        return "Glm4MoeShardingStrategy"
    elif issubclass(
        model_class,
        (
            Qwen3Model,
            Qwen3MoeModel,
            Qwen3NextModel,
            Qwen3_5TextModel,
            Qwen3_5MoeModel,
            Qwen3VLModel,
        ),
    ):
        return "QwenShardingStrategy"
    elif issubclass(model_class, GptOssModel):
        return "GptOssShardingStrategy"
    elif issubclass(model_class, Step35Model):
        return "Step35ShardingStrategy"
    elif issubclass(model_class, NemotronHModel):
        return "NemotronHShardingStrategy"
    elif issubclass(model_class, Gemma4Model):
        return "Gemma4ShardingStrategy"
    # Changed on purpose: Hy3 is new (exo's own port).
    elif issubclass(model_class, HyV3Model):
        return "HyV3ShardingStrategy"
    else:
        return None


def _main_layer_count(model_class: type) -> str | None:
    if issubclass(
        model_class,
        (
            DeepseekV3Model,
            DeepseekV32Model,
            DeepseekV4Model,
            Glm4MoeModel,
            KimiK25Model,
        ),
    ):
        return "num_layers"
    elif issubclass(model_class, Qwen3MoeModel):
        return "num_hidden_layers"
    return None


# (parser, whether the thinking parser runs first); None = the generic parsers.
def _main_output_parser(
    model_class: type, normalized_id: str
) -> tuple[str, bool] | None:
    if issubclass(model_class, GptOssModel):
        return ("parse_gpt_oss", False)
    elif issubclass(model_class, DeepseekV32Model) and "deepseek" in normalized_id:
        return ("parse_deepseek_v32", True)
    elif issubclass(model_class, DeepseekV4Model) and "deepseek-v4" in normalized_id:
        return ("parse_deepseek_v4", True)
    return None


def _main_prompt_encoder(model: str) -> str | None:
    if "deepseek-v3.2" in model.lower():
        return "deepseek_v32"
    if "deepseek-v4" in model.lower():
        return "deepseek_v4"
    return None


def _main_prepares_gpt_oss_messages(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _main_loads_kimi_tokenizer(model_id: str) -> bool:
    return "kimi-k2" in model_id.lower()


def _loads_hy3_tokenizer(model_id: str) -> bool:
    # Changed on purpose: Hy3 is new, and its tokenizer needs its reasoning tags and
    # tool-call parser.
    name = model_id.rsplit("/", 1)[-1].lower()
    return name == "hy3" or name.startswith(("hy3-", "hy3_"))


# The pipeline fixups each checked isinstance on the inner model.
_FIXUP_FOR_INNER_CLASS: list[tuple[type, PipelineShardFixup]] = [
    (GptOssMoeModel, gpt_oss_fixup),
    (Step35InnerModel, step_fixup),
    (Qwen3_5TextModelInner, qwen3_fixup),
    (Qwen3NextInnerModel, qwen3_fixup),
    (NemotronHInnerModel, nemotron_h_fixup),
]


def _mlx_lm_model_classes() -> dict[str, type]:
    classes: dict[str, type] = {}
    for module_info in pkgutil.iter_modules(mlx_lm.models.__path__):
        try:
            module = importlib.import_module(f"mlx_lm.models.{module_info.name}")
        except (
            BaseException
        ):  # some modules exit when an optional dependency is missing
            continue
        model_class = cast(object, getattr(module, "Model", None))
        if isinstance(model_class, type):
            classes[module_info.name] = model_class
    # Model types whose model a family ships itself, because mlx-lm has none
    for model_type in ("deepseek_v4", "hy_v3"):
        assert model_type not in classes
        classes[model_type] = registry.get_model_classes({"model_type": model_type})[0]
    return classes


_MLX_LM_MODEL_CLASSES = _mlx_lm_model_classes()
_MODEL_TYPES = sorted(_MLX_LM_MODEL_CLASSES)

_SUPPORTED_MODEL_TYPES = [
    "deepseek_v3",
    "deepseek_v32",
    "deepseek_v4",
    "gemma4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm_moe_dsa",
    "gpt_oss",
    "hy_v3",
    "kimi_k25",
    "llama",
    "minimax",
    "ministral3",
    "nemotron_h",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_vl",
    "step3p5",
]


def _strategy_name(model_class: type) -> str | None:
    architecture = registry.architecture_for_model_class(model_class)
    if architecture is None or architecture.tensor_parallel is None:
        return None
    return architecture.tensor_parallel.__name__


def test_every_supported_model_type_has_a_tensor_parallel_strategy() -> None:
    for model_type in _SUPPORTED_MODEL_TYPES:
        model_class = _MLX_LM_MODEL_CLASSES[model_type]
        assert _strategy_name(model_class) is not None, model_type
    assert len(_MODEL_TYPES) > 100


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_tensor_parallel_strategy_is_unchanged(model_type: str) -> None:
    model_class = _MLX_LM_MODEL_CLASSES[model_type]
    assert _strategy_name(model_class) == _main_tensor_strategy(model_class)


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_pipeline_layer_count_is_unchanged(model_type: str) -> None:
    model_class = _MLX_LM_MODEL_CLASSES[model_type]
    architecture = registry.architecture_for_model_class(model_class)
    layer_count = None if architecture is None else architecture.pipeline_layer_count
    assert layer_count == _main_layer_count(model_class)


def _card_ids() -> list[str]:
    paths = (Path(RESOURCES_DIR) / "inference_model_cards").rglob("*.toml")
    return [cast(str, tomllib.loads(p.read_text())["model_id"]) for p in paths]


_MODEL_IDS = _card_ids() + [
    "someone/My-Finetune",
    "someone/deepseek-distill-GLM-5",
    "someone/GLM-5-deepseek-v4-merge",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "someone/gpt-oss-deepseek-v3.2-kimi-k2",
    "moonshotai/Kimi-K2-Instruct",
    "mlx-community/DeepSeek-V3.2-Speciale-4bit",
]


def _parser_name(parse: object) -> str:
    for name, function in (
        ("parse_gpt_oss", parse_gpt_oss),
        ("parse_deepseek_v32", parse_deepseek_v32),
        ("parse_deepseek_v4", parse_deepseek_v4),
    ):
        if parse is function:
            return name
    raise AssertionError(f"unknown output parser {parse}")


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_output_parser_choice_is_unchanged(model_type: str) -> None:
    model_class = _MLX_LM_MODEL_CLASSES[model_type]
    architecture = registry.architecture_for_model_class(model_class)
    output_parser = None if architecture is None else architecture.output_parser
    for model_id in _MODEL_IDS:
        normalized_id = ModelId(model_id).normalize().lower()
        chosen = (
            None
            if output_parser is None
            or not output_parser.applies_to_model_id(normalized_id)
            else (
                _parser_name(output_parser.parse),
                output_parser.after_thinking_parser,
            )
        )
        assert chosen == _main_output_parser(model_class, normalized_id), model_id


def _reachable_classes(model_type: str) -> set[type]:
    """Classes the module (or the module of a base class of its Model) can build: the
    classes in its namespace and in the namespaces of mlx-lm modules it imports."""
    model_class = _MLX_LM_MODEL_CLASSES[model_type]
    modules = {
        cls.__module__
        for cls in model_class.__mro__
        if cls.__module__.startswith("mlx_lm.")
    }
    reachable: set[type] = set()
    for module_name in modules:
        namespace = cast(dict[str, object], vars(importlib.import_module(module_name)))
        for value in namespace.values():
            candidates: list[object] = [value]
            if isinstance(value, types.ModuleType) and value.__name__.startswith(
                "mlx_lm.models."
            ):
                candidates = list(vars(value).values())
            reachable.update(c for c in candidates if isinstance(c, type))
    return reachable


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_pipeline_fixups_cover_every_model_type_they_applied_to(
    model_type: str,
) -> None:
    """The old fixups ran for any model whose inner model was of their class. Each must
    now be the fixup of every model type that can build such an inner model."""
    reachable = _reachable_classes(model_type)
    expected = {
        fixup
        for inner_class, fixup in _FIXUP_FOR_INNER_CLASS
        if any(issubclass(cls, inner_class) for cls in reachable)
    }
    architecture = registry.architecture_for_model_class(
        _MLX_LM_MODEL_CLASSES[model_type]
    )
    fixup = None if architecture is None else architecture.fix_pipeline_shard
    assert expected == ({fixup} if fixup is not None else set())


def test_architectures_are_keyed_by_their_mlx_lm_model_type() -> None:
    for family in registry.FAMILIES:
        shipped = {c.model_type: c.model_class for c in family.model_classes}
        for architecture in family.architectures:
            if architecture.model_type in shipped:
                # The family's own model.py, for a model mlx-lm doesn't have
                assert architecture.model_class is shipped[architecture.model_type]
            else:
                assert (
                    architecture.model_class.__module__
                    == f"mlx_lm.models.{architecture.model_type}"
                )
            assert architecture.model_type in family.spec.model_types
    assert tuple(family.spec for family in registry.FAMILIES) == FAMILY_SPECS


def _task(model: str) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_model_id_hooks_are_unchanged(model_id: str) -> None:
    task = _task(model_id)

    encoder = registry.prompt_encoder_for(task)
    families_with_encoders = {
        family.spec.name: family.prompt_encoder for family in registry.FAMILIES
    }
    expected = _main_prompt_encoder(model_id)
    if expected is None:
        assert encoder is None
    else:
        family_name = {"deepseek_v32": "deepseek_v3", "deepseek_v4": "deepseek_v4"}[
            expected
        ]
        assert encoder is families_with_encoders[family_name]

    hooks = registry.template_messages_hooks_for(task)
    assert len(hooks) == (1 if _main_prepares_gpt_oss_messages(model_id) else 0)

    loader = registry.tokenizer_loader_for(ModelId(model_id))
    assert (loader is not None) == (
        _main_loads_kimi_tokenizer(model_id) or _loads_hy3_tokenizer(model_id)
    )


@pytest.mark.parametrize(
    ("model_type", "module"),
    [
        ("llama", "llama"),
        ("mistral", "llama"),
        ("kimi_k2", "deepseek_v3"),
        ("minimax_m2", "minimax"),
        ("qwen3_5_moe", "qwen3_5_moe"),
    ],
)
def test_model_classes_default_to_mlx_lm(model_type: str, module: str) -> None:
    model_class, model_args_class = registry.get_model_classes(
        {"model_type": model_type}
    )
    mlx_lm_module = importlib.import_module(f"mlx_lm.models.{module}")
    assert model_class is cast(type, mlx_lm_module.Model)
    assert model_args_class is cast(type, mlx_lm_module.ModelArgs)


def test_deepseek_v4_loads_with_the_model_in_its_family_folder() -> None:
    assert registry.get_model_classes({"model_type": "deepseek_v4"}) == (
        DeepseekV4Model,
        DeepseekV4ModelArgs,
    )


def test_a_family_can_provide_its_own_model_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MyModel(nn.Module):
        pass

    class MyModelArgs:
        pass

    monkeypatch.setitem(
        registry._MODEL_CLASSES_BY_MODEL_TYPE,  # pyright: ignore[reportPrivateUsage]
        "my_model",
        ModelClasses("my_model", MyModel, MyModelArgs),
    )
    assert registry.get_model_classes({"model_type": "my_model"}) == (
        MyModel,
        MyModelArgs,
    )
    with pytest.raises(ValueError, match="not supported"):
        registry.get_model_classes({"model_type": "no_such_model_type"})
