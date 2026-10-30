"""The model families, keyed by config.json ``model_type``: the part the master and the API
use (placement rules, which Hugging Face architectures support tensor parallelism, EOS ids).

Importable without MLX. The worker-side registry, with sharding, parsers and encoders, is
``exo.worker.engines.mlx.models.registry``.
"""

from exo.worker.engines.mlx.models.deepseek_v3.spec import SPEC as DEEPSEEK_V3
from exo.worker.engines.mlx.models.deepseek_v4.spec import SPEC as DEEPSEEK_V4
from exo.worker.engines.mlx.models.gemma.spec import SPEC as GEMMA
from exo.worker.engines.mlx.models.glm.spec import SPEC as GLM
from exo.worker.engines.mlx.models.gpt_oss.spec import SPEC as GPT_OSS
from exo.worker.engines.mlx.models.inkling.spec import SPEC as INKLING
from exo.worker.engines.mlx.models.kimi.spec import SPEC as KIMI
from exo.worker.engines.mlx.models.llama.spec import SPEC as LLAMA
from exo.worker.engines.mlx.models.minimax.spec import SPEC as MINIMAX
from exo.worker.engines.mlx.models.nemotron_h.spec import SPEC as NEMOTRON_H
from exo.worker.engines.mlx.models.qwen3.spec import SPEC as QWEN3
from exo.worker.engines.mlx.models.spec import FamilySpec, PlacementRules
from exo.worker.engines.mlx.models.step.spec import SPEC as STEP

# Rules that match on the model id (EOS ids here; prompt encoders and tokenizer loaders in
# the worker registry) are tried family by family in this order, first match wins. The order
# keeps the precedence those rules had before they moved into the family folders.
FAMILY_SPECS: tuple[FamilySpec, ...] = (
    KIMI,
    GLM,
    GPT_OSS,
    QWEN3,
    GEMMA,
    DEEPSEEK_V3,
    DEEPSEEK_V4,
    LLAMA,
    MINIMAX,
    NEMOTRON_H,
    STEP,
    INKLING,
)


def _by_model_type() -> dict[str, FamilySpec]:
    by_model_type: dict[str, FamilySpec] = {}
    for spec in FAMILY_SPECS:
        for model_type in spec.model_types:
            assert model_type not in by_model_type, (
                f"model_type {model_type} is in two families"
            )
            by_model_type[model_type] = spec
    return by_model_type


_BY_MODEL_TYPE = _by_model_type()
_DEFAULT_PLACEMENT_RULES = PlacementRules()


def family_spec_for_model_type(model_type: str | None) -> FamilySpec | None:
    if model_type is None:
        return None
    return _BY_MODEL_TYPE.get(model_type)


def placement_rules_for_model_type(model_type: str | None) -> PlacementRules:
    spec = family_spec_for_model_type(model_type)
    return _DEFAULT_PLACEMENT_RULES if spec is None else spec.placement


def tensor_parallel_architectures() -> frozenset[str]:
    return frozenset(
        architecture
        for spec in FAMILY_SPECS
        for architecture in spec.tensor_parallel_architectures
    )


def eos_token_ids_for_model_id(model_id: str) -> list[int] | None:
    model_id_lower = model_id.lower()
    for spec in FAMILY_SPECS:
        for rule in spec.eos_rules:
            if any(
                substring in model_id_lower for substring in rule.model_id_substrings
            ):
                return list(rule.eos_token_ids)
    return None


def mlx_vlm_model_type_aliases() -> dict[str, str]:
    return {
        model_type: module
        for spec in FAMILY_SPECS
        for model_type, module in spec.mlx_vlm_model_type_aliases
    }
