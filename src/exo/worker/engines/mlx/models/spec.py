"""What the master and the API need to know about a model family.

Pure Python: the master runs without MLX (the ``mlx-none`` extra), so nothing here, and
nothing in any family's ``spec.py``, may import mlx, mlx-lm or mlx-vlm.
"""

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class EosRule:
    """Explicit EOS token ids for models whose (lowercased) id contains any of the
    substrings, for checkpoints whose tokenizer config doesn't list them all."""

    model_id_substrings: tuple[str, ...]
    eos_token_ids: tuple[int, ...]


@final
@dataclass(frozen=True)
class PlacementRules:
    """Constraints the master applies when placing an instance of the family."""

    # Tensor parallelism needs num_key_value_heads to divide by the node count, unless the
    # family's strategy doesn't split KV heads.
    tensor_parallel_splits_kv_heads: bool = True
    # Numbers the node count must divide for tensor parallelism, beyond hidden_size and the
    # KV heads: e.g. attention head groups the family's strategy splits whole.
    tensor_parallel_divides: tuple[int, ...] = ()
    # Set when pipeline parallelism across more than one node isn't supported: the error
    # given when no single node can hold the model.
    multi_node_pipeline_error: str | None = None
    # Checkpoints that can't run pipeline-parallel at all: (model id, error message).
    pipeline_unsupported_model_ids: tuple[tuple[str, str], ...] = ()


@final
@dataclass(frozen=True)
class FamilySpec:
    # The family's folder under exo/worker/engines/mlx/models/.
    name: str
    # config.json `model_type` values of the family's checkpoints (as on the model cards).
    model_types: frozenset[str]
    # Hugging Face `architectures` for which a card fetched from Hugging Face gets
    # supports_tensor = true.
    tensor_parallel_architectures: frozenset[str] = frozenset()
    placement: PlacementRules = PlacementRules()
    eos_rules: tuple[EosRule, ...] = ()
    # mlx-vlm module to load the vision tower from, for vision model_types whose module
    # has another name: (model_type, mlx-vlm module).
    mlx_vlm_model_type_aliases: tuple[tuple[str, str], ...] = ()
