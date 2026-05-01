from exo.shared.types.common import ModelFileFormat, ModelId, ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import FrozenModel


class LocalModelEntry(FrozenModel):
    """A model that exists on a worker's local disk, regardless of which tool put it there."""

    node_id: NodeId
    source: ModelSourceKind
    external_id: str
    """Natural identifier in the source's namespace (e.g. ``mlx-community/Llama-3.1-8B-Instruct-4bit``,
    ``llama3:8b``). Not necessarily unique across sources — pair with ``source`` for a stable key."""

    display_name: str
    path: str
    """Absolute path to the model directory (HF/MLX/LMStudio MLX) or to the weight file (GGUF)."""

    format: ModelFileFormat
    size_bytes: Memory

    loadable_with_mlx: bool = False
    """True if exo's MLX engine can load this entry as-is (directory of safetensors + config.json)."""

    matched_model_id: ModelId | None = None
    """If this entry corresponds to an exo-known model card, the canonical exo ``ModelId``."""
