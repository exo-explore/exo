from pydantic import BaseModel, PositiveInt

from exo.shared.types.common import Id
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class ModelId(Id):
    pass


class ModelMetadata(CamelCaseModel):
    model_id: ModelId
    pretty_name: str
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    supports_tensor: bool


class BaseModelConfig(BaseModel):
    """Configuration for a base model (shared across quantization variants).

    This is the schema for entries in registry/base_models.json.
    """

    model_config = {"extra": "forbid", "strict": True}

    id: str  # Short identifier (e.g., "llama-3.1-8b")
    family: str  # Model family (e.g., "llama", "qwen", "deepseek")
    name: str  # Display name (e.g., "Llama 3.1 8B")
    description: str  # Model description
    architecture: str  # HuggingFace model_type (e.g., "llama", "qwen2")
    n_layers: PositiveInt
    hidden_size: PositiveInt
    tagline: str = ""  # Short description (max 60 chars) for UI display
    capabilities: list[str] = []  # e.g., ["text", "thinking", "code", "vision"]


class VariantConfig(BaseModel):
    """Configuration for a quantization variant of a base model.

    This is the schema for entries in registry/variants.json.
    """

    model_config = {"extra": "forbid", "strict": True}

    id: str  # Short identifier (e.g., "llama-3.1-8b-4bit")
    base_model: str  # Reference to base model id
    model_id: str  # HuggingFace repo ID
    quantization: str  # Quantization type (e.g., "4bit", "8bit", "bf16")
    storage_size_bytes: int  # Total model size in bytes


class ModelConfig(BaseModel):
    """JSON-serializable model configuration for built-in and user-added models.

    This is the schema for JSON files in cards/ and ~/.exo/models/.
    Extended with architecture and grouping fields for the new registry structure.
    """

    model_config = {"extra": "forbid", "strict": True}

    model_id: str  # HuggingFace repo ID (e.g., "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    name: str  # Display name (e.g., "Llama 3.1 8B (4-bit)")
    description: str = ""
    tags: list[str] = []
    supports_tensor: bool = False
    storage_size_bytes: int  # Total model size in bytes
    n_layers: PositiveInt
    hidden_size: PositiveInt
    is_user_added: bool = False  # True for models added via dashboard
    # New fields for grouped model support
    architecture: str = ""  # HuggingFace model_type (e.g., "llama", "qwen2")
    base_model_id: str = ""  # Reference to base model (e.g., "llama-3.1-8b")
    base_model_name: str = ""  # Display name of base model (e.g., "Llama 3.1 8B")
    quantization: str = ""  # Quantization type (e.g., "4bit", "8bit")
    # UI display fields
    tagline: str = ""  # Short description (max 60 chars)
    capabilities: list[str] = []  # e.g., ["text", "thinking", "code", "vision"]
    family: str = ""  # Model family (e.g., "llama", "qwen", "deepseek")
