"""Architecture support mapping for tensor parallelism and capabilities.

This module provides a single source of truth for which model architectures
support tensor parallelism and other capabilities. The mapping is derived
from the MLX model classes in exo.worker.engines.mlx.auto_parallel.
"""

from typing import Final

# Model architectures (HuggingFace model_type values) that support tensor parallelism.
# This mapping corresponds to the model classes in auto_parallel.py:
#
# | model_type      | MLX Class           |
# |-----------------|---------------------|
# | llama           | LlamaModel          |
# | mistral         | LlamaModel          |
# | qwen2           | LlamaModel          |
# | ministral3      | Ministral3Model     |
# | deepseek_v3     | DeepseekV3Model     |
# | deepseek_v32    | DeepseekV32Model    |
# | minimax         | MiniMaxModel        |
# | qwen3_moe       | Qwen3MoeModel       |
# | glm4_moe        | Glm4MoeModel        |
# | qwen3_next      | Qwen3NextModel      |
# | gpt_oss         | GptOssModel         |
# | gpt_oss_moe     | GptOssMoeModel      |
#
TENSOR_PARALLEL_ARCHITECTURES: Final[frozenset[str]] = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "ministral3",
        "deepseek_v3",
        "deepseek_v32",
        "minimax",
        "qwen3_moe",
        "glm4_moe",
        "qwen3_next",
        "gpt_oss",
        "gpt_oss_moe",
    }
)

# Model architectures (HuggingFace model_type values) that support vision input.
# These architectures have native image understanding capabilities.
VISION_ARCHITECTURES: Final[frozenset[str]] = frozenset(
    {
        "llava",  # LLaVA vision-language models
        "qwen2_5_vl",  # Qwen 2.5 Vision-Language
        "qwen2_vl",  # Qwen 2 Vision-Language
        "phi4mm",  # Phi-4 multimodal
        "mllama",  # Llama 3.2 Vision (MLlama)
        "paligemma",  # PaLI-GEMMA
        "idefics2",  # IDEFICS2
    }
)


def supports_tensor_parallel(architecture: str) -> bool:
    """Check if an architecture supports tensor parallelism.

    Args:
        architecture: The HuggingFace model_type value (e.g., "llama", "qwen2").

    Returns:
        True if the architecture supports tensor parallelism, False otherwise.
    """
    return architecture.lower() in TENSOR_PARALLEL_ARCHITECTURES


def supports_vision(architecture: str) -> bool:
    """Check if an architecture supports vision/image input.

    Args:
        architecture: The HuggingFace model_type value (e.g., "llava", "qwen2_vl").

    Returns:
        True if the architecture supports vision input, False otherwise.
    """
    return architecture.lower() in VISION_ARCHITECTURES
