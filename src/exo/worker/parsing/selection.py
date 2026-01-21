"""Parser configuration selection for different model types.

This module provides intelligent selection of parser configurations based on
model classes and model identifiers. It implements a tiered approach:

1. Model class detection (highest confidence): Uses isinstance checks for MLX model classes
2. Model ID fallback (legacy): Uses string matching for backward compatibility

The selection prioritizes robustness over simplicity, choosing parser configurations
based on the most reliable information available.
"""

from typing import Any

from exo.worker.parsing.stream import ChunkParserConfig


def select_chunk_parser_config(
    *,
    model_id: str,
    model: object | None = None,
) -> ChunkParserConfig:
    """Select the appropriate chunk parser configuration for a model.

    This function implements a two-tier selection strategy:
    1. Try model class-based detection for reliable, type-safe selection
    2. Fall back to model ID string matching for compatibility

    Args:
        model_id: The model identifier string (e.g., "mlx-community/GPT-OSS-7B")
        model: The loaded MLX model instance, or None if not available

    Returns:
        ChunkParserConfig: Parser configuration with reasoning_parser_name,
                          tool_parser_name, and enable_thinking settings
    """
    if model is not None:
        config = _select_by_model_class(model, model_id)
        if config is not None:
            return config

    # Fall back to model ID-based selection
    return _select_by_model_id(model_id)


def _select_by_model_id(model_id: str) -> ChunkParserConfig:
    """Select parser configuration using model ID string matching.

    This is the legacy fallback method that uses substring matching to determine
    the appropriate parser configuration. It's less robust than model class
    detection but provides compatibility with models that don't have distinct
    MLX classes or when model instances aren't available.

    Args:
        model_id: The model identifier string

    Returns:
        ChunkParserConfig: Parser configuration based on string patterns
    """
    lower_id = (model_id or "").lower()

    if "harmony" in lower_id or "gpt-oss" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="harmony",
            tool_parser_name="harmony",
            enable_thinking=True,
        )

    if "solar" in lower_id and "open" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="solar_open",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "minimax" in lower_id or "m2" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="minimax_m2",
            tool_parser_name="minimax_m2",
            enable_thinking=True,
        )

    if "glm" in lower_id or "chatglm" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="glm4_moe",
            tool_parser_name="glm47",
            enable_thinking=True,
        )

    if "nemotron" in lower_id and ("nano" in lower_id or "3" in lower_id):
        return ChunkParserConfig(
            reasoning_parser_name="nemotron3_nano",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "qwen3" in lower_id:
        if "vl" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_vl",
                tool_parser_name=None,
                enable_thinking=True,
            )

        if "moe" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_moe",
                tool_parser_name=None,
                enable_thinking=True,
            )

        if "coder" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3",
                tool_parser_name="qwen3_coder",
                enable_thinking=True,
            )

        return ChunkParserConfig(
            reasoning_parser_name="qwen3",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "functiongemma" in lower_id or ("gemma" in lower_id and "function" in lower_id):
        return ChunkParserConfig(
            reasoning_parser_name=None,
            tool_parser_name="function_gemma",
            enable_thinking=True,
        )

    if "hermes" in lower_id or "tool-use" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="hermes",
            tool_parser_name="json_tools",
            enable_thinking=True,
        )

    return ChunkParserConfig(
        reasoning_parser_name=None,
        tool_parser_name=None,
        enable_thinking=True,
    )


def _select_by_model_class(model: object, model_id: str) -> ChunkParserConfig | None:
    """Select parser configuration using MLX model class detection.

    This function uses isinstance checks against MLX model classes to provide
    robust, type-safe parser selection. It implements two categories of rules:

    - Class-only detection: For models with distinct MLX classes and stable
      parsing requirements across fine-tunes
    - Hybrid detection: For models that share base classes but require
      model ID refinement for specific variants

    Args:
        model: The loaded MLX model instance
        model_id: The model identifier string for refinement

    Returns:
        ChunkParserConfig | None: Parser configuration if confident match found,
                                 None to fall back to model ID selection
    """
    lower_id = (model_id or "").lower()

    # GPT-OSS: class-only detection (highest confidence)
    try:
        from mlx_lm.models.gpt_oss import Model as GptOssModel

        if isinstance(model, GptOssModel):
            return ChunkParserConfig(
                reasoning_parser_name="harmony",
                tool_parser_name="harmony",
                enable_thinking=True,
            )
    except ImportError:
        pass

    # GptOssMoeModel: class-only detection
    try:
        from mlx_lm.models.gpt_oss import GptOssMoeModel

        if isinstance(model, GptOssMoeModel):
            return ChunkParserConfig(
                reasoning_parser_name="harmony",
                tool_parser_name="harmony",
                enable_thinking=True,
            )
    except ImportError:
        pass

    # MiniMax: class-only detection
    try:
        from mlx_lm.models.minimax import Model as MiniMaxModel

        if isinstance(model, MiniMaxModel):
            return ChunkParserConfig(
                reasoning_parser_name="minimax_m2",
                tool_parser_name="minimax_m2",
                enable_thinking=True,
            )
    except ImportError:
        pass

    # Qwen3 VL: class-only detection
    try:
        from mlx_lm.models.qwen3_vl import Model as Qwen3VlModel

        if isinstance(model, Qwen3VlModel):
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_vl",
                tool_parser_name=None,
                enable_thinking=True,
            )
    except ImportError:
        pass

    # Qwen3 MoE: class-only detection
    try:
        from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel

        if isinstance(model, Qwen3MoeModel):
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_moe",
                tool_parser_name=None,
                enable_thinking=True,
            )
    except ImportError:
        pass

    # Qwen3 base: hybrid detection with model ID refinement for Coder variants
    try:
        from mlx_lm.models.qwen3 import Model as Qwen3Model

        if isinstance(model, Qwen3Model):
            tool_parser = "qwen3_coder" if "coder" in lower_id else None
            return ChunkParserConfig(
                reasoning_parser_name="qwen3",
                tool_parser_name=tool_parser,
                enable_thinking=True,
            )
    except ImportError:
        pass

    # Qwen3 Next: hybrid detection with model ID refinement for Coder variants
    try:
        from mlx_lm.models.qwen3_next import Model as Qwen3NextModel

        if isinstance(model, Qwen3NextModel):
            tool_parser = "qwen3_coder" if "coder" in lower_id else None
            return ChunkParserConfig(
                reasoning_parser_name="qwen3",
                tool_parser_name=tool_parser,
                enable_thinking=True,
            )
    except ImportError:
        pass

    # GLM family: hybrid detection with Solar Open disambiguation
    # Note: Solar Open is aliased to GLM4-Moe in current MLX version
    try:
        from mlx_lm.models.glm4_moe import Model as Glm4MoeModel

        if isinstance(model, Glm4MoeModel):
            if "solar" in lower_id and "open" in lower_id:
                return ChunkParserConfig(
                    reasoning_parser_name="solar_open",
                    tool_parser_name=None,
                    enable_thinking=True,
                )
            return ChunkParserConfig(
                reasoning_parser_name="glm4_moe",
                tool_parser_name="glm47",
                enable_thinking=True,
            )
    except ImportError:
        pass

    try:
        from mlx_lm.models.glm4 import Model as Glm4Model

        if isinstance(model, Glm4Model):
            if "solar" in lower_id and "open" in lower_id:
                return ChunkParserConfig(
                    reasoning_parser_name="solar_open",
                    tool_parser_name=None,
                    enable_thinking=True,
                )
            return ChunkParserConfig(
                reasoning_parser_name="glm4_moe",
                tool_parser_name="glm47",
                enable_thinking=True,
            )
    except ImportError:
        pass

    try:
        from mlx_lm.models.glm import Model as GlmModel

        if isinstance(model, GlmModel):
            if "solar" in lower_id and "open" in lower_id:
                return ChunkParserConfig(
                    reasoning_parser_name="solar_open",
                    tool_parser_name=None,
                    enable_thinking=True,
                )
            return ChunkParserConfig(
                reasoning_parser_name="glm4_moe",
                tool_parser_name="glm47",
                enable_thinking=True,
            )
    except ImportError:
        pass

    # Nemotron: hybrid detection with nano/3 refinement to avoid overreach
    try:
        from mlx_lm.models.nemotron import Model as NemotronModel

        if isinstance(model, NemotronModel):
            if "nano" in lower_id or "3" in lower_id:
                return ChunkParserConfig(
                    reasoning_parser_name="nemotron3_nano",
                    tool_parser_name=None,
                    enable_thinking=True,
                )
            return None
    except ImportError:
        pass

    # No confident class-based detection available
    return None
