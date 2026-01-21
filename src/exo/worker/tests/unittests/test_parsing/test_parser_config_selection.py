# pyright: reportAny=false

# pyright: reportPrivateUsage=false

from unittest.mock import Mock

from exo.worker.parsing.selection import select_chunk_parser_config


def test_select_chunk_parser_config_by_model_id() -> None:
    """Test that model ID-based selection works correctly."""

    # Test GPT-OSS/Harmony mapping
    cfg = select_chunk_parser_config(model_id="mlx-community/GPT-OSS-7B", model=None)
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"

    # Test Solar Open mapping
    cfg = select_chunk_parser_config(model_id="solar-open-chat", model=None)
    assert cfg.reasoning_parser_name == "solar_open"
    assert cfg.tool_parser_name is None

    # Test MiniMax mapping
    cfg = select_chunk_parser_config(model_id="minimax-m2-chat", model=None)
    assert cfg.reasoning_parser_name == "minimax_m2"
    assert cfg.tool_parser_name == "minimax_m2"

    # Test GLM mapping
    cfg = select_chunk_parser_config(model_id="glm-4.7-chat", model=None)
    assert cfg.reasoning_parser_name == "glm4_moe"
    assert cfg.tool_parser_name == "glm47"

    # Test Qwen3 VL mapping
    cfg = select_chunk_parser_config(model_id="qwen3-vl-instruct", model=None)
    assert cfg.reasoning_parser_name == "qwen3_vl"
    assert cfg.tool_parser_name is None

    # Test Qwen3 MoE mapping
    cfg = select_chunk_parser_config(model_id="qwen3-moe-chat", model=None)
    assert cfg.reasoning_parser_name == "qwen3_moe"
    assert cfg.tool_parser_name is None

    # Test Qwen3 Coder mapping
    cfg = select_chunk_parser_config(model_id="qwen3-coder-instruct", model=None)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    # Test Qwen3 base mapping
    cfg = select_chunk_parser_config(model_id="qwen3-base-chat", model=None)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name is None

    # Test Nemotron mapping
    cfg = select_chunk_parser_config(model_id="nemotron-3-nano-chat", model=None)
    assert cfg.reasoning_parser_name == "nemotron3_nano"
    assert cfg.tool_parser_name is None

    # Test FunctionGemma mapping
    cfg = select_chunk_parser_config(model_id="functiongemma-2b", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name == "function_gemma"

    # Test Hermes mapping
    cfg = select_chunk_parser_config(model_id="hermes-3-chat", model=None)
    assert cfg.reasoning_parser_name == "hermes"
    assert cfg.tool_parser_name == "json_tools"

    # Test unknown model (should default to no parsing)
    cfg = select_chunk_parser_config(model_id="completely-unknown-model", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_select_chunk_parser_config_falls_back_to_model_id() -> None:
    """Test that when model is not recognized by class, it falls back to model ID."""

    # Use a mock model that isn't any known MLX class
    unknown_model = Mock()
    unknown_model.__class__.__name__ = "UnknownModel"

    # Should fall back to model ID mapping
    cfg = select_chunk_parser_config(
        model_id="qwen3-coder-instruct", model=unknown_model
    )
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    # Should fall back to unknown model default
    cfg = select_chunk_parser_config(model_id="unknown-model", model=unknown_model)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_select_chunk_parser_config_with_model_class() -> None:
    """Test that model class-based selection takes precedence when available."""

    # Test GPT-OSS model class detection
    try:
        from mlx_lm.models.gpt_oss import Model as GptOssModel

        gpt_oss_model = Mock(spec=GptOssModel)

        # Should use class-based detection regardless of model_id
        cfg = select_chunk_parser_config(
            model_id="weird-non-standard-id", model=gpt_oss_model
        )
        assert cfg.reasoning_parser_name == "harmony"
        assert cfg.tool_parser_name == "harmony"
    except ImportError:
        # GptOssModel not available in this MLX version
        pass

    # Test MiniMax model class detection
    try:
        from mlx_lm.models.minimax import Model as MiniMaxModel

        minimax_model = Mock(spec=MiniMaxModel)

        # Should use class-based detection regardless of model_id
        cfg = select_chunk_parser_config(model_id="any-model-id", model=minimax_model)
        assert cfg.reasoning_parser_name == "minimax_m2"
        assert cfg.tool_parser_name == "minimax_m2"
    except ImportError:
        # MiniMaxModel not available in this MLX version
        pass


def test_select_chunk_parser_config_with_none_model() -> None:
    """Test that function works correctly when model is None."""

    # Should fall back to model ID mapping
    cfg = select_chunk_parser_config(model_id="qwen3-coder-instruct", model=None)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    cfg = select_chunk_parser_config(model_id="glm-4.7-chat", model=None)
    assert cfg.reasoning_parser_name == "glm4_moe"
    assert cfg.tool_parser_name == "glm47"


def test_select_chunk_parser_config_case_insensitive() -> None:
    """Test that model ID matching is case-insensitive."""

    # Test various case combinations
    configs = [
        select_chunk_parser_config(model_id="HARMONY-GPT-OSS", model=None),
        select_chunk_parser_config(model_id="harmony-gpt-oss", model=None),
        select_chunk_parser_config(model_id="HARMONY", model=None),
        select_chunk_parser_config(model_id="harmony", model=None),
    ]

    for cfg in configs:
        assert cfg.reasoning_parser_name == "harmony"
        assert cfg.tool_parser_name == "harmony"


def test_select_chunk_parser_config_gpt_oss_variants() -> None:
    """Test that both 'harmony' and 'gpt-oss' in model ID work."""

    # Test 'harmony' in model ID
    cfg = select_chunk_parser_config(model_id="some-harmony-model", model=None)
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"

    # Test 'gpt-oss' in model ID
    cfg = select_chunk_parser_config(model_id="some-gpt-oss-model", model=None)
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"


def test_select_chunk_parser_config_solar_open_requirements() -> None:
    """Test that Solar Open requires both 'solar' AND 'open' in model ID."""

    # Should match when both are present
    cfg = select_chunk_parser_config(model_id="solar-open-chat", model=None)
    assert cfg.reasoning_parser_name == "solar_open"
    assert cfg.tool_parser_name is None

    # Should NOT match when only one is present
    cfg = select_chunk_parser_config(model_id="solar-chat", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None

    cfg = select_chunk_parser_config(model_id="open-chat", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_select_chunk_parser_config_nemotron_requirements() -> None:
    """Test that Nemotron requires 'nemotron' AND ('nano' OR '3') in model ID."""

    # Should match when all conditions are met
    cfg = select_chunk_parser_config(model_id="nemotron-nano-chat", model=None)
    assert cfg.reasoning_parser_name == "nemotron3_nano"
    assert cfg.tool_parser_name is None

    cfg = select_chunk_parser_config(model_id="nemotron-3-chat", model=None)
    assert cfg.reasoning_parser_name == "nemotron3_nano"
    assert cfg.tool_parser_name is None

    # Should NOT match when 'nemotron' is present but neither 'nano' nor '3'
    cfg = select_chunk_parser_config(model_id="nemotron-regular-chat", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None

    # Should NOT match when 'nano' or '3' are present but 'nemotron' is not
    cfg = select_chunk_parser_config(model_id="nano-chat", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_select_chunk_parser_config_functiongemma_variants() -> None:
    """Test that FunctionGemma matches both 'functiongemma' and 'gemma' + 'function'."""

    # Test 'functiongemma' direct match
    cfg = select_chunk_parser_config(model_id="functiongemma-2b", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name == "function_gemma"

    # Test 'gemma' + 'function' combination
    cfg = select_chunk_parser_config(model_id="gemma-function-chat", model=None)
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name == "function_gemma"


def test_select_chunk_parser_config_qwen3_precedence() -> None:
    """Test that Qwen3 model ID matching follows proper precedence order."""

    # VL should take precedence over base
    cfg = select_chunk_parser_config(model_id="qwen3-vl-instruct", model=None)
    assert cfg.reasoning_parser_name == "qwen3_vl"
    assert cfg.tool_parser_name is None

    # MoE should take precedence over base
    cfg = select_chunk_parser_config(model_id="qwen3-moe-chat", model=None)
    assert cfg.reasoning_parser_name == "qwen3_moe"
    assert cfg.tool_parser_name is None

    # Coder should take precedence over base (has tools)
    cfg = select_chunk_parser_config(model_id="qwen3-coder-instruct", model=None)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    # Base case
    cfg = select_chunk_parser_config(model_id="qwen3-base-chat", model=None)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name is None


def test_select_chunk_parser_config_hermes_variants() -> None:
    """Test that Hermes matches both 'hermes' and 'tool-use'."""

    # Test 'hermes' match
    cfg = select_chunk_parser_config(model_id="hermes-3-chat", model=None)
    assert cfg.reasoning_parser_name == "hermes"
    assert cfg.tool_parser_name == "json_tools"

    # Test 'tool-use' match
    cfg = select_chunk_parser_config(model_id="tool-use-chat", model=None)
    assert cfg.reasoning_parser_name == "hermes"
    assert cfg.tool_parser_name == "json_tools"
