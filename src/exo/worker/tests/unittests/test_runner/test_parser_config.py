# pyright: reportAny=false

# pyright: reportPrivateUsage=false

from unittest.mock import Mock

from exo.worker.parsing.selection import select_chunk_parser_config, _select_by_model_id
from mlx_lm.models.gpt_oss import Model as GptOssModel


def test_get_parser_config_maps_known_model_ids() -> None:
    cfg = _select_by_model_id("some-gpt-oss-harmony-model")
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"

    cfg = _select_by_model_id("SOLAR-OPEN")
    assert cfg.reasoning_parser_name == "solar_open"
    assert cfg.tool_parser_name is None

    cfg = _select_by_model_id("qwen3-coder")
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    cfg = _select_by_model_id("nemotron-3-nano")
    assert cfg.reasoning_parser_name == "nemotron3_nano"
    assert cfg.tool_parser_name is None

    cfg = _select_by_model_id("some-unknown-model")
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_get_parser_config_for_loaded_model_prioritizes_model_class() -> None:
    """Test that model class-based selection takes precedence over model_id."""

    # Test GPT-OSS model class detection (primary use case from code review)
    gpt_oss_model = Mock(spec=GptOssModel)

    # Even with non-standard model_id, should use harmony parser for GPT-OSS class
    cfg = select_chunk_parser_config(
        model_id="some-weird-model-id", model=gpt_oss_model
    )
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"

    # Test with actual GPT-OSS model_id to ensure it still works
    cfg = select_chunk_parser_config(
        model_id="mlx-community/GPT-OSS-7B", model=gpt_oss_model
    )
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"


def test_get_parser_config_for_loaded_model_fallback_to_model_id() -> None:
    """Test that non-GPT-OSS models fall back to model_id mapping."""

    # Non-GPT-OSS model should use model_id mapping
    other_model = Mock()

    # Should fall back to qwen3 mapping
    cfg = select_chunk_parser_config(model_id="qwen3-coder", model=other_model)
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    # Should fall back to solar_open mapping
    cfg = select_chunk_parser_config(model_id="SOLAR-OPEN", model=other_model)
    assert cfg.reasoning_parser_name == "solar_open"
    assert cfg.tool_parser_name is None

    # Should fall back to default (no parsing) for unknown models
    cfg = select_chunk_parser_config(
        model_id="completely-unknown-model", model=other_model
    )
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None


def test_bucket_1_model_classes_are_detected_correctly() -> None:
    """Test that Bucket 1 model classes are detected via isinstance checks."""

    # Test MiniMax model class detection
    try:
        from mlx_lm.models.minimax import Model as MiniMaxModel

        minimax_model = Mock(spec=MiniMaxModel)
        cfg = select_chunk_parser_config(model_id="any-model-id", model=minimax_model)
        assert cfg.reasoning_parser_name == "minimax_m2"
        assert cfg.tool_parser_name == "minimax_m2"
    except ImportError:
        # MiniMaxModel not available, skip this test
        pass

    # Test Qwen3 VL model class detection
    try:
        from mlx_lm.models.qwen3_vl import Model as Qwen3VlModel

        qwen3_vl_model = Mock(spec=Qwen3VlModel)
        cfg = select_chunk_parser_config(model_id="any-model-id", model=qwen3_vl_model)
        assert cfg.reasoning_parser_name == "qwen3_vl"
        assert cfg.tool_parser_name is None
    except ImportError:
        # Qwen3VlModel not available, skip this test
        pass

    # Test Qwen3 MoE model class detection
    try:
        from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel

        qwen3_moe_model = Mock(spec=Qwen3MoeModel)
        cfg = select_chunk_parser_config(model_id="any-model-id", model=qwen3_moe_model)
        assert cfg.reasoning_parser_name == "qwen3_moe"
        assert cfg.tool_parser_name is None
    except ImportError:
        # Qwen3MoeModel not available, skip this test
        pass

    # Test GptOssMoeModel class detection
    try:
        from mlx_lm.models.gpt_oss import GptOssMoeModel

        gpt_oss_moe_model = Mock(spec=GptOssMoeModel)
        cfg = select_chunk_parser_config(
            model_id="any-model-id", model=gpt_oss_moe_model
        )
        assert cfg.reasoning_parser_name == "harmony"
        assert cfg.tool_parser_name == "harmony"
    except ImportError:
        # GptOssMoeModel not available, skip this test
        pass


def test_bucket_2_hybrid_detection_works_correctly() -> None:
    """Test that Bucket 2 hybrid detection rules work correctly."""

    # Test Qwen3 base vs Coder disambiguation
    try:
        from mlx_lm.models.qwen3 import Model as Qwen3Model

        qwen3_model = Mock(spec=Qwen3Model)

        # Qwen3 base model with coder model_id should use qwen3_coder tools
        cfg = select_chunk_parser_config(
            model_id="qwen3-coder-instruct", model=qwen3_model
        )
        assert cfg.reasoning_parser_name == "qwen3"
        assert cfg.tool_parser_name == "qwen3_coder"

        # Qwen3 base model with non-coder model_id should have no tools
        cfg = select_chunk_parser_config(model_id="qwen3-base-chat", model=qwen3_model)
        assert cfg.reasoning_parser_name == "qwen3"
        assert cfg.tool_parser_name is None
    except ImportError:
        # Qwen3Model not available, skip this test
        pass

    # Test GLM vs Solar Open disambiguation
    try:
        from mlx_lm.models.glm4_moe import Model as Glm4MoeModel

        glm_model = Mock(spec=Glm4MoeModel)

        # GLM model with Solar Open model_id should use solar_open parsing
        cfg = select_chunk_parser_config(model_id="solar-open-chat", model=glm_model)
        assert cfg.reasoning_parser_name == "solar_open"
        assert cfg.tool_parser_name is None

        # GLM model with regular GLM model_id should use GLM parsing
        cfg = select_chunk_parser_config(model_id="glm-4.7-chat", model=glm_model)
        assert cfg.reasoning_parser_name == "glm4_moe"
        assert cfg.tool_parser_name == "glm47"
    except ImportError:
        # Glm4MoeModel not available, skip this test
        pass

    # Test Nemotron refinement
    try:
        from mlx_lm.models.nemotron import Model as NemotronModel

        nemotron_model = Mock(spec=NemotronModel)

        # Nemotron model with nano in model_id should use nemotron3_nano parsing
        cfg = select_chunk_parser_config(
            model_id="nemotron-nano-chat", model=nemotron_model
        )
        assert cfg.reasoning_parser_name == "nemotron3_nano"
        assert cfg.tool_parser_name is None

        # Nemotron model with "3" in model_id should use nemotron3_nano parsing
        cfg = select_chunk_parser_config(
            model_id="nemotron-3-chat", model=nemotron_model
        )
        assert cfg.reasoning_parser_name == "nemotron3_nano"
        assert cfg.tool_parser_name is None

        # Nemotron model without nano/3 should fall back to model_id mapping
        cfg = select_chunk_parser_config(
            model_id="nemotron-regular", model=nemotron_model
        )
        # Should fallback to model_id mapping since this doesn't match our refinement rule
        assert (
            cfg.reasoning_parser_name is None
        )  # Unknown model falls back to no parsing
    except ImportError:
        # NemotronModel not available, skip this test
        pass
