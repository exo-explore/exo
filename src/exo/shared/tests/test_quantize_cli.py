import os
import tempfile

import pytest

from exo.cli.commands.quantize import execute_quantize_command
from exo.shared.types.quantize_cli import (
    LayerSensitivityRecord,
    QuantizationConfig,
    QuantizationReport,
)
from exo.worker.engines.quantization.quantize_compiler import (
    ExoModelCardGenerator,
    ModelSensitivityAnalyzer,
    QuantizeCompiler,
    SmoothQuantCalibrator,
)


def test_quantize_config_and_report_models():
    """Verify data containers and report serialization."""
    rec = LayerSensitivityRecord(
        layer_name="layer_0",
        sensitivity_score=0.9,
        is_protected=True,
        assigned_bits=16.0,
    )
    assert rec.is_protected is True

    cfg = QuantizationConfig(
        model_id="test/model-1b",
        bits=1.58,
        output_dir="./out",
    )
    assert cfg.bits == 1.58

    report = QuantizationReport(
        model_id="test/model-1b",
        target_bits=1.58,
        original_size_mb=2000.0,
        quantized_size_mb=350.0,
        compression_ratio=5.71,
        layers_quantized=24,
        layers_protected=4,
        model_card_path="./out/model_card.toml",
        duration_sec=1.23,
    )
    t_dict = report.to_dict()
    assert t_dict["compression_ratio"] == 5.71
    assert t_dict["layers_quantized"] == 24


def test_model_sensitivity_analyzer():
    """Verify critical layer detection and precision protection logic."""
    analyzer = ModelSensitivityAnalyzer()

    layer_names = [
        "model.embed_tokens",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.norm",
        "lm_head",
    ]

    records = analyzer.analyze_layers(
        layer_names=layer_names,
        protect_attention_out=True,
        protect_gate_proj=True,
        target_bits=1.58,
    )
    rec_dict = {r.layer_name: r for r in records}

    # embed_tokens, norm, and lm_head must be protected with 16-bit
    assert rec_dict["model.embed_tokens"].is_protected is True
    assert rec_dict["model.embed_tokens"].assigned_bits == 16.0
    assert rec_dict["model.norm"].is_protected is True
    assert rec_dict["lm_head"].is_protected is True

    # o_proj and gate_proj should be protected when flags are True
    assert rec_dict["model.layers.0.self_attn.o_proj"].is_protected is True
    assert rec_dict["model.layers.0.mlp.gate_proj"].is_protected is True

    # q_proj and up_proj should be quantized to 1.58-bit
    assert rec_dict["model.layers.0.self_attn.q_proj"].is_protected is False
    assert rec_dict["model.layers.0.self_attn.q_proj"].assigned_bits == 1.58
    assert rec_dict["model.layers.0.mlp.up_proj"].is_protected is False


def test_smoothquant_calibrator():
    """Verify migration factors calculated from activation and weight distributions."""
    calibrator = SmoothQuantCalibrator(alpha=0.5)

    act_peaks = {"layer_a": 4.0, "layer_b": 9.0}
    weight_peaks = {"layer_a": 1.0, "layer_b": 1.0}

    # s = 4.0^0.5 / 1.0^0.5 = 2.0 / 1.0 = 2.0
    # s = 9.0^0.5 / 1.0^0.5 = 3.0 / 1.0 = 3.0
    scales = calibrator.compute_scaling_factors(act_peaks, weight_peaks)
    assert scales["layer_a"] == 2.0
    assert scales["layer_b"] == 3.0


def test_exo_model_card_generator():
    """Verify generation of valid Exo .toml model cards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ExoModelCardGenerator()
        config = QuantizationConfig(
            model_id="meta-llama/Llama-3.2-1B",
            bits=1.58,
            output_dir=tmpdir,
        )

        card_path = generator.generate_and_save(
            config=config,
            report_summary={"compression_ratio": 4.8, "original_size_mb": 2400.0, "quantized_size_mb": 500.0},
        )

        assert os.path.exists(card_path)
        with open(card_path, encoding="utf-8") as f:
            content = f.read()

        assert "name = \"meta-llama--Llama-3.2-1B-1.58bit\"" in content
        assert "target_bits = 1.58" in content
        assert "compression_ratio = 4.80" in content


def test_quantize_compiler_full_pipeline():
    """Verify end-to-end compilation with different bit targets."""
    compiler = QuantizeCompiler()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1.0-bit BNN compilation (with default protected layers)
        config_bnn = QuantizationConfig(
            model_id="test/llama-small",
            bits=1,
            output_dir=tmpdir,
        )
        report_bnn = compiler.compile(config_bnn)
        assert report_bnn.target_bits == 1.0
        assert report_bnn.compression_ratio > 1.5
        assert report_bnn.layers_quantized > 0

        # Test aggressive 1.0-bit BNN compilation without attention/gate protection
        config_bnn_aggressive = QuantizationConfig(
            model_id="test/llama-small",
            bits=1,
            output_dir=tmpdir,
            protect_attention_out=False,
            protect_gate_proj=False,
        )
        report_aggressive = compiler.compile(config_bnn_aggressive)
        assert report_aggressive.compression_ratio > 2.0

        # Test 1.58-bit compilation
        config_158 = QuantizationConfig(
            model_id="test/llama-small",
            bits=1.58,
            output_dir=tmpdir,
        )
        report_158 = compiler.compile(config_158)
        assert report_158.target_bits == 1.58
        assert report_158.compression_ratio > 1.3

        # Test 2-bit compilation
        config_2bit = QuantizationConfig(
            model_id="test/llama-small",
            bits=2,
            output_dir=tmpdir,
        )
        report_2bit = compiler.compile(config_2bit)
        assert report_2bit.target_bits == 2.0


def test_execute_quantize_command_cli():
    """Verify CLI command invocation and argument validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = execute_quantize_command(
            model_id="meta-llama/Llama-3.2-1B",
            bits=1.58,
            output_dir=tmpdir,
            verbose=False,
        )
        assert report.model_id == "meta-llama/Llama-3.2-1B"
        assert report.target_bits == 1.58

        # Invalid bits should raise ValueError
        with pytest.raises(ValueError, match="Unsupported quantization bits"):
            execute_quantize_command(
                model_id="test",
                bits=4.0,  # Unsupported
                output_dir=tmpdir,
                verbose=False,
            )
