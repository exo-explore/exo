import sys
from typing import Any

from exo.shared.types.quantize_cli import (
    DEFAULT_CALIBRATION_SAMPLES,
    DEFAULT_SMOOTH_QUANT_ALPHA,
    QuantBits,
    QuantizationConfig,
    QuantizationReport,
)
from exo.worker.engines.quantization.quantize_compiler import QuantizeCompiler


def execute_quantize_command(
    model_id: str,
    bits: float = 1.58,
    output_dir: str = "./quantized_models",
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    protect_attention_out: bool = True,
    protect_gate_proj: bool = True,
    smooth_quant_alpha: float = DEFAULT_SMOOTH_QUANT_ALPHA,
    verbose: bool = True,
) -> QuantizationReport:
    """Executes the 'exo quantize' CLI command pipeline."""
    # Validate target bits
    if bits == 1.0:
        target_bits: QuantBits = 1
    elif bits == 1.58:
        target_bits = 1.58
    elif bits == 2.0:
        target_bits = 2
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}. Must be one of [1, 1.58, 2].")

    config = QuantizationConfig(
        model_id=model_id,
        bits=target_bits,
        output_dir=output_dir,
        calibration_samples=calibration_samples,
        protect_attention_out=protect_attention_out,
        protect_gate_proj=protect_gate_proj,
        smooth_quant_alpha=smooth_quant_alpha,
    )

    compiler = QuantizeCompiler()
    report = compiler.compile(config)

    if verbose:
        print("\n========================================================")
        print(" [EXO QUANTIZE] Local Model Quantization Complete!")
        print("========================================================")
        print(f" Model ID          : {report.model_id}")
        print(f" Target Precision  : {report.target_bits}-bit")
        print(f" Original Size     : {report.original_size_mb:.2f} MB")
        print(f" Quantized Size    : {report.quantized_size_mb:.2f} MB")
        print(f" Compression Ratio : {report.compression_ratio:.2f}x")
        print(f" Layers Quantized  : {report.layers_quantized}")
        print(f" Layers Protected  : {report.layers_protected}")
        print(f" Model Card Path   : {report.model_card_path}")
        print(f" Elapsed Time      : {report.duration_sec:.3f} s")
        print("========================================================\n")

    return report


def main(argv: list[str] | None = None) -> Any:
    """CLI entrypoint when called as a subcommand."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="exo quantize",
        description="One-click local model quantization compiler for Exo.",
    )
    parser.add_argument("model_id", type=str, help="Hugging Face model ID or local directory")
    parser.add_argument(
        "--bits",
        type=float,
        default=1.58,
        choices=[1.0, 1.58, 2.0],
        help="Target quantization precision bits (1, 1.58, or 2, default: 1.58)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_models",
        dest="output_dir",
        help="Output directory to save quantized weights and model card",
    )
    parser.add_argument(
        "--no-protect-attn",
        action="store_false",
        dest="protect_attention_out",
        help="Disable precision protection on attention output projections",
    )
    parser.add_argument(
        "--no-protect-gate",
        action="store_false",
        dest="protect_gate_proj",
        help="Disable precision protection on MLP gate projections",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_SMOOTH_QUANT_ALPHA,
        dest="smooth_quant_alpha",
        help="SmoothQuant alpha hyperparameter (default: 0.5)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return execute_quantize_command(
        model_id=args.model_id,
        bits=args.bits,
        output_dir=args.output_dir,
        protect_attention_out=args.protect_attention_out,
        protect_gate_proj=args.protect_gate_proj,
        smooth_quant_alpha=args.smooth_quant_alpha,
    )
