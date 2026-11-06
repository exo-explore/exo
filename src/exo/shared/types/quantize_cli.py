from typing import Any, Literal

from pydantic import ConfigDict, Field

from exo.utils.pydantic_ext import FrozenModel

QuantBits = Literal[1, 1.58, 2]
DEFAULT_CALIBRATION_SAMPLES: int = 128
DEFAULT_SMOOTH_QUANT_ALPHA: float = 0.5


class LayerSensitivityRecord(FrozenModel):
    """Captures quantization sensitivity and protection status for an individual model layer."""

    model_config = ConfigDict(frozen=True, strict=True)

    layer_name: str
    sensitivity_score: float
    is_protected: bool
    assigned_bits: float


class QuantizationConfig(FrozenModel):
    """User-specified and derived configuration for the exo quantize pipeline."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str
    bits: QuantBits = 1.58
    output_dir: str
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES
    protect_attention_out: bool = True
    protect_gate_proj: bool = True
    smooth_quant_alpha: float = DEFAULT_SMOOTH_QUANT_ALPHA


class QuantizationReport(FrozenModel):
    """Detailed completion summary of a model quantization compilation run."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str
    target_bits: float
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    layers_quantized: int
    layers_protected: int
    model_card_path: str
    duration_sec: float
    layer_records: tuple[LayerSensitivityRecord, ...] = Field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_bits": self.target_bits,
            "original_size_mb": round(self.original_size_mb, 2),
            "quantized_size_mb": round(self.quantized_size_mb, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "layers_quantized": self.layers_quantized,
            "layers_protected": self.layers_protected,
            "model_card_path": self.model_card_path,
            "duration_sec": round(self.duration_sec, 3),
        }
