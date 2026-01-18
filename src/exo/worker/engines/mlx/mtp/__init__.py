"""Multi-Token Prediction (MTP) module for DeepSeek V3 speculative decoding."""

from exo.worker.engines.mlx.mtp.module import MTPModule
from exo.worker.engines.mlx.mtp.speculative_decode import mtp_speculative_generate

__all__ = ["MTPModule", "mtp_speculative_generate"]
