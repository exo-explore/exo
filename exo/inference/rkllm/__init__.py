"""
RKLLM Inference Engine for Rockchip RK3588/RK3576 NPU devices.

This module provides inference capabilities using the RKLLM runtime
for Rockchip NPU devices. The recommended mode is HTTP, which connects
to a running rkllama server.

Usage:
  # Start rkllama server on RK3588 device:
  python server.py --target_platform rk3588 --port 8080

  # Then use exo with rkllm engine:
  exo --inference-engine rkllm

Environment variables:
  RKLLM_SERVER_HOST: Host of rkllama server (default: localhost)
  RKLLM_SERVER_PORT: Port of rkllama server (default: 8080)
"""

# Lazy imports to avoid circular dependency with exo.models
# The full import chain rkllm_engine -> shard_download -> exo.models
# would cause issues if exo.models imports from this package first.


def __getattr__(name):
  """Lazy import for module attributes to avoid circular imports."""
  if name == "RKLLMInferenceEngine":
    from exo.inference.rkllm.rkllm_engine import RKLLMInferenceEngine
    return RKLLMInferenceEngine
  elif name == "RKLLMHTTPClient":
    from exo.inference.rkllm.rkllm_http_client import RKLLMHTTPClient
    return RKLLMHTTPClient
  elif name == "RKLLMServerConfig":
    from exo.inference.rkllm.rkllm_http_client import RKLLMServerConfig
    return RKLLMServerConfig
  elif name == "detect_rockchip_npu":
    from exo.inference.rkllm.detection import detect_rockchip_npu
    return detect_rockchip_npu
  elif name == "get_rockchip_soc_name":
    from exo.inference.rkllm.detection import get_rockchip_soc_name
    return get_rockchip_soc_name
  elif name == "RKLLM_MODELS":
    from exo.inference.rkllm.models import RKLLM_MODELS
    return RKLLM_MODELS
  elif name == "RKLLM_PRETTY_NAMES":
    from exo.inference.rkllm.models import RKLLM_PRETTY_NAMES
    return RKLLM_PRETTY_NAMES
  elif name == "is_streaming_model":
    from exo.inference.rkllm.models import is_streaming_model
    return is_streaming_model
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core engine
  "RKLLMInferenceEngine",
  "RKLLMHTTPClient",
  "RKLLMServerConfig",
  # Detection
  "detect_rockchip_npu",
  "get_rockchip_soc_name",
  # Models
  "RKLLM_MODELS",
  "RKLLM_PRETTY_NAMES",
  "is_streaming_model",
]
