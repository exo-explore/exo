"""
RKLLM model definitions for Rockchip RK3588/RK3576 NPU.

This module contains all RKLLM-specific model configurations.
Models require pre-converted .rkllm files in ~/RKLLAMA/models/.
The HuggingFace repos specified are used for tokenizer loading only.

See: https://github.com/airockchip/rknn-llm for model conversion toolkit.
"""

# RKLLM model cards - maps model IDs to layer counts and tokenizer repos
RKLLM_MODELS = {
  "qwen2.5-1.5b-rkllm": {
    "layers": 28,
    "repo": {"RKLLMInferenceEngine": "Qwen/Qwen2.5-1.5B-Instruct",},
  },
  "qwen2.5-1.5b-instruct-rkllm": {
    "layers": 28,
    "repo": {"RKLLMInferenceEngine": "Qwen/Qwen2.5-1.5B-Instruct",},
  },
  "qwen2.5-3b-rkllm": {
    "layers": 36,
    "repo": {"RKLLMInferenceEngine": "Qwen/Qwen2.5-3B-Instruct",},
  },
  "qwen2.5-7b-rkllm": {
    "layers": 28,
    "repo": {"RKLLMInferenceEngine": "Qwen/Qwen2.5-7B-Instruct",},
  },
  "deepseek-r1-1.5b-rkllm": {
    "layers": 28,
    "repo": {"RKLLMInferenceEngine": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",},
  },
  "phi-3-mini-rkllm": {
    "layers": 32,
    "repo": {"RKLLMInferenceEngine": "microsoft/Phi-3-mini-4k-instruct",},
  },
}

# Human-readable names for RKLLM models
RKLLM_PRETTY_NAMES = {
  "qwen2.5-1.5b-rkllm": "Qwen 2.5 1.5B (RKLLM)",
  "qwen2.5-1.5b-instruct-rkllm": "Qwen 2.5 1.5B Instruct (RKLLM)",
  "qwen2.5-3b-rkllm": "Qwen 2.5 3B (RKLLM)",
  "qwen2.5-7b-rkllm": "Qwen 2.5 7B (RKLLM)",
  "deepseek-r1-1.5b-rkllm": "DeepSeek R1 1.5B (RKLLM)",
  "phi-3-mini-rkllm": "Phi-3 Mini (RKLLM)",
}

# Models that support streaming (long-form generation)
RKLLM_STREAMING_MODELS = {
  "deepseek-r1-1.5b-rkllm",
}


def get_rkllm_models() -> dict:
  """Return all RKLLM model definitions."""
  return RKLLM_MODELS.copy()


def get_rkllm_pretty_names() -> dict:
  """Return human-readable names for RKLLM models."""
  return RKLLM_PRETTY_NAMES.copy()


def is_streaming_model(model_id: str) -> bool:
  """Check if a model should use streaming generation."""
  return model_id in RKLLM_STREAMING_MODELS
