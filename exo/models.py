from exo.inference.shard import Shard
from typing import Optional, List

model_cards = {
  ### llama
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
      "TorchDynamicShardInferenceEngine": "meta-llama/Llama-3.2-1B-Instruct"
    },
  },
  "llama-3.2-3b": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.1-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    },
  },
  "llama-3.1-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3.1-70b-bf16": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  },
  "llama-3-8b": {
    "layers": 32,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R",
    },
  },
  "llama-3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R",
    },
  },
  "llama-3.1-405b": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-4bit", }, },
  "llama-3.1-405b-8bit": { "layers": 126, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", }, },
  ### mistral
  "mistral-nemo": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Nemo-Instruct-2407-4bit", }, },
  "mistral-large": { "layers": 88, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Large-Instruct-2407-4bit", }, },
  ### deepseek
  "deepseek-coder-v2-lite": { "layers": 27, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", }, },
  "deepseek-coder-v2.5": { "layers": 60, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", }, },
  ### llava
  "llava-1.5-7b-hf": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf", }, },
  ### qwen
  "qwen-2.5-0.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", }, },
  "qwen-2.5-coder-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", }, },
  "qwen-2.5-coder-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", }, },
  "qwen-2.5-coder-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", }, },
  "qwen-2.5-coder-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", }, },
  "qwen-2.5-coder-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", }, },
  "qwen-2.5-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit", }, },
  "qwen-2.5-math-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit", }, },
  "qwen-2.5-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit", }, },
  "qwen-2.5-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit", }, },
  "qwen-2.5-math-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit", }, },
  ### nemotron
  "nemotron-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", }, },
  "nemotron-70b-bf16": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", }, },
  # gemma
  "gemma2-9b": { "layers": 42, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit", }, },
  "gemma2-27b": { "layers": 46, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit", }, },
  # dummy
  "dummy": { "layers": 8, "repo": { "DummyInferenceEngine": "dummy", }, },
}

pretty_name = {
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.1-8b": "Llama 3.1 8B",
  "llama-3.1-70b": "Llama 3.1 70B",
  "llama-3.1-70b-bf16": "Llama 3.1 70B (BF16)",
  "llama-3.1-405b": "Llama 3.1 405B",
  "llama-3.1-405b-8bit": "Llama 3.1 405B (8-bit)",
  "gemma2-9b": "Gemma2 9B",
  "gemma2-27b": "Gemma2 27B",
  "nemotron-70b": "Nemotron 70B",
  "nemotron-70b-bf16": "Nemotron 70B (BF16)",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
}

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  return model_cards.get(model_id, {}).get("repo", {}).get(inference_engine_classname, None)

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  repo = get_repo(model_id, inference_engine_classname)
  n_layers = model_cards.get(model_id, {}).get("layers", 0)
  if repo is None or n_layers < 1:
    return None
  return Shard(model_id, 0, 0, n_layers)

def get_supported_models(supported_inference_engine_lists: List[List[str]]) -> List[str]:
  if not supported_inference_engine_lists:
    return list(model_cards.keys())

  from exo.inference.inference_engine import inference_engine_classes
  supported_inference_engine_lists = [
    [inference_engine_classes[engine] if engine in inference_engine_classes else engine for engine in engine_list]
    for engine_list in supported_inference_engine_lists
  ]

  def has_any_engine(model_info: dict, engine_list: List[str]) -> bool:
    return any(engine in model_info.get("repo", {}) for engine in engine_list)

  def supports_all_engine_lists(model_info: dict) -> bool:
    return all(has_any_engine(model_info, engine_list)
              for engine_list in supported_inference_engine_lists)

  return [
    model_id for model_id, model_info in model_cards.items()
    if supports_all_engine_lists(model_info)
  ]
