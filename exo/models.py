from exo.inference.shard import Shard
from typing import Optional

model_cards = {
  ### llama
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
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
    "llava-1.5-7b-hf": {
        "layers": 32,
        "repo": {
            "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf",
        },
    },
    ### QWEN
    "qwen-2.5-coder-0.5b-4bit": {
        "layers": 24,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-0.5b-8bit": {
        "layers": 24,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-0.5b-bf16": {
        "layers": 24,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
        },
    },
    "qwen-2.5-coder-1.5b-4bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-1.5b-8bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-1.5b-bf16": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
        },
    },
    "qwen-2.5-coder-3b-4bit": {
        "layers": 36,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-3b-8bit": {
        "layers": 36,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-3b-bf16": {
        "layers": 36,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-bf16",
        },
    },
    "qwen-2.5-coder-7b-4bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-7b-8bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-7b-bf16": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-bf16",
        },
    },
    "qwen-2.5-coder-14b-4bit": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-14b-8bit": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-14b-bf16": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-bf16",
        },
    },
    "qwen-2.5-coder-32b-4bit": {
        "layers": 64,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        },
    },
    "qwen-2.5-coder-32b-8bit": {
        "layers": 64,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit",
        },
    },
    "qwen-2.5-coder-32b-bf16": {
        "layers": 64,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-bf16",
        },
    },
    "qwen-2.5-7b-4bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        },
    },
    "qwen-2.5-7b-8bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        },
    },
    "qwen-2.5-math-7b-4bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
        },
    },
    "qwen-2.5-math-7b-8bit": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-8bit",
        },
    },
    "qwen-2.5-math-7b-bf16": {
        "layers": 28,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-bf16",
        },
    },
    "qwen-2.5-14b-4bit": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        },
    },
    "qwen-2.5-14b-8bit": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-8bit",
        },
    },
    "qwen-2.5-14b-bf16": {
        "layers": 48,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-bf16",
        },
    },
    "qwen-2.5-72b-4bit": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit",
        },
    },
    "qwen-2.5-72b-8bit": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-8bit",
        },
    },
    "qwen-2.5-72b-bf16": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-bf16",
        },
    },
    "qwen-2.5-math-72b-4bit": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit",
        },
    },
    "qwen-2.5-math-72b-8bit": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-8bit",
        },
    },
    "qwen-2.5-math-72b-bf16": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-bf16",
        },
    },

    ### NEMOTRON
    "nemotron-70b": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit",
        },
    },
    "nemotron-70b-bf16": {
        "layers": 80,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16",
        },
    },

    ### GEMMA 2
    "gemma2-2b": {
        "layers": 26,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-2b-it",
        },
    },
    "gemma2-2b-4bit": {
        "layers": 26,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-2b-it-4bit",
        },
    },
    "gemma2-2b-8bit": {
        "layers": 26,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-2b-it-8bit",
        },
    },
    "gemma2-2b-fp16": {
        "layers": 26,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-2b-it-fp16",
        },
    },
    "gemma2-9b-4bit": {
        "layers": 42,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit",
        },
    },
    "gemma2-9b-8bit": {
        "layers": 42,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-8bit",
        },
    },
    "gemma2-9b-fp16": {
        "layers": 42,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-fp16",
        },
    },
    "gemma2-27b-4bit": {
        "layers": 46,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit",
        },
    },
    "gemma2-27b-8bit": {
        "layers": 46,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-8bit",
        },
    },
    "gemma2-27b-bf16": {
        "layers": 46,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-bf16",
        },
    },
    "gemma2-27b-bf16-4bit": {
        "layers": 46,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-bf16-4bit",
        },
    },
    "gemma2-27b-bf16-8bit": {
        "layers": 46,
        "repo": {
            "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-bf16-8bit",
        },
    },

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
  "nemotron-70b": "Nemotron 70B",
  "nemotron-70b-bf16": "Nemotron 70B (BF16)",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
"qwen-2.5-coder-0.5b-4bit": "Qwen 2.5 Coder 0.5B (4-bit)",
  "qwen-2.5-coder-0.5b-8bit": "Qwen 2.5 Coder 0.5B (8-bit)",
  "qwen-2.5-coder-0.5b-bf16": "Qwen 2.5 Coder 0.5B (BF16)",
  "qwen-2.5-coder-1.5b-4bit": "Qwen 2.5 Coder 1.5B (4-bit)",
  "qwen-2.5-coder-1.5b-8bit": "Qwen 2.5 Coder 1.5B (8-bit)",
  "qwen-2.5-coder-1.5b-bf16": "Qwen 2.5 Coder 1.5B (BF16)",
  "qwen-2.5-coder-3b-4bit": "Qwen 2.5 Coder 3B (4-bit)",
  "qwen-2.5-coder-3b-8bit": "Qwen 2.5 Coder 3B (8-bit)",
  "qwen-2.5-coder-3b-bf16": "Qwen 2.5 Coder 3B (BF16)",
  "qwen-2.5-coder-7b-4bit": "Qwen 2.5 Coder 7B (4-bit)",
  "qwen-2.5-coder-7b-8bit": "Qwen 2.5 Coder 7B (8-bit)",
  "qwen-2.5-coder-7b-bf16": "Qwen 2.5 Coder 7B (BF16)",
  "qwen-2.5-coder-14b-4bit": "Qwen 2.5 Coder 14B (4-bit)",
  "qwen-2.5-coder-14b-8bit": "Qwen 2.5 Coder 14B (8-bit)",
  "qwen-2.5-coder-14b-bf16": "Qwen 2.5 Coder 14B (BF16)",
  "qwen-2.5-coder-32b-4bit": "Qwen 2.5 Coder 32B (4-bit)",
  "qwen-2.5-coder-32b-8bit": "Qwen 2.5 Coder 32B (8-bit)",
  "qwen-2.5-coder-32b-bf16": "Qwen 2.5 Coder 32B (BF16)",
  "qwen-2.5-7b-4bit": "Qwen 2.5 7B (4-bit)",
  "qwen-2.5-7b-8bit": "Qwen 2.5 7B (8-bit)",
  "qwen-2.5-math-7b-4bit": "Qwen 2.5 7B (Math, 4-bit)",
  "qwen-2.5-math-7b-8bit": "Qwen 2.5 7B (Math, 8-bit)",
  "qwen-2.5-math-7b-bf16": "Qwen 2.5 7B (Math, BF16)",
  "qwen-2.5-14b-4bit": "Qwen 2.5 14B (4-bit)",
  "qwen-2.5-14b-8bit": "Qwen 2.5 14B (8-bit)",
  "qwen-2.5-14b-bf16": "Qwen 2.5 14B (BF16)",
  "qwen-2.5-72b-4bit": "Qwen 2.5 72B (4-bit)",
  "qwen-2.5-72b-8bit": "Qwen 2.5 72B (8-bit)",
  "qwen-2.5-72b-bf16": "Qwen 2.5 72B (BF16)",
  "qwen-2.5-math-72b-4bit": "Qwen 2.5 72B (Math, 4-bit)",
  "qwen-2.5-math-72b-8bit": "Qwen 2.5 72B (Math, 8-bit)",
  "qwen-2.5-math-72b-bf16": "Qwen 2.5 72B (Math, BF16)",
  "gemma2-2b": "Gemma2 2B",
  "gemma2-2b-4bit": "Gemma2 2B (4-bit)",
  "gemma2-2b-8bit": "Gemma2 2B (8-bit)",
  "gemma2-2b-fp16": "Gemma2 2B (FP16)",
  "gemma2-9b-4bit": "Gemma2 9B (4-bit)",
  "gemma2-9b-8bit": "Gemma2 9B (8-bit)",
  "gemma2-9b-fp16": "Gemma2 9B (FP16)",
  "gemma2-27b-4bit": "Gemma2 27B (4-bit)",
  "gemma2-27b-8bit": "Gemma2 27B (8-bit)",
  "gemma2-27b-bf16": "Gemma2 27B (BF16)",
  "gemma2-27b-bf16-4bit": "Gemma2 27B (BF16, 4-bit)",
  "gemma2-27b-bf16-8bit": "Gemma2 27B (BF16, 8-bit)",
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

