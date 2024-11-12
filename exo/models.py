from exo.inference.shard import Shard

model_base_shards = {
  ### llama
  "llama-3.2-1b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Llama-3.2-1B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=16),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="unsloth/Llama-3.2-1B-Instruct", start_layer=0, end_layer=0, n_layers=16),
  },
  "llama-3.2-3b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Llama-3.2-3B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="unsloth/Llama-3.2-3B-Instruct", start_layer=0, end_layer=0, n_layers=28),
  },
  "llama-3.1-8b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=32),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", start_layer=0, end_layer=0, n_layers=32),
  },
  "llama-3.1-70b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="NousResearch/Meta-Llama-3.1-70B-Instruct", start_layer=0, end_layer=0, n_layers=80),
  },
  "llama-3.1-70b-bf16": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED", start_layer=0, end_layer=0, n_layers=80),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="NousResearch/Meta-Llama-3.1-70B-Instruct", start_layer=0, end_layer=0, n_layers=80),
  },
  "llama-3.1-405b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3.1-405B-4bit", start_layer=0, end_layer=0, n_layers=126),},
  "llama-3.1-405b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=126),},
  "llama-3-8b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=32),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R", start_layer=0, end_layer=0, n_layers=32),
  },
  "llama-3-70b": {
    "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3-70B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),
    "TinygradDynamicShardInferenceEngine": Shard(model_id="TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R", start_layer=0, end_layer=0, n_layers=80),
  },
  ### mistral
  "mistral-nemo": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Mistral-Nemo-Instruct-2407-4bit", start_layer=0, end_layer=0, n_layers=40),},
  "mistral-large": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Mistral-Large-Instruct-2407-4bit", start_layer=0, end_layer=0, n_layers=88),},
  ### deepseek
  "deepseek-coder-v2-lite": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", start_layer=0, end_layer=0, n_layers=27),},
  "deepseek-coder-v2.5": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", start_layer=0, end_layer=0, n_layers=60),},
  ### llava
  "llava-1.5-7b-hf": {"MLXDynamicShardInferenceEngine": Shard(model_id="llava-hf/llava-1.5-7b-hf", start_layer=0, end_layer=0, n_layers=32),},
  ### QWEN
  "qwen-2.5-coder-0.5b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit", start_layer=0,end_layer=0, n_layers=24), },
  "qwen-2.5-coder-0.5b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit", start_layer=0,end_layer=0, n_layers=24), },
  "qwen-2.5-coder-0.5b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16", start_layer=0,end_layer=0, n_layers=24), },
  "qwen-2.5-coder-1.5b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-1.5b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-1.5B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-1.5b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-3b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=36),},
  "qwen-2.5-coder-3b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-3B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=36),},
  "qwen-2.5-coder-3b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-3B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=36),},
  "qwen-2.5-coder-7b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-7b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-7b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-14b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-coder-14b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-coder-14b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-coder-32b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=64),},
  "qwen-2.5-coder-32b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=64),},
  "qwen-2.5-coder-32b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=64),},
  "qwen-2.5-7b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-7b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-7B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-math-7b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-math-7b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-7B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-math-7b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-7B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-14b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-14B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-14b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-14B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-14b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-14B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-72b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-72B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-72b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-72B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-72b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-72B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-math-72b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-72B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-math-72b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-72B-Instruct-8bit", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-math-72b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-72B-Instruct-bf16", start_layer=0, end_layer=0, n_layers=80),},
  ### nemotron
  "nemotron-70b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", start_layer=0, end_layer=0, n_layers=80),},
  "nemotron-70b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", start_layer=0, end_layer=0, n_layers=80),},
  # GEMMA 2
  "gemma2-2b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-2b-it", start_layer=0, end_layer=0, n_layers=26), },
  "gemma2-2b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-2b-it-4bit", start_layer=0, end_layer=0, n_layers=26), },
  "gemma2-2b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-2b-it-8bit", start_layer=0, end_layer=0, n_layers=26), },
  "gemma2-2b-fp16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-2b-it-fp16", start_layer=0, end_layer=0, n_layers=26), },
  "gemma2-9b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-9b-it-4bit", start_layer=0, end_layer=0, n_layers=42),},
  "gemma2-9b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-9b-it-8bit", start_layer=0, end_layer=0, n_layers=42), },
  "gemma2-9b-fp16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-9b-it-fp16", start_layer=0, end_layer=0, n_layers=42), },
  "gemma2-27b-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-4bit", start_layer=0, end_layer=0, n_layers=46), },
  "gemma2-27b-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-8bit", start_layer=0, end_layer=0, n_layers=46),},
  "gemma2-27b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-bf16", start_layer=0, end_layer=0, n_layers=46), },
  "gemma2-27b-bf16-4bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-bf16-4bit", start_layer=0, end_layer=0, n_layers=46), },
  "gemma2-27b-bf16-8bit": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-bf16-8bit", start_layer=0, end_layer=0, n_layers=46), },
  # dummy
  "dummy": {"DummyInferenceEngine": Shard(model_id="dummy", start_layer=0, end_layer=7, n_layers=8),},
}
