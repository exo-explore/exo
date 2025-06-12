from exo.inference.shard import Shard
from typing import Optional, List

model_cards = {
  ### llama
  "llama-3.3-70b": {
    "layers": 80,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.3-70B-Instruct",
       "LlamaCppInferenceEngine": "unsloth/Llama-3.3-70B-Instruct-GGUF",
    },
  },
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
      "LlamaCppInferenceEngine": "unsloth/Llama-3.2-1B-Instruct-GGUF",
    },
  },
  
  # Llama-3.2-1B GGUF quantization variants
  "llama-3.2-1b-q4-k-m": {
    "layers": 16,
    "repo": {
      "LlamaCppInferenceEngine": "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
  },
  "llama-3.2-1b-q5-k-m": {
    "layers": 16,
    "repo": {
      "LlamaCppInferenceEngine": "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
    },
  },
  "llama-3.2-1b-q8-0": {
    "layers": 16,
    "repo": {
      "LlamaCppInferenceEngine": "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf",
    },
  },
  "llama-3.2-1b-f16": {
    "layers": 16,
    "repo": {
      "LlamaCppInferenceEngine": "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-F16.gguf",
    },
  },
  
  "llama-3.2-3b": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
       "LlamaCppInferenceEngine": "unsloth/Llama-3.2-3B-Instruct-GGUF",
    },
  },
  "llama-3.2-3b-8bit": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-8bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  },
  "llama-3.2-3b-bf16": {
    "layers": 28,
    "repo": {
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct",
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
  "deepseek-v3": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-4bit", }, },
  "deepseek-v3-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-3bit", }, },
  "deepseek-r1": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-4bit", }, },
  "deepseek-r1-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit", }, },
  ### deepseek distills
  "deepseek-r1-distill-qwen-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/deepseek-r1-distill-qwen-1.5b", }, },
  "deepseek-r1-distill-qwen-1.5b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit", }, },
  "deepseek-r1-distill-qwen-1.5b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-6bit", }, },
  "deepseek-r1-distill-qwen-1.5b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit", }, },
  "deepseek-r1-distill-qwen-1.5b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16", }, },
  "deepseek-r1-distill-qwen-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", }, },
  "deepseek-r1-distill-qwen-7b-3bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit", }, },
  "deepseek-r1-distill-qwen-7b-6bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-6bit", }, },
  "deepseek-r1-distill-qwen-7b-8bit": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit", }, },
  "deepseek-r1-distill-qwen-7b-bf16": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16", }, },
  "deepseek-r1-distill-qwen-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit", }, },
  "deepseek-r1-distill-qwen-14b-3bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit", }, },
  "deepseek-r1-distill-qwen-14b-6bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-6bit", }, },
  "deepseek-r1-distill-qwen-14b-8bit": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit", }, },
  "deepseek-r1-distill-qwen-14b-bf16": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16", }, },
  "deepseek-r1-distill-qwen-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit", }, },
  "deepseek-r1-distill-qwen-32b-3bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-3bit", }, },
  "deepseek-r1-distill-qwen-32b-6bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-6bit", }, },
  "deepseek-r1-distill-qwen-32b-8bit": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit", }, },
  "deepseek-r1-distill-qwen-32b-bf16": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16", }, },
  "deepseek-r1-distill-llama-8b": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit", }, },
  "deepseek-r1-distill-llama-8b-3bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-3bit", }, },
  "deepseek-r1-distill-llama-8b-6bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-6bit", }, },
  "deepseek-r1-distill-llama-8b-8bit": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit", }, },
  "deepseek-r1-distill-llama-8b-bf16": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16", }, },
  "deepseek-r1-distill-llama-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", }, },
  "deepseek-r1-distill-llama-70b-3bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-3bit", }, },
  "deepseek-r1-distill-llama-70b-6bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-6bit", }, },
  "deepseek-r1-distill-llama-70b-8bit": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit", }, },
  ### llava
  "llava-1.5-7b-hf": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf", }, },
  ### qwen
  "qwen-2.5-0.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", }, },
  "qwen-2.5-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", }, },
  "qwen-2.5-coder-1.5b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", }, },
  "qwen-2.5-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-3B-Instruct-4bit", }, },
  "qwen-2.5-coder-3b": { "layers": 36, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", }, },
  "qwen-2.5-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit", }, },
  "qwen-2.5-coder-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", }, },
  "qwen-2.5-math-7b": { "layers": 28, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit", }, },
  "qwen-2.5-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit", }, },
  "qwen-2.5-coder-14b": { "layers": 48, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", }, },
  "qwen-2.5-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-32B-Instruct-4bit", }, },
  "qwen-2.5-coder-32b": { "layers": 64, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", }, },
  "qwen-2.5-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit", }, },
  "qwen-2.5-math-72b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit", }, },
  ### qwen3 (latest generation with thinking mode support) - separate quantizations
  # Qwen3-0.6B variants
  "qwen3-0.6b-q4-k-m": { "layers": 24, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", }, },
  "qwen3-0.6b-q5-k-m": { "layers": 24, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q5_K_M.gguf", }, },
  "qwen3-0.6b-q8-0": { "layers": 24, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", }, },
  
  # Qwen3-4B variants  
  "qwen3-4b-q4-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", }, },
  "qwen3-4b-q5-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q5_K_M.gguf", }, },
  "qwen3-4b-q8-0": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf", }, },
  
  # Qwen3-32B variants
  "qwen3-32b-q4-k-m": { "layers": 64, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf", }, },
  "qwen3-32b-q5-k-m": { "layers": 64, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-32B-GGUF/Qwen3-32B-Q5_K_M.gguf", }, },
  "qwen3-32b-q8-0": { "layers": 64, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-32B-GGUF/Qwen3-32B-Q8_0.gguf", }, },
  
  # Qwen3-30B-A3B variants (MoE model - your requested model!)
  "qwen3-30b-a3b-q2-k": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q2_K.gguf", }, },
  "qwen3-30b-a3b-q3-k-m": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q3_K_M.gguf", }, },
  "qwen3-30b-a3b-q4-k-m": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf", }, },
  "qwen3-30b-a3b-q5-k-m": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q5_K_M.gguf", }, },
  "qwen3-30b-a3b-q6-k": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q6_K.gguf", }, },
  "qwen3-30b-a3b-q8-0": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q8_0.gguf", }, },
  "qwen3-30b-a3b-bf16": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-GGUF/BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf", }, },
  
  # Qwen3-30B-A3B 128K Context variants
  "qwen3-30b-a3b-128k-q4-k-m": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-128K-GGUF/Qwen3-30B-A3B-128K-Q4_K_M.gguf", }, },
  "qwen3-30b-a3b-128k-q5-k-m": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-128K-GGUF/Qwen3-30B-A3B-128K-Q5_K_M.gguf", }, },
  "qwen3-30b-a3b-128k-q8-0": { "layers": 48, "repo": { "LlamaCppInferenceEngine": "unsloth/Qwen3-30B-A3B-128K-GGUF/Qwen3-30B-A3B-128K-Q8_0.gguf", }, },
  ### additional unsloth models with specific quantizations
  # Unsloth Llama 3.1 8B variants
  "unsloth-llama-3.1-8b-q4-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Llama-3.1-8B-Instruct-GGUF/Q4_K_M.gguf", }, },
  "unsloth-llama-3.1-8b-q5-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Llama-3.1-8B-Instruct-GGUF/Q5_K_M.gguf", }, },
  "unsloth-llama-3.1-8b-q8-0": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Llama-3.1-8B-Instruct-GGUF/Q8_0.gguf", }, },
  
  # Unsloth Llama 3.1 70B variants 
  "unsloth-llama-3.1-70b-q4-k-m": { "layers": 80, "repo": { "LlamaCppInferenceEngine": "unsloth/Llama-3.1-70B-Instruct-GGUF/Q4_K_M.gguf", }, },
  "unsloth-llama-3.1-70b-q5-k-m": { "layers": 80, "repo": { "LlamaCppInferenceEngine": "unsloth/Llama-3.1-70B-Instruct-GGUF/Q5_K_M.gguf", }, },
  
  # Unsloth Mistral 7B variants
  "unsloth-mistral-7b-q4-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Mistral-7B-Instruct-v0.3-GGUF/Q4_K_M.gguf", }, },
  "unsloth-mistral-7b-q5-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/Mistral-7B-Instruct-v0.3-GGUF/Q5_K_M.gguf", }, },
  
  # Unsloth CodeLlama variants
  "unsloth-codellama-7b-q4-k-m": { "layers": 32, "repo": { "LlamaCppInferenceEngine": "unsloth/CodeLlama-7b-Instruct-hf-GGUF/Q4_K_M.gguf", }, },
  "unsloth-codellama-13b-q4-k-m": { "layers": 40, "repo": { "LlamaCppInferenceEngine": "unsloth/CodeLlama-13b-Instruct-hf-GGUF/Q4_K_M.gguf", }, },
  ### nemotron
  "nemotron-70b": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit", }, },
  "nemotron-70b-bf16": { "layers": 80, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16", }, },
  # gemma
  "gemma2-9b": { "layers": 42, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit", }, },
  "gemma2-27b": { "layers": 46, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit", }, },
  # stable diffusion
  "stable-diffusion-2-1-base": { "layers": 31, "repo": { "MLXDynamicShardInferenceEngine": "stabilityai/stable-diffusion-2-1-base" } },
  # phi
  "phi-3.5-mini": { "layers": 32, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/Phi-3.5-mini-instruct-4bit", }, },
  "phi-4": { "layers": 40, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/phi-4-4bit", }, },
  # dummy
  "dummy": { "layers": 8, "repo": { "DummyInferenceEngine": "dummy", }, },
}

pretty_name = {
  "llama-3.3-70b": "Llama 3.3 70B",
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-1b-q4-k-m": "Llama 3.2 1B Q4_K_M (~1GB)",
  "llama-3.2-1b-q5-k-m": "Llama 3.2 1B Q5_K_M (~1.2GB)",
  "llama-3.2-1b-q8-0": "Llama 3.2 1B Q8_0 (~1.6GB)",
  "llama-3.2-1b-f16": "Llama 3.2 1B F16 (~2.6GB)",
  "llama-3.2-1b-8bit": "Llama 3.2 1B (8-bit)",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.2-3b-8bit": "Llama 3.2 3B (8-bit)",
  "llama-3.2-3b-bf16": "Llama 3.2 3B (BF16)",
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
  "deepseek-v3": "Deepseek V3 (4-bit)",
  "deepseek-v3-3bit": "Deepseek V3 (3-bit)",
  "deepseek-r1": "Deepseek R1 (4-bit)",
  "deepseek-r1-3bit": "Deepseek R1 (3-bit)",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-0.5b": "Qwen 2.5 0.5B",
  "qwen-2.5-1.5b": "Qwen 2.5 1.5B",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-3b": "Qwen 2.5 3B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-32b": "Qwen 2.5 32B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "phi-3.5-mini": "Phi-3.5 Mini",
  "phi-4": "Phi-4",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
  "stable-diffusion-2-1-base": "Stable Diffusion 2.1",
  "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 Distill Qwen 1.5B",
  "deepseek-r1-distill-qwen-1.5b-3bit": "DeepSeek R1 Distill Qwen 1.5B (3-bit)",
  "deepseek-r1-distill-qwen-1.5b-6bit": "DeepSeek R1 Distill Qwen 1.5B (6-bit)",
  "deepseek-r1-distill-qwen-1.5b-8bit": "DeepSeek R1 Distill Qwen 1.5B (8-bit)",
  "deepseek-r1-distill-qwen-1.5b-bf16": "DeepSeek R1 Distill Qwen 1.5B (BF16)",
  "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
  "deepseek-r1-distill-qwen-7b-3bit": "DeepSeek R1 Distill Qwen 7B (3-bit)",
  "deepseek-r1-distill-qwen-7b-6bit": "DeepSeek R1 Distill Qwen 7B (6-bit)",
  "deepseek-r1-distill-qwen-7b-8bit": "DeepSeek R1 Distill Qwen 7B (8-bit)",
  "deepseek-r1-distill-qwen-7b-bf16": "DeepSeek R1 Distill Qwen 7B (BF16)",
  "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
  "deepseek-r1-distill-qwen-14b-3bit": "DeepSeek R1 Distill Qwen 14B (3-bit)",
  "deepseek-r1-distill-qwen-14b-6bit": "DeepSeek R1 Distill Qwen 14B (6-bit)",
  "deepseek-r1-distill-qwen-14b-8bit": "DeepSeek R1 Distill Qwen 14B (8-bit)",
  "deepseek-r1-distill-qwen-14b-bf16": "DeepSeek R1 Distill Qwen 14B (BF16)",
  "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
  "deepseek-r1-distill-qwen-32b-3bit": "DeepSeek R1 Distill Qwen 32B (3-bit)",
  "deepseek-r1-distill-qwen-32b-8bit": "DeepSeek R1 Distill Qwen 32B (8-bit)",
  "deepseek-r1-distill-qwen-32b-bf16": "DeepSeek R1 Distill Qwen 32B (BF16)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
  "deepseek-r1-distill-llama-8b-3bit": "DeepSeek R1 Distill Llama 8B (3-bit)",
  "deepseek-r1-distill-llama-8b-6bit": "DeepSeek R1 Distill Llama 8B (6-bit)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-8b-bf16": "DeepSeek R1 Distill Llama 8B (BF16)",
  "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
  "deepseek-r1-distill-llama-70b-3bit": "DeepSeek R1 Distill Llama 70B (3-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-qwen-32b-6bit": "DeepSeek R1 Distill Qwen 32B (6-bit)",
  # qwen3 models with quantization info
  "qwen3-0.6b-q4-k-m": "Qwen3 0.6B Q4_K_M (~1.5GB)",
  "qwen3-0.6b-q5-k-m": "Qwen3 0.6B Q5_K_M (~1.8GB)",
  "qwen3-0.6b-q8-0": "Qwen3 0.6B Q8_0 (~2.3GB)",
  "qwen3-4b-q4-k-m": "Qwen3 4B Q4_K_M (~2.8GB)",
  "qwen3-4b-q5-k-m": "Qwen3 4B Q5_K_M (~3.2GB)",
  "qwen3-4b-q8-0": "Qwen3 4B Q8_0 (~4.1GB)",
  "qwen3-32b-q4-k-m": "Qwen3 32B Q4_K_M (~20GB)",
  "qwen3-32b-q5-k-m": "Qwen3 32B Q5_K_M (~23GB)",
  "qwen3-32b-q8-0": "Qwen3 32B Q8_0 (~33GB)",
  "qwen3-30b-a3b-q2-k": "Qwen3 30B-A3B Q2_K (~11GB, MoE)",
  "qwen3-30b-a3b-q3-k-m": "Qwen3 30B-A3B Q3_K_M (~15GB, MoE)",
  "qwen3-30b-a3b-q4-k-m": "Qwen3 30B-A3B Q4_K_M (~19GB, MoE)",
  "qwen3-30b-a3b-q5-k-m": "Qwen3 30B-A3B Q5_K_M (~22GB, MoE)",
  "qwen3-30b-a3b-q6-k": "Qwen3 30B-A3B Q6_K (~25GB, MoE)",
  "qwen3-30b-a3b-q8-0": "Qwen3 30B-A3B Q8_0 (~33GB, MoE)",
  "qwen3-30b-a3b-bf16": "Qwen3 30B-A3B BF16 (~61GB, MoE)",
  "qwen3-30b-a3b-128k-q4-k-m": "Qwen3 30B-A3B 128K Q4_K_M (~19GB, Extended Context)",
  "qwen3-30b-a3b-128k-q5-k-m": "Qwen3 30B-A3B 128K Q5_K_M (~22GB, Extended Context)", 
  "qwen3-30b-a3b-128k-q8-0": "Qwen3 30B-A3B 128K Q8_0 (~33GB, Extended Context)",
  # unsloth models with quantization info
  "unsloth-llama-3.1-8b-q4-k-m": "Unsloth Llama 3.1 8B Q4_K_M (~5GB)",
  "unsloth-llama-3.1-8b-q5-k-m": "Unsloth Llama 3.1 8B Q5_K_M (~6GB)",
  "unsloth-llama-3.1-8b-q8-0": "Unsloth Llama 3.1 8B Q8_0 (~8GB)",
  "unsloth-llama-3.1-70b-q4-k-m": "Unsloth Llama 3.1 70B Q4_K_M (~41GB)",
  "unsloth-llama-3.1-70b-q5-k-m": "Unsloth Llama 3.1 70B Q5_K_M (~48GB)",
  "unsloth-mistral-7b-q4-k-m": "Unsloth Mistral 7B Q4_K_M (~4GB)",
  "unsloth-mistral-7b-q5-k-m": "Unsloth Mistral 7B Q5_K_M (~5GB)",
  "unsloth-codellama-7b-q4-k-m": "Unsloth CodeLlama 7B Q4_K_M (~4GB)",
  "unsloth-codellama-13b-q4-k-m": "Unsloth CodeLlama 13B Q4_K_M (~8GB)",
}

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  return model_cards.get(model_id, {}).get("repo", {}).get(inference_engine_classname, None)

def get_pretty_name(model_id: str) -> Optional[str]:
  return pretty_name.get(model_id, None)

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  repo = get_repo(model_id, inference_engine_classname)
  n_layers = model_cards.get(model_id, {}).get("layers", 0)
  if repo is None or n_layers < 1:
    return None
  return Shard(model_id, 0, 0, n_layers)

def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  base_shard = build_base_shard(model_id, inference_engine_classname)
  if base_shard is None: return None
  return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)

def get_supported_models(supported_inference_engine_lists: Optional[List[List[str]]] = None) -> List[str]:
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
