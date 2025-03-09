from pydantic import BaseModel
from pathlib import Path

from exo.inference.shard import Shard
from typing import Optional, List, Literal

# InferenceEngineType = Literal["MLXDynamicShardInferenceEngine", "TinygradDynamicShardInferenceEngine", "DummyInferenceEngine"]
InferenceEngineType = str

class ModelCard(BaseModel):
  pretty_name: str
  layers: int
  repo: dict[InferenceEngineType, str]

  chat_template: Optional[str] = None

ModelCardCollection = dict[str, ModelCard]

model_cards: ModelCardCollection = {
  ### llama
  "llama-3.3-70b": ModelCard(
    pretty_name="Llama 3.3 70B",
    layers=80,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.3-70B-Instruct",
    },
  ),
  "llama-3.2-1b": ModelCard(
    pretty_name="Llama 3.2 1B",
    layers=16,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  ),
  "llama-3.2-1b-8bit": ModelCard(
    pretty_name="Llama 3.2 1B (8-bit)",
    layers=16,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-1B-Instruct-8bit",
      "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-1B-Instruct",
    },
  ),
  "llama-3.2-3b": ModelCard(
    pretty_name="Llama 3.2 3B",
    layers=28,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  ),
  "llama-3.2-3b-8bit": ModelCard(
    pretty_name="Llama 3.2 3B (8-bit)",
    layers=28,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct-8bit",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  ),
  "llama-3.2-3b-bf16": ModelCard(
    pretty_name="Llama 3.2 3B (BF16)",
    layers=28,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.2-3B-Instruct",
       "TinygradDynamicShardInferenceEngine": "unsloth/Llama-3.2-3B-Instruct",
    },
  ),
  "llama-3.1-8b": ModelCard(
    pretty_name="Llama 3.1 8B",
    layers=32,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    },
  ),
  "llama-3.1-70b": ModelCard(
    pretty_name="Llama 3.1 70B",
    layers=80,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  ),
  "llama-3.1-70b-bf16": ModelCard(
    pretty_name="Llama 3.1 70B (BF16)",
    layers=80,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED",
       "TinygradDynamicShardInferenceEngine": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    },
  ),
  "llama-3-8b": ModelCard(
    pretty_name="Llama 3 8B",
    layers=32,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R",
    },
  ),
  "llama-3-70b": ModelCard(
    pretty_name="Llama 3 70B",
    layers=80,
    repo={
       "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
       "TinygradDynamicShardInferenceEngine": "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R",
    },
  ),
  "llama-3.1-405b": ModelCard(
    pretty_name="Llama 3.1 405B",
    layers=126,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-4bit",
    },
  ),
  "llama-3.1-405b-8bit": ModelCard(
    pretty_name="Llama 3.1 405B (8-bit)",
    layers=126,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit",
    },
  ),
  ### mistral
  "mistral-nemo": ModelCard(
    pretty_name="Mistral Nemo",
    layers=40,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    },
  ),
  "mistral-large": ModelCard(
    pretty_name="Mistral Large",
    layers=88,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Mistral-Large-Instruct-2407-4bit",
    },
  ),
  ### deepseek
  "deepseek-coder-v2-lite": ModelCard(
    pretty_name="Deepseek Coder V2 Lite",
    layers=27,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    },
  ),
  "deepseek-coder-v2.5": ModelCard(
    pretty_name="Deepseek Coder V2.5",
    layers=60,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64",
    },
  ),
  "deepseek-v3": ModelCard(
    pretty_name="Deepseek V3 (4-bit)",
    layers=61,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-4bit",
    },
  ),
  "deepseek-v3-3bit": ModelCard(
    pretty_name="Deepseek V3 (3-bit)",
    layers=61,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-V3-3bit",
    },
  ),
  "deepseek-r1": ModelCard(
    pretty_name="Deepseek R1 (4-bit)",
    layers=61,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-4bit",
    },
  ),
  "deepseek-r1-3bit": ModelCard(
    pretty_name="Deepseek R1 (3-bit)",
    layers=61,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit",
    },
  ),
  ### deepseek distills
  "deepseek-r1-distill-qwen-1.5b": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 1.5B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/deepseek-r1-distill-qwen-1.5b",
    },
  ),
  "deepseek-r1-distill-qwen-1.5b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 1.5B (3-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit",
    },
  ),
  "deepseek-r1-distill-qwen-1.5b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 1.5B (6-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-6bit",
    },
  ),
  "deepseek-r1-distill-qwen-1.5b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 1.5B (8-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit",
    },
  ),
  "deepseek-r1-distill-qwen-1.5b-bf16": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 1.5B (BF16)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-bf16",
    },
  ),
  "deepseek-r1-distill-qwen-7b": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 7B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    },
  ),
  "deepseek-r1-distill-qwen-7b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 7B (3-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit",
    },
  ),
  "deepseek-r1-distill-qwen-7b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 7B (6-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-6bit",
    },
  ),
  "deepseek-r1-distill-qwen-7b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 7B (8-bit)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit",
    },
  ),
  "deepseek-r1-distill-qwen-7b-bf16": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 7B (BF16)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16",
    },
  ),
  "deepseek-r1-distill-qwen-14b": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 14B",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    },
  ),
  "deepseek-r1-distill-qwen-14b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 14B (3-bit)",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-3bit",
    },
  ),
  "deepseek-r1-distill-qwen-14b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 14B (6-bit)",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-6bit",
    },
  ),
  "deepseek-r1-distill-qwen-14b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 14B (8-bit)",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit",
    },
  ),
  "deepseek-r1-distill-qwen-14b-bf16": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 14B (BF16)",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-bf16",
    },
  ),
  "deepseek-r1-distill-qwen-32b": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 32B",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    },
  ),
  "deepseek-r1-distill-qwen-32b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 32B (3-bit)",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-3bit",
    },
  ),
  "deepseek-r1-distill-qwen-32b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 32B (6-bit)",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-6bit",
    },
  ),
  "deepseek-r1-distill-qwen-32b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 32B (8-bit)",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit",
    },
  ),
  "deepseek-r1-distill-qwen-32b-bf16": ModelCard(
    pretty_name="DeepSeek R1 Distill Qwen 32B (BF16)",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-bf16",
    },
  ),
  "deepseek-r1-distill-llama-8b": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 8B",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
    },
  ),
  "deepseek-r1-distill-llama-8b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 8B (3-bit)",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-3bit",
    },
  ),
  "deepseek-r1-distill-llama-8b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 8B (6-bit)",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-6bit",
    },
  ),
  "deepseek-r1-distill-llama-8b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 8B (8-bit)",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit",
    },
  ),
  "deepseek-r1-distill-llama-8b-bf16": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 8B (BF16)",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16",
    },
  ),
  "deepseek-r1-distill-llama-70b": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 70B",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
    },
  ),
  "deepseek-r1-distill-llama-70b-3bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 70B (3-bit)",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-3bit",
    },
  ),
  "deepseek-r1-distill-llama-70b-6bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 70B (6-bit)",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-6bit",
    },
  ),
  "deepseek-r1-distill-llama-70b-8bit": ModelCard(
    pretty_name="DeepSeek R1 Distill Llama 70B (8-bit)",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit",
    },
  ),
  ### llava
  "llava-1.5-7b-hf": ModelCard(
    pretty_name="LLaVa 1.5 7B (Vision Model)",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "llava-hf/llava-1.5-7b-hf",
    },
  ),
  ### qwen
  "qwen-2.5-0.5b": ModelCard(
    pretty_name="Qwen 2.5 0.5B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    },
  ),
  "qwen-2.5-1.5b": ModelCard(
    pretty_name="Qwen 2.5 1.5B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    },
  ),
  "qwen-2.5-coder-1.5b": ModelCard(
    pretty_name="Qwen 2.5 Coder 1.5B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    },
  ),
  "qwen-2.5-3b": ModelCard(
    pretty_name="Qwen 2.5 3B",
    layers=36,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    },
  ),
  "qwen-2.5-coder-3b": ModelCard(
    pretty_name="Qwen 2.5 Coder 3B",
    layers=36,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
    },
  ),
  "qwen-2.5-7b": ModelCard(
    pretty_name="Qwen 2.5 7B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
  ),
  "qwen-2.5-coder-7b": ModelCard(
    pretty_name="Qwen 2.5 Coder 7B",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    },
  ),
  "qwen-2.5-math-7b": ModelCard(
    pretty_name="Qwen 2.5 7B (Math)",
    layers=28,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
    },
  ),
  "qwen-2.5-14b": ModelCard(
    pretty_name="Qwen 2.5 14B",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    },
  ),
  "qwen-2.5-coder-14b": ModelCard(
    pretty_name="Qwen 2.5 Coder 14B",
    layers=48,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    },
  ),
  "qwen-2.5-32b": ModelCard(
    pretty_name="Qwen 2.5 32B",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    },
  ),
  "qwen-2.5-coder-32b": ModelCard(
    pretty_name="Qwen 2.5 Coder 32B",
    layers=64,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    },
  ),
  "qwen-2.5-72b": ModelCard(
    pretty_name="Qwen 2.5 72B",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-72B-Instruct-4bit",
    },
  ),
  "qwen-2.5-math-72b": ModelCard(
    pretty_name="Qwen 2.5 72B (Math)",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Qwen2.5-Math-72B-Instruct-4bit",
    },
  ),
  ### nemotron
  "nemotron-70b": ModelCard(
    pretty_name="Nemotron 70B",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/nvidia_Llama-3.1-Nemotron-70B-Instruct-HF_4bit",
    },
  ),
  "nemotron-70b-bf16": ModelCard(
    pretty_name="Nemotron 70B (BF16)",
    layers=80,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-bf16",
    },
  ),
  # gemma
  "gemma2-9b": ModelCard(
    pretty_name="Gemma2 9B",
    layers=42,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-9b-it-4bit",
    },
  ),
  "gemma2-27b": ModelCard(
    pretty_name="Gemma2 27B",
    layers=46,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/gemma-2-27b-it-4bit",
    },
  ),
  # stable diffusion
  "stable-diffusion-2-1-base": ModelCard(
    pretty_name="Stable Diffusion 2.1",
    layers=31,
    repo={
      "MLXDynamicShardInferenceEngine": "stabilityai/stable-diffusion-2-1-base"
    },
  ),
  # phi
  "phi-3.5-mini": ModelCard(
    pretty_name="Phi-3.5 Mini",
    layers=32,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/Phi-3.5-mini-instruct-4bit",
    },
  ),
  "phi-4": ModelCard(
    pretty_name="Phi-4",
    layers=40,
    repo={
      "MLXDynamicShardInferenceEngine": "mlx-community/phi-4-4bit",
    },
  ),
  # dummy
  "dummy": ModelCard(
    pretty_name="Dummy",
    layers=8,
    repo={
      "DummyInferenceEngine": "dummy",
    },
  ),
}

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  model_card = model_cards.get(model_id)

  if model_card:
    return model_card.repo.get(inference_engine_classname)
  else:
    return None

def get_pretty_name(model_id: str) -> Optional[str]:
  model_card = model_cards.get(model_id)
  return model_card.pretty_name if model_card else None

def get_default_tool_format(model_id: str) -> Optional[str]:
  ...

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  model_card = model_cards.get(model_id)
  if not model_card:
    return None
  repo = get_repo(model_id, inference_engine_classname)
  n_layers = model_card.layers if model_card else 0
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

  def has_any_engine(model_info: ModelCard, engine_list: List[str]) -> bool:
    return any(engine in model_info.repo for engine in engine_list)

  def supports_all_engine_lists(model_info: ModelCard) -> bool:
    return all(has_any_engine(model_info, engine_list)
              for engine_list in supported_inference_engine_lists)

  return [
    model_id for model_id, model_info in model_cards.items()
    if supports_all_engine_lists(model_info)
  ]


def load_additional_models(additional_models_path: Path):
  import json

  try:
    with open(additional_models_path, 'r') as f:
      additional_models = json.load(f)

    for model_id, model_info in additional_models.items():
      model_cards[model_id] = ModelCard.model_validate(model_info)

    print(f"Loaded {len(additional_models)} additional models from {additional_models_path}")
  except Exception as e:
    print(f"Error loading additional models from {additional_models_path}: {e}")
