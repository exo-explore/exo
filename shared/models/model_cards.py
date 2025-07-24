from typing import List

from pydantic import BaseModel

from shared.types.models import ModelMetadata


class ModelCard(BaseModel):
    id: str
    repo_id: str
    name: str
    description: str
    tags: List[str]
    metadata: ModelMetadata


MODEL_CARDS = {
  "llama-3.3": ModelCard(
    id="llama-3.3",
    repo_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
    name="Llama 3.3 70B",
    description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
      pretty_name="Llama 3.3 70B",
      storage_size_kilobytes=38758160,
      n_layers=80,
    ),
  ),
  "llama-3.3:70b": ModelCard(
    id="llama-3.3:70b",
    repo_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
    name="Llama 3.3 70B",
    description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
      pretty_name="Llama 3.3 70B",
      storage_size_kilobytes=38758160,
      n_layers=80,
    ),
  ),
  "llama-3.2": ModelCard(
    id="llama-3.2",
    repo_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
    name="Llama 3.2 1B",
    description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
      pretty_name="Llama 3.2 1B",
      storage_size_kilobytes=678948,
      n_layers=16,
    ),
  ),
  "llama-3.2:1b": ModelCard(
    id="llama-3.2:1b",
    repo_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
    name="Llama 3.2 1B",
    description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
      pretty_name="Llama 3.2 1B",
      storage_size_kilobytes=678948,
      n_layers=16,
    ),
  ),
  "llama-3.2:3b": ModelCard(
    id="llama-3.2:3b",
    repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
    name="Llama 3.2 3B",
    description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
      pretty_name="Llama 3.2 3B",
      storage_size_kilobytes=1765062,
      n_layers=28,
    ),
  ),
  "llama-3.1:8b": ModelCard(
    id="llama-3.1:8b",
    repo_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    name="Llama 3.1 8B",
    description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
      pretty_name="Llama 3.1 8B",
      storage_size_kilobytes=4411528,
      n_layers=32,
    ),
  ),
  "llama-3.1-70b": ModelCard(
    id="llama-3.1-70b",
    repo_id="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    name="Llama 3.1 70B",
    description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
      pretty_name="Llama 3.1 70B",
      storage_size_kilobytes=38758160,
      n_layers=80,
    ),
  ),
  "deepseek-r1": ModelCard(
    id="deepseek-r1",
    repo_id="mlx-community/DeepSeek-R1-4bit",
    name="DeepSeek R1 671B (4-bit)",
    description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/DeepSeek-R1-4bit",
      pretty_name="DeepSeek R1 671B (4-bit)",
      storage_size_kilobytes=409706307,
      n_layers=61,
    ),
  ),
  "deepseek-r1:671b": ModelCard(
    id="deepseek-r1:671b",
    repo_id="mlx-community/DeepSeek-R1-4bit",
    name="DeepSeek R1 671B",
    description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/DeepSeek-R1-4bit",
      pretty_name="DeepSeek R1 671B",
      storage_size_kilobytes=409706307,
      n_layers=61,
    ),
  ),
  "deepseek-v3": ModelCard(
    id="deepseek-v3",
    repo_id="mlx-community/DeepSeek-V3-0324-4bit",
    name="DeepSeek V3 4B",
    description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/DeepSeek-V3-0324-4bit",
      pretty_name="DeepSeek V3 4B",
      storage_size_kilobytes=368756663,
      n_layers=61,
    ),
  ),
  "deepseek-v3:671b": ModelCard(
    id="deepseek-v3:671b",
    repo_id="mlx-community/DeepSeek-V3-0324-4bit",
    name="DeepSeek V3 671B",
    description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/DeepSeek-V3-0324-4bit",
      pretty_name="DeepSeek V3 671B",
      storage_size_kilobytes=368756663,
      n_layers=61,
    ),
  ),
  "phi-3-mini": ModelCard(
    id="phi-3-mini",
    repo_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
    name="Phi 3 Mini 128k",
    description="""Phi 3 Mini is a large language model trained on the Phi 3 Mini dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
      pretty_name="Phi 3 Mini 128k",
      storage_size_kilobytes=2099262,
      n_layers=32,
    ),
  ),
  "phi-3-mini:128k": ModelCard(
    id="phi-3-mini:128k",
    repo_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
    name="Phi 3 Mini 128k",
    description="""Phi 3 Mini is a large language model trained on the Phi 3 Mini dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
      pretty_name="Phi 3 Mini 128k",
      storage_size_kilobytes=2099262,
      n_layers=32,
    ),
  ),
  "qwen3-0.6b": ModelCard(
    id="qwen3-0.6b",
    repo_id="mlx-community/Qwen3-0.6B-4bit",
    name="Qwen3 0.6B",
    description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Qwen3-0.6B-4bit",
      pretty_name="Qwen3 0.6B",
      storage_size_kilobytes=327512,
      n_layers=28,
    ),
  ),
  "qwen3-30b": ModelCard(
    id="qwen3-30b",
    repo_id="mlx-community/Qwen3-30B-A3B-4bit",
    name="Qwen3 30B (Active 3B)",
    description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/Qwen3-30B-A3B-4bit",
      pretty_name="Qwen3 30B (Active 3B)",
      storage_size_kilobytes=16772092,
      n_layers=48,
    ),
  ),
  "granite-3.3-2b": ModelCard(
    id="granite-3.3-2b",
    repo_id="mlx-community/granite-3.3-2b-instruct-fp16",
    name="Granite 3.3 2B",
    description="""Granite-3.3-2B-Instruct is a 2-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/granite-3.3-2b-instruct-fp16",
      pretty_name="Granite 3.3 2B",
      storage_size_kilobytes=4948320,
      n_layers=40,
    ),
  ),
  "granite-3.3-8b": ModelCard(
    id="granite-3.3-8b",
    repo_id="mlx-community/granite-3.3-8b-instruct-fp16",
    name="Granite 3.3 8B",
    description="""Granite-3.3-8B-Instruct is a 8-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/granite-3.3-8b-instruct-fp16",
      pretty_name="Granite 3.3 8B",
      storage_size_kilobytes=15958720,
      n_layers=40,
    ),
  ),
  "smol-lm-135m": ModelCard(
    id="smol-lm-135m",
    repo_id="mlx-community/SmolLM-135M-4bit",
    name="Smol LM 135M",
    description="""SmolLM is a series of state-of-the-art small language models available in three sizes: 135M, 360M, and 1.7B parameters. """,
    tags=[],
    metadata=ModelMetadata(
      model_id="mlx-community/SmolLM-135M-4bit",
      pretty_name="Smol LM 135M",
      storage_size_kilobytes=73940,
      n_layers=30,
    ),
  ),
}
