import json
import re
from typing import Any, cast
from loguru import logger
from exo.shared.constants import EXO_CONFIG_HOME
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    short_id: str
    model_id: ModelId
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata


MODEL_CARDS: dict[str, ModelCard] = {
    # deepseek v3
    # "deepseek-v3-0324:4bit": ModelCard(
    #     short_id="deepseek-v3-0324:4bit",
    #     model_id="mlx-community/DeepSeek-V3-0324-4bit",
    #     name="DeepSeek V3 0324 (4-bit)",
    #     description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-V3-0324-4bit"),
    #         pretty_name="DeepSeek V3 0324 (4-bit)",
    #         storage_size=Memory.from_kb(409706307),
    #         n_layers=61,
    #     ),
    # ),
    # "deepseek-v3-0324": ModelCard(
    #     short_id="deepseek-v3-0324",
    #     model_id="mlx-community/DeepSeek-v3-0324-8bit",
    #     name="DeepSeek V3 0324 (8-bit)",
    #     description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-v3-0324-8bit"),
    #         pretty_name="DeepSeek V3 0324 (8-bit)",
    #         storage_size=Memory.from_kb(754706307),
    #         n_layers=61,
    #     ),
    # ),
    "deepseek-v3.1-4bit": ModelCard(
        short_id="deepseek-v3.1-4bit",
        model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
        name="DeepSeek V3.1 (4-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
            pretty_name="DeepSeek V3.1 (4-bit)",
            storage_size=Memory.from_gb(378),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    "deepseek-v3.1-8bit": ModelCard(
        short_id="deepseek-v3.1-8bit",
        model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
        name="DeepSeek V3.1 (8-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
            pretty_name="DeepSeek V3.1 (8-bit)",
            storage_size=Memory.from_gb(713),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    # "deepseek-v3.2": ModelCard(
    #     short_id="deepseek-v3.2",
    #     model_id=ModelId("mlx-community/DeepSeek-V3.2-8bit"),
    #     name="DeepSeek V3.2 (8-bit)",
    #     description="""DeepSeek V3.2 is a large language model trained on the DeepSeek V3.2 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-V3.2-8bit"),
    #         pretty_name="DeepSeek V3.2 (8-bit)",
    #         storage_size=Memory.from_kb(754706307),
    #         n_layers=61,
    #         hidden_size=7168,
    #     ),
    # ),
    # "deepseek-v3.2-4bit": ModelCard(
    #     short_id="deepseek-v3.2-4bit",
    #     model_id=ModelId("mlx-community/DeepSeek-V3.2-4bit"),
    #     name="DeepSeek V3.2 (4-bit)",
    #     description="""DeepSeek V3.2 is a large language model trained on the DeepSeek V3.2 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-V3.2-4bit"),
    #         pretty_name="DeepSeek V3.2 (4-bit)",
    #         storage_size=Memory.from_kb(754706307 // 2),  # TODO !!!!!
    #         n_layers=61,
    #         hidden_size=7168,
    #     ),
    # ),
    # deepseek r1
    # "deepseek-r1-0528-4bit": ModelCard(
    #     short_id="deepseek-r1-0528-4bit",
    #     model_id="mlx-community/DeepSeek-R1-0528-4bit",
    #     name="DeepSeek-R1-0528 (4-bit)",
    #     description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-R1-0528-4bit"),
    #         pretty_name="DeepSeek R1 671B (4-bit)",
    #         storage_size=Memory.from_kb(409706307),
    #         n_layers=61,
    #         hidden_size=7168,
    #     ),
    # ),
    # "deepseek-r1-0528": ModelCard(
    #     short_id="deepseek-r1-0528",
    #     model_id="mlx-community/DeepSeek-R1-0528-8bit",
    #     name="DeepSeek-R1-0528 (8-bit)",
    #     description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/DeepSeek-R1-0528-8bit"),
    #         pretty_name="DeepSeek R1 671B (8-bit)",
    #         storage_size=Memory.from_bytes(754998771712),
    #         n_layers=61,
    # .       hidden_size=7168,
    #     ),
    # ),
    # kimi k2
    "kimi-k2-instruct-4bit": ModelCard(
        short_id="kimi-k2-instruct-4bit",
        model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
        name="Kimi K2 Instruct (4-bit)",
        description="""Kimi K2 is a large language model trained on the Kimi K2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
            pretty_name="Kimi K2 Instruct (4-bit)",
            storage_size=Memory.from_gb(578),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    "kimi-k2-thinking": ModelCard(
        short_id="kimi-k2-thinking",
        model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
        name="Kimi K2 Thinking (4-bit)",
        description="""Kimi K2 Thinking is the latest, most capable version of open-source thinking model.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
            pretty_name="Kimi K2 Thinking (4-bit)",
            storage_size=Memory.from_gb(658),
            n_layers=61,
            hidden_size=7168,
            supports_tensor=True,
        ),
    ),
    # llama-3.1
    "llama-3.1-8b": ModelCard(
        short_id="llama-3.1-8b",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        name="Llama 3.1 8B (4-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
            pretty_name="Llama 3.1 8B (4-bit)",
            storage_size=Memory.from_mb(4423),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-8b-8bit": ModelCard(
        short_id="llama-3.1-8b-8bit",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
        name="Llama 3.1 8B (8-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
            pretty_name="Llama 3.1 8B (8-bit)",
            storage_size=Memory.from_mb(8540),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-8b-bf16": ModelCard(
        short_id="llama-3.1-8b-bf16",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
        name="Llama 3.1 8B (BF16)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
            pretty_name="Llama 3.1 8B (BF16)",
            storage_size=Memory.from_mb(16100),
            n_layers=32,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "llama-3.1-70b": ModelCard(
        short_id="llama-3.1-70b",
        model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
        name="Llama 3.1 70B (4-bit)",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
            pretty_name="Llama 3.1 70B (4-bit)",
            storage_size=Memory.from_mb(38769),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    # llama-3.2
    "llama-3.2-1b": ModelCard(
        short_id="llama-3.2-1b",
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        name="Llama 3.2 1B (4-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
            pretty_name="Llama 3.2 1B (4-bit)",
            storage_size=Memory.from_mb(696),
            n_layers=16,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "llama-3.2-3b": ModelCard(
        short_id="llama-3.2-3b",
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        name="Llama 3.2 3B (4-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
            pretty_name="Llama 3.2 3B (4-bit)",
            storage_size=Memory.from_mb(1777),
            n_layers=28,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
    "llama-3.2-3b-8bit": ModelCard(
        short_id="llama-3.2-3b-8bit",
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
        name="Llama 3.2 3B (8-bit)",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
            pretty_name="Llama 3.2 3B (8-bit)",
            storage_size=Memory.from_mb(3339),
            n_layers=28,
            hidden_size=3072,
            supports_tensor=True,
        ),
    ),
    # llama-3.3
    "llama-3.3-70b": ModelCard(
        short_id="llama-3.3-70b",
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
        name="Llama 3.3 70B (4-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
            pretty_name="Llama 3.3 70B",
            storage_size=Memory.from_mb(38769),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    "llama-3.3-70b-8bit": ModelCard(
        short_id="llama-3.3-70b-8bit",
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
        name="Llama 3.3 70B (8-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
            pretty_name="Llama 3.3 70B (8-bit)",
            storage_size=Memory.from_mb(73242),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    "llama-3.3-70b-fp16": ModelCard(
        short_id="llama-3.3-70b-fp16",
        model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
        name="Llama 3.3 70B (FP16)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
            pretty_name="Llama 3.3 70B (FP16)",
            storage_size=Memory.from_mb(137695),
            n_layers=80,
            hidden_size=8192,
            supports_tensor=True,
        ),
    ),
    # qwen3
    "qwen3-0.6b": ModelCard(
        short_id="qwen3-0.6b",
        model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
        name="Qwen3 0.6B (4-bit)",
        description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
            pretty_name="Qwen3 0.6B (4-bit)",
            storage_size=Memory.from_mb(327),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
        ),
    ),
    "qwen3-0.6b-8bit": ModelCard(
        short_id="qwen3-0.6b-8bit",
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        name="Qwen3 0.6B (8-bit)",
        description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
            pretty_name="Qwen3 0.6B (8-bit)",
            storage_size=Memory.from_mb(666),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
        ),
    ),
    "qwen3-30b": ModelCard(
        short_id="qwen3-30b",
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
        name="Qwen3 30B A3B (4-bit)",
        description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
            pretty_name="Qwen3 30B A3B (4-bit)",
            storage_size=Memory.from_mb(16797),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-30b-8bit": ModelCard(
        short_id="qwen3-30b-8bit",
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
        name="Qwen3 30B A3B (8-bit)",
        description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
            pretty_name="Qwen3 30B A3B (8-bit)",
            storage_size=Memory.from_mb(31738),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-4bit": ModelCard(
        short_id="qwen3-80b-a3B-4bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
        name="Qwen3 80B A3B (4-bit)",
        description="""Qwen3 80B""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
            pretty_name="Qwen3 80B A3B (4-bit)",
            storage_size=Memory.from_mb(44800),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-8bit": ModelCard(
        short_id="qwen3-80b-a3B-8bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
        name="Qwen3 80B A3B (8-bit)",
        description="""Qwen3 80B""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
            pretty_name="Qwen3 80B A3B (8-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-thinking-4bit": ModelCard(
        short_id="qwen3-80b-a3B-thinking-4bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
        name="Qwen3 80B A3B Thinking (4-bit)",
        description="""Qwen3 80B Reasoning model""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
            pretty_name="Qwen3 80B A3B (4-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-80b-a3B-thinking-8bit": ModelCard(
        short_id="qwen3-80b-a3B-thinking-8bit",
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
        name="Qwen3 80B A3B Thinking (8-bit)",
        description="""Qwen3 80B Reasoning model""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
            pretty_name="Qwen3 80B A3B (8-bit)",
            storage_size=Memory.from_mb(84700),
            n_layers=48,
            hidden_size=2048,
            supports_tensor=True,
        ),
    ),
    "qwen3-235b-a22b-4bit": ModelCard(
        short_id="qwen3-235b-a22b-4bit",
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
        name="Qwen3 235B A22B (4-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
            pretty_name="Qwen3 235B A22B (4-bit)",
            storage_size=Memory.from_gb(132),
            n_layers=94,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "qwen3-235b-a22b-8bit": ModelCard(
        short_id="qwen3-235b-a22b-8bit",
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
        name="Qwen3 235B A22B (8-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
            pretty_name="Qwen3 235B A22B (8-bit)",
            storage_size=Memory.from_gb(250),
            n_layers=94,
            hidden_size=4096,
            supports_tensor=True,
        ),
    ),
    "qwen3-coder-480b-a35b-4bit": ModelCard(
        short_id="qwen3-coder-480b-a35b-4bit",
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
        name="Qwen3 Coder 480B A35B (4-bit)",
        description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
            pretty_name="Qwen3 Coder 480B A35B (4-bit)",
            storage_size=Memory.from_gb(270),
            n_layers=62,
            hidden_size=6144,
            supports_tensor=True,
        ),
    ),
    "qwen3-coder-480b-a35b-8bit": ModelCard(
        short_id="qwen3-coder-480b-a35b-8bit",
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
        name="Qwen3 Coder 480B A35B (8-bit)",
        description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
            pretty_name="Qwen3 Coder 480B A35B (8-bit)",
            storage_size=Memory.from_gb(540),
            n_layers=62,
            hidden_size=6144,
            supports_tensor=True,
        ),
    ),
    # gpt-oss
    "gpt-oss-120b-MXFP4-Q8": ModelCard(
        short_id="gpt-oss-120b-MXFP4-Q8",
        model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        name="GPT-OSS 120B (MXFP4-Q8, MLX)",
        description="""OpenAI's GPT-OSS 120B is a 117B-parameter Mixture-of-Experts model designed for high-reasoning and general-purpose use; this variant is a 4-bit MLX conversion for Apple Silicon.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
            pretty_name="GPT-OSS 120B (MXFP4-Q8, MLX)",
            storage_size=Memory.from_kb(68_996_301),
            n_layers=36,
            hidden_size=2880,
            supports_tensor=True,
        ),
    ),
    # "granite-3.3-8b": ModelCard(
    #     short_id="granite-3.3-8b",
    #     model_id=ModelId("mlx-community/granite-3.3-8b-instruct-fp16"),
    #     name="Granite 3.3 8B",
    #     description="""Granite-3.3-8B-Instruct is a 8-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/granite-3.3-8b-instruct-fp16"),
    #         pretty_name="Granite 3.3 8B",
    #         storage_size=Memory.from_kb(15958720),
    #         n_layers=40,
    #     ),
    # ),
    # smol-lm
    # "smol-lm-135m": ModelCard(
    #     short_id="smol-lm-135m",
    #     model_id="mlx-community/SmolLM-135M-4bit",
    #     name="Smol LM 135M",
    #     description="""SmolLM is a series of state-of-the-art small language models available in three sizes: 135M, 360M, and 1.7B parameters. """,
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/SmolLM-135M-4bit"),
    #         pretty_name="Smol LM 135M",
    #         storage_size=Memory.from_kb(73940),
    #         n_layers=30,
    #     ),
    # ),
    # gpt-oss
    # "gpt-oss-120b-MXFP4-Q8": ModelCard(
    #     short_id="gpt-oss-120b-MXFP4-Q8",
    #     model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
    #     name="GPT-OSS 120B (MXFP4-Q8, MLX)",
    #     description="""OpenAI's GPT-OSS 120B is a 117B-parameter Mixture-of-Experts model designed for high-reasoning and general-purpose use; this variant is a 4-bit MLX conversion for Apple Silicon.""",
    #     tags=[],
    #     metadata=ModelMetadata(
    #         model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
    #         pretty_name="GPT-OSS 120B (MXFP4-Q8, MLX)",
    #         storage_size=Memory.from_kb(68_996_301),
    #         n_layers=36,
    #         hidden_size=2880,
    #         supports_tensor=True,
    #     ),
    # ),
    #
    #
}


def get_pretty_name_from_model_id(model_id: str) -> str:
    """Generates a pretty name from a model ID."""
    # Strip namespace
    if "/" in model_id:
        name = model_id.split("/")[-1]
    else:
        name = model_id

    # Handle common replacements
    name = name.replace("-", " ").replace("_", " ")

    # Handle quantization formats like "4bit", "8bit", "Q4_K_M"
    # format "model 4bit" -> model (4-bit)"
    def quant_replacer(match: re.Match[str]) -> str:
        bits = match.group(1)
        return f"({bits}-bit)"
    
    name = re.sub(r"\b(\d+)\s*-?bit\b", quant_replacer, name, flags=re.IGNORECASE)

    # Remove extra spaces
    name = " ".join(name.split())


    # Capitalize words
    words = name.split()
    capitalized_words: list[str] = []
    for word in words:
        # If it's already mixed case or all caps (and not just one letter), keep it
        if (any(c.isupper() for c in word) and len(word) > 1):
            capitalized_words.append(word)
        else:
            # Capitalize the first letter, keep the rest as is
            capitalized_words.append(word[0].upper() + word[1:] if word else "")

    name = " ".join(capitalized_words)
    name = name.replace(" (", "(").replace("(", " (").strip() # Ensure single space before (

    return name


PERSISTENT_FILE_PATH = EXO_CONFIG_HOME / "custom_models.json"
custom_models_loaded = False

def load_custom_models_once() -> None:
    """Loads custom models from persistent storage (called once)"""
    global custom_models_loaded
    if custom_models_loaded:
        return
    custom_models_loaded = True

    logger.debug(f"Attempting to load custom models from: {PERSISTENT_FILE_PATH}")
    
    if not PERSISTENT_FILE_PATH.exists():
        logger.debug(f"Custom models file does not exist yet: {PERSISTENT_FILE_PATH}")
        return
    
    try:
        with open(PERSISTENT_FILE_PATH, "r") as f:
            data = cast(dict[str, dict[str, Any]], json.load(f))
            logger.debug(f"Loaded {len(data)} entries from custom_models.json")
            
            loaded_count = 0
            any_changes = False
            for key, card_data in data.items():
                try:
                    card = ModelCard.model_validate(card_data)
                    # prettify the model_id name
                    if card.name == str(card.model_id) or card.name == key:
                        card.name = get_pretty_name_from_model_id(str(card.model_id))
                        if card.metadata.pretty_name == str(card.model_id):
                             card.metadata.pretty_name = card.name
                        any_changes = True

                    if key not in MODEL_CARDS:
                        MODEL_CARDS[key] = card
                        loaded_count += 1
                        logger.debug(f"Loaded custom model: {key} ({card.model_id})")
                    else:
                        logger.debug(f"Skipping {key} - already in MODEL_CARDS")
                except Exception as e:
                    logger.warning(f"Failed to load model card for {key}: {e}")
            
            if any_changes:
                logger.info("Custom model names were prettified. Saving changes.")
                save_custom_models()
            
            logger.info(f"✓ Loaded {loaded_count} custom models from {PERSISTENT_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading custom models from {PERSISTENT_FILE_PATH}: {e}")

def save_custom_models() -> None:
    """Saves custom models from MODEL_CARDS to persistent storage."""
    logger.debug(f"Attempting to save custom models to: {PERSISTENT_FILE_PATH}")
    
    try:
        PERSISTENT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        custom_models = {
            key: card.model_dump(mode="json") 
            for key, card in MODEL_CARDS.items() 
            if "custom" in card.tags
        }
        
        with open(PERSISTENT_FILE_PATH, "w") as f:
            json.dump(custom_models, f, indent=4)
        
        logger.info(f"✓ Saved {len(custom_models)} custom models to {PERSISTENT_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save custom models to {PERSISTENT_FILE_PATH}: {e}")

def register_custom_model(model_id: ModelId, metadata: ModelMetadata, name: str | None = None, description: str | None = None) -> ModelCard:
    """Registers a custom model in MODEL_CARDS and persists it."""
    load_custom_models_once()
    
    short_id = str(model_id)
    if "/" in short_id:
        short_id = short_id.split("/")[-1]
    
    # Check if model is already registered
    if short_id in MODEL_CARDS:
        return MODEL_CARDS[short_id]
    
    logger.debug(f"Registering new model with short_id: {short_id}")
    
    card = ModelCard(
        short_id=short_id,
        model_id=model_id,
        name=name or get_pretty_name_from_model_id(str(model_id)),
        description=description or f"Custom model from {str(model_id).split('/')[0] if '/' in str(model_id) else 'Hugging Face'}",
        tags=["custom"],
        metadata=metadata,
    )
    MODEL_CARDS[short_id] = card
    logger.info(f"✓ Registered custom model: {short_id} ({model_id})")
    
    # Persist to storage
    save_custom_models()
    return card

def get_model_cards() -> dict[str, ModelCard]:
    """Returns MODEL_CARDS, loading custom models on first access."""
    load_custom_models_once()
    return MODEL_CARDS
