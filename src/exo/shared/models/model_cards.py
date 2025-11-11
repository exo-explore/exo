from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    short_id: str
    model_id: str
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata


MODEL_CARDS: dict[str, ModelCard] = {
    # deepseek v3
    "deepseek-v3-0324:4bit": ModelCard(
        short_id="deepseek-v3-0324:4bit",
        model_id="mlx-community/DeepSeek-V3-0324-4bit",
        name="DeepSeek V3 0324 (4-bit)",
        description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3-0324-4bit"),
            pretty_name="DeepSeek V3 0324 (4-bit)",
            storage_size=Memory.from_kb(409706307),
            n_layers=61,
        ),
    ),
    "deepseek-v3-0324": ModelCard(
        short_id="deepseek-v3-0324",
        model_id="mlx-community/DeepSeek-v3-0324-8bit",
        name="DeepSeek V3 0324 (8-bit)",
        description="""DeepSeek V3 is a large language model trained on the DeepSeek V3 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-v3-0324-8bit"),
            pretty_name="DeepSeek V3 0324 (8-bit)",
            storage_size=Memory.from_kb(754706307),
            n_layers=61,
        ),
    ),
    "deepseek-v3.1": ModelCard(
        short_id="deepseek-v3.1",
        model_id="mlx-community/DeepSeek-V3.1-8bit",
        name="DeepSeek V3.1 (8-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
            pretty_name="DeepSeek V3.1 (8-bit)",
            storage_size=Memory.from_kb(754706307),
            n_layers=61,
        ),
    ),
    "deepseek-v3.1:4bit": ModelCard(
        short_id="deepseek-v3.1:4bit",
        model_id="mlx-community/DeepSeek-V3.1-4bit",
        name="DeepSeek V3.1 (4-bit)",
        description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
            pretty_name="DeepSeek V3.1 (4-bit)",
            storage_size=Memory.from_kb(754706307 // 2),  # TODO !!!!!
            n_layers=61,
        ),
    ),
    # deepseek r1
    "deepseek-r1-0528:4bit": ModelCard(
        short_id="deepseek-r1-0528:4bit",
        model_id="mlx-community/DeepSeek-R1-0528-4bit",
        name="DeepSeek-R1-0528 (4-bit)",
        description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-R1-0528-4bit"),
            pretty_name="DeepSeek R1 671B (4-bit)",
            storage_size=Memory.from_kb(409706307),
            n_layers=61,
        ),
    ),
    "deepseek-r1-0528": ModelCard(
        short_id="deepseek-r1-0528",
        model_id="mlx-community/DeepSeek-R1-0528-8bit",
        name="DeepSeek-R1-0528 (8-bit)",
        description="""DeepSeek R1 is a large language model trained on the DeepSeek R1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/DeepSeek-R1-0528-8bit"),
            pretty_name="DeepSeek R1 671B (8-bit)",
            storage_size=Memory.from_bytes(754998771712),
            n_layers=61,
        ),
    ),
    # kimi k2
    "kimi-k2-instruct-4bit": ModelCard(
        short_id="kimi-k2-instruct-4bit",
        model_id="mlx-community/Kimi-K2-Instruct-4bit",
        name="Kimi K2 Instruct (4-bit)",
        description="""Kimi K2 is a large language model trained on the Kimi K2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
            pretty_name="Kimi K2 Instruct (4-bit)",
            storage_size=Memory.from_bytes(577597603840),
            n_layers=61,
        ),
    ),
    "kimi-k2-thinking": ModelCard(
        short_id="kimi-k2-thinking",
        model_id="mlx-community/Kimi-K2-Thinking",
        name="Kimi K2 Thinking",
        description="""Kimi K2 Thinking is the latest, most capable version of open-source thinking model.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
            pretty_name="Kimi K2 Thinking",
            storage_size=Memory.from_bytes(577597603840),
            n_layers=61,
        ),
    ),
    # llama-3.1
    "llama-3.1-8b": ModelCard(
        short_id="llama-3.1-8b",
        model_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        name="Llama 3.1 8B",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
            pretty_name="Llama 3.1 8B",
            storage_size=Memory.from_kb(4411528),
            n_layers=32,
        ),
    ),
    "llama-3.1-70b": ModelCard(
        short_id="llama-3.1-70b",
        model_id="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        name="Llama 3.1 70B",
        description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
            pretty_name="Llama 3.1 70B",
            storage_size=Memory.from_kb(38758160),
            n_layers=80,
        ),
    ),
    # llama-3.2
    "llama-3.2-1b": ModelCard(
        short_id="llama-3.2-1b",
        model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        name="Llama 3.2 1B",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
            pretty_name="Llama 3.2 1B",
            storage_size=Memory.from_kb(678948),
            n_layers=16,
        ),
    ),
    "llama-3.2-3b": ModelCard(
        short_id="llama-3.2-3b",
        model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        name="Llama 3.2 3B",
        description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
            pretty_name="Llama 3.2 3B",
            storage_size=Memory.from_kb(1765062),
            n_layers=28,
        ),
    ),
    # llama-3.3
    "llama-3.3-70b": ModelCard(
        short_id="llama-3.3-70b",
        model_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
        name="Llama 3.3 70B (4-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
            pretty_name="Llama 3.3 70B",
            storage_size=Memory.from_kb(38758160),
            n_layers=80,
        ),
    ),
    "llama-3.3-70b-8bit": ModelCard(
        short_id="llama-3.3-70b-8bit",
        model_id="mlx-community/Llama-3.3-70B-Instruct-8bit",
        name="Llama 3.3 70B (8-bit)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
            pretty_name="Llama 3.3 70B (8-bit)",
            storage_size=Memory.from_kb(77516320),
            n_layers=80,
        ),
    ),
    "llama-3.3-70b-fp16": ModelCard(
        short_id="llama-3.3-70b-fp16",
        model_id="mlx-community/llama-3.3-70b-instruct-fp16",
        name="Llama 3.3 70B (FP16)",
        description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
            pretty_name="Llama 3.3 70B (FP16)",
            storage_size=Memory.from_kb(155032640),
            n_layers=80,
        ),
    ),
    # phi-3
    "phi-3-mini": ModelCard(
        short_id="phi-3-mini",
        model_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
        name="Phi 3 Mini 128k",
        description="""Phi 3 Mini is a large language model trained on the Phi 3 Mini dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Phi-3-mini-128k-instruct-4bit"),
            pretty_name="Phi 3 Mini 128k",
            storage_size=Memory.from_kb(2099262),
            n_layers=32,
        ),
    ),
    "phi-3-mini:128k": ModelCard(
        short_id="phi-3-mini:128k",
        model_id="mlx-community/Phi-3-mini-128k-instruct-4bit",
        name="Phi 3 Mini 128k",
        description="""Phi 3 Mini is a large language model trained on the Phi 3 Mini dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Phi-3-mini-128k-instruct-4bit"),
            pretty_name="Phi 3 Mini 128k",
            storage_size=Memory.from_kb(2099262),
            n_layers=32,
        ),
    ),
    # qwen3
    "qwen3-0.6b": ModelCard(
        short_id="qwen3-0.6b",
        model_id="mlx-community/Qwen3-0.6B-4bit",
        name="Qwen3 0.6B",
        description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
            pretty_name="Qwen3 0.6B",
            storage_size=Memory.from_kb(327512),
            n_layers=28,
        ),
    ),
    "qwen3-30b": ModelCard(
        short_id="qwen3-30b",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
        name="Qwen3 30B (Active 3B)",
        description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
            pretty_name="Qwen3 30B (Active 3B)",
            storage_size=Memory.from_kb(16772092),
            n_layers=48,
        ),
    ),
    "qwen3-235b-a22b": ModelCard(
        short_id="qwen3-235b-a22b",
        model_id="mlx-community/Qwen3-235B-A22B-4bit",
        name="Qwen3 235B, Active 22B (4-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-4bit"),
            pretty_name="Qwen3 235B, Active 22B (4-bit)",
            storage_size=Memory.from_kb(123207680),
            n_layers=94,
        ),
    ),
    "qwen3-235b-a22b-8bit": ModelCard(
        short_id="qwen3-235b-a22b-8bit",
        model_id="mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit",
        name="Qwen3 235B, Active 22B (8-bit)",
        description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
            pretty_name="Qwen3 235B, Active 22B (8-bit)",
            storage_size=Memory.from_kb(246415360),
            n_layers=94,
        ),
    ),
    # granite
    "granite-3.3-2b": ModelCard(
        short_id="granite-3.3-2b",
        model_id="mlx-community/granite-3.3-2b-instruct-fp16",
        name="Granite 3.3 2B",
        description="""Granite-3.3-2B-Instruct is a 2-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/granite-3.3-2b-instruct-fp16"),
            pretty_name="Granite 3.3 2B",
            storage_size=Memory.from_kb(4948320),
            n_layers=40,
        ),
    ),
    "granite-3.3-8b": ModelCard(
        short_id="granite-3.3-8b",
        model_id="mlx-community/granite-3.3-8b-instruct-fp16",
        name="Granite 3.3 8B",
        description="""Granite-3.3-8B-Instruct is a 8-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/granite-3.3-8b-instruct-fp16"),
            pretty_name="Granite 3.3 8B",
            storage_size=Memory.from_kb(15958720),
            n_layers=40,
        ),
    ),
    # smol-lm
    "smol-lm-135m": ModelCard(
        short_id="smol-lm-135m",
        model_id="mlx-community/SmolLM-135M-4bit",
        name="Smol LM 135M",
        description="""SmolLM is a series of state-of-the-art small language models available in three sizes: 135M, 360M, and 1.7B parameters. """,
        tags=[],
        metadata=ModelMetadata(
            model_id=ModelId("mlx-community/SmolLM-135M-4bit"),
            pretty_name="Smol LM 135M",
            storage_size=Memory.from_kb(73940),
            n_layers=30,
        ),
    ),
}
