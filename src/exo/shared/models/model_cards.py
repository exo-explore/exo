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


# =============================================================================
# MLX MODEL CARDS - Apple Silicon Only (DISABLED for Android-focused builds)
# =============================================================================
# These models require the MLX framework which only works on Apple Silicon Macs.
# For Android/Termux, use GGUF_MODEL_CARDS below instead.
# To re-enable MLX models:
#   1. Uncomment the MODEL_CARDS dict below
#   2. Change ALL_MODEL_CARDS to: {**MODEL_CARDS, **GGUF_MODEL_CARDS}
# =============================================================================

# MODEL_CARDS: dict[str, ModelCard] = {
#     "deepseek-v3.1-4bit": ModelCard(
#         short_id="deepseek-v3.1-4bit",
#         model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
#         name="DeepSeek V3.1 (4-bit)",
#         description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
#             pretty_name="DeepSeek V3.1 (4-bit)",
#             storage_size=Memory.from_gb(378),
#             n_layers=61,
#         ),
#     ),
#     "deepseek-v3.1-8bit": ModelCard(
#         short_id="deepseek-v3.1-8bit",
#         model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
#         name="DeepSeek V3.1 (8-bit)",
#         description="""DeepSeek V3.1 is a large language model trained on the DeepSeek V3.1 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
#             pretty_name="DeepSeek V3.1 (8-bit)",
#             storage_size=Memory.from_gb(713),
#             n_layers=61,
#         ),
#     ),
#     "kimi-k2-instruct-4bit": ModelCard(
#         short_id="kimi-k2-instruct-4bit",
#         model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
#         name="Kimi K2 Instruct (4-bit)",
#         description="""Kimi K2 is a large language model trained on the Kimi K2 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
#             pretty_name="Kimi K2 Instruct (4-bit)",
#             storage_size=Memory.from_gb(578),
#             n_layers=61,
#         ),
#     ),
#     "kimi-k2-thinking": ModelCard(
#         short_id="kimi-k2-thinking",
#         model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
#         name="Kimi K2 Thinking (4-bit)",
#         description="""Kimi K2 Thinking is the latest, most capable version of open-source thinking model.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
#             pretty_name="Kimi K2 Thinking (4-bit)",
#             storage_size=Memory.from_gb(658),
#             n_layers=61,
#         ),
#     ),
#     "llama-3.1-8b": ModelCard(
#         short_id="llama-3.1-8b",
#         model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
#         name="Llama 3.1 8B (4-bit)",
#         description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
#             pretty_name="Llama 3.1 8B (4-bit)",
#             storage_size=Memory.from_mb(4423),
#             n_layers=32,
#         ),
#     ),
#     "llama-3.1-70b": ModelCard(
#         short_id="llama-3.1-70b",
#         model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
#         name="Llama 3.1 70B (4-bit)",
#         description="""Llama 3.1 is a large language model trained on the Llama 3.1 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
#             pretty_name="Llama 3.1 70B (4-bit)",
#             storage_size=Memory.from_mb(38769),
#             n_layers=80,
#         ),
#     ),
#     "llama-3.2-1b": ModelCard(
#         short_id="llama-3.2-1b",
#         model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
#         name="Llama 3.2 1B (4-bit)",
#         description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
#             pretty_name="Llama 3.2 1B (4-bit)",
#             storage_size=Memory.from_mb(696),
#             n_layers=16,
#         ),
#     ),
#     "llama-3.2-3b": ModelCard(
#         short_id="llama-3.2-3b",
#         model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
#         name="Llama 3.2 3B (4-bit)",
#         description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
#             pretty_name="Llama 3.2 3B (4-bit)",
#             storage_size=Memory.from_mb(1777),
#             n_layers=28,
#         ),
#     ),
#     "llama-3.2-3b-8bit": ModelCard(
#         short_id="llama-3.2-3b-8bit",
#         model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
#         name="Llama 3.2 3B (8-bit)",
#         description="""Llama 3.2 is a large language model trained on the Llama 3.2 dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
#             pretty_name="Llama 3.2 3B (8-bit)",
#             storage_size=Memory.from_mb(3339),
#             n_layers=28,
#         ),
#     ),
#     "llama-3.3-70b": ModelCard(
#         short_id="llama-3.3-70b",
#         model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
#         name="Llama 3.3 70B (4-bit)",
#         description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
#             pretty_name="Llama 3.3 70B",
#             storage_size=Memory.from_mb(38769),
#             n_layers=80,
#         ),
#     ),
#     "llama-3.3-70b-8bit": ModelCard(
#         short_id="llama-3.3-70b-8bit",
#         model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
#         name="Llama 3.3 70B (8-bit)",
#         description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
#             pretty_name="Llama 3.3 70B (8-bit)",
#             storage_size=Memory.from_mb(73242),
#             n_layers=80,
#         ),
#     ),
#     "llama-3.3-70b-fp16": ModelCard(
#         short_id="llama-3.3-70b-fp16",
#         model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
#         name="Llama 3.3 70B (FP16)",
#         description="""The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out)""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
#             pretty_name="Llama 3.3 70B (FP16)",
#             storage_size=Memory.from_mb(137695),
#             n_layers=80,
#         ),
#     ),
#     "phi-3-mini": ModelCard(
#         short_id="phi-3-mini",
#         model_id=ModelId("mlx-community/Phi-3-mini-128k-instruct-4bit"),
#         name="Phi 3 Mini 128k (4-bit)",
#         description="""Phi 3 Mini is a large language model trained on the Phi 3 Mini dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Phi-3-mini-128k-instruct-4bit"),
#             pretty_name="Phi 3 Mini 128k (4-bit)",
#             storage_size=Memory.from_mb(2099),
#             n_layers=32,
#         ),
#     ),
#     "qwen3-0.6b": ModelCard(
#         short_id="qwen3-0.6b",
#         model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
#         name="Qwen3 0.6B (4-bit)",
#         description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
#             pretty_name="Qwen3 0.6B (4-bit)",
#             storage_size=Memory.from_mb(327),
#             n_layers=28,
#         ),
#     ),
#     "qwen3-0.6b-8bit": ModelCard(
#         short_id="qwen3-0.6b-8bit",
#         model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
#         name="Qwen3 0.6B (8-bit)",
#         description="""Qwen3 0.6B is a large language model trained on the Qwen3 0.6B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
#             pretty_name="Qwen3 0.6B (8-bit)",
#             storage_size=Memory.from_mb(666),
#             n_layers=28,
#         ),
#     ),
#     "qwen3-30b": ModelCard(
#         short_id="qwen3-30b",
#         model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
#         name="Qwen3 30B A3B (4-bit)",
#         description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
#             pretty_name="Qwen3 30B A3B (4-bit)",
#             storage_size=Memory.from_mb(16797),
#             n_layers=48,
#         ),
#     ),
#     "qwen3-30b-8bit": ModelCard(
#         short_id="qwen3-30b-8bit",
#         model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
#         name="Qwen3 30B A3B (8-bit)",
#         description="""Qwen3 30B is a large language model trained on the Qwen3 30B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
#             pretty_name="Qwen3 30B A3B (8-bit)",
#             storage_size=Memory.from_mb(31738),
#             n_layers=48,
#         ),
#     ),
#     "qwen3-235b-a22b-4bit": ModelCard(
#         short_id="qwen3-235b-a22b-4bit",
#         model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
#         name="Qwen3 235B A22B (4-bit)",
#         description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
#             pretty_name="Qwen3 235B A22B (4-bit)",
#             storage_size=Memory.from_gb(132),
#             n_layers=94,
#         ),
#     ),
#     "qwen3-235b-a22b-8bit": ModelCard(
#         short_id="qwen3-235b-a22b-8bit",
#         model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
#         name="Qwen3 235B A22B (8-bit)",
#         description="""Qwen3 235B (Active 22B) is a large language model trained on the Qwen3 235B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
#             pretty_name="Qwen3 235B A22B (8-bit)",
#             storage_size=Memory.from_gb(250),
#             n_layers=94,
#         ),
#     ),
#     "qwen3-coder-480b-a35b-4bit": ModelCard(
#         short_id="qwen3-coder-480b-a35b-4bit",
#         model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
#         name="Qwen3 Coder 480B A35B (4-bit)",
#         description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
#             pretty_name="Qwen3 Coder 480B A35B (4-bit)",
#             storage_size=Memory.from_gb(270),
#             n_layers=62,
#         ),
#     ),
#     "qwen3-coder-480b-a35b-8bit": ModelCard(
#         short_id="qwen3-coder-480b-a35b-8bit",
#         model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
#         name="Qwen3 Coder 480B A35B (8-bit)",
#         description="""Qwen3 Coder 480B (Active 35B) is a large language model trained on the Qwen3 Coder 480B dataset.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
#             pretty_name="Qwen3 Coder 480B A35B (8-bit)",
#             storage_size=Memory.from_gb(540),
#             n_layers=62,
#         ),
#     ),
#     "granite-3.3-2b": ModelCard(
#         short_id="granite-3.3-2b",
#         model_id=ModelId("mlx-community/granite-3.3-2b-instruct-fp16"),
#         name="Granite 3.3 2B (FP16)",
#         description="""Granite-3.3-2B-Instruct is a 2-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities.""",
#         tags=[],
#         metadata=ModelMetadata(
#             model_id=ModelId("mlx-community/granite-3.3-2b-instruct-fp16"),
#             pretty_name="Granite 3.3 2B (FP16)",
#             storage_size=Memory.from_mb(4951),
#             n_layers=40,
#         ),
#     ),
# }

# Empty MODEL_CARDS for Android-only builds (MLX is disabled)
MODEL_CARDS: dict[str, ModelCard] = {}


# =============================================================================
# GGUF MODEL CARDS - llama.cpp backend (Android, Linux, cross-platform)
# =============================================================================
# These models work with llama.cpp and are the primary format for Android/Termux.
# =============================================================================

GGUF_MODEL_CARDS: dict[str, ModelCard] = {
    # Llama 3.2 GGUF models (good for mobile/Android)
    "llama-3.2-1b-gguf": ModelCard(
        short_id="llama-3.2-1b-gguf",
        model_id=ModelId("bartowski/Llama-3.2-1B-Instruct-GGUF"),
        name="Llama 3.2 1B (GGUF Q4_K_M)",
        description="""Llama 3.2 1B in GGUF format for llama.cpp. Optimized for mobile and edge devices.""",
        tags=["gguf", "mobile", "edge"],
        metadata=ModelMetadata(
            model_id=ModelId("bartowski/Llama-3.2-1B-Instruct-GGUF"),
            pretty_name="Llama 3.2 1B (GGUF)",
            storage_size=Memory.from_mb(750),
            n_layers=16,
        ),
    ),
    "llama-3.2-3b-gguf": ModelCard(
        short_id="llama-3.2-3b-gguf",
        model_id=ModelId("bartowski/Llama-3.2-3B-Instruct-GGUF"),
        name="Llama 3.2 3B (GGUF Q4_K_M)",
        description="""Llama 3.2 3B in GGUF format for llama.cpp. Good balance of size and capability.""",
        tags=["gguf", "mobile"],
        metadata=ModelMetadata(
            model_id=ModelId("bartowski/Llama-3.2-3B-Instruct-GGUF"),
            pretty_name="Llama 3.2 3B (GGUF)",
            storage_size=Memory.from_mb(2000),
            n_layers=28,
        ),
    ),
    # Qwen2.5 GGUF models
    "qwen2.5-0.5b-gguf": ModelCard(
        short_id="qwen2.5-0.5b-gguf",
        model_id=ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        name="Qwen 2.5 0.5B (GGUF Q4_K_M)",
        description="""Qwen 2.5 0.5B - ultra-lightweight model for constrained devices.""",
        tags=["gguf", "mobile", "edge", "tiny"],
        metadata=ModelMetadata(
            model_id=ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            pretty_name="Qwen 2.5 0.5B (GGUF)",
            storage_size=Memory.from_mb(400),
            n_layers=24,
        ),
    ),
    "qwen2.5-1.5b-gguf": ModelCard(
        short_id="qwen2.5-1.5b-gguf",
        model_id=ModelId("Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
        name="Qwen 2.5 1.5B (GGUF Q4_K_M)",
        description="""Qwen 2.5 1.5B - efficient model for mobile inference.""",
        tags=["gguf", "mobile"],
        metadata=ModelMetadata(
            model_id=ModelId("Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
            pretty_name="Qwen 2.5 1.5B (GGUF)",
            storage_size=Memory.from_mb(1000),
            n_layers=28,
        ),
    ),
    "qwen2.5-3b-gguf": ModelCard(
        short_id="qwen2.5-3b-gguf",
        model_id=ModelId("Qwen/Qwen2.5-3B-Instruct-GGUF"),
        name="Qwen 2.5 3B (GGUF Q4_K_M)",
        description="""Qwen 2.5 3B - capable model for mobile and edge devices.""",
        tags=["gguf", "mobile"],
        metadata=ModelMetadata(
            model_id=ModelId("Qwen/Qwen2.5-3B-Instruct-GGUF"),
            pretty_name="Qwen 2.5 3B (GGUF)",
            storage_size=Memory.from_mb(2000),
            n_layers=36,
        ),
    ),
    # Phi-3 GGUF models (Microsoft's efficient models)
    "phi-3-mini-gguf": ModelCard(
        short_id="phi-3-mini-gguf",
        model_id=ModelId("bartowski/Phi-3.5-mini-instruct-GGUF"),
        name="Phi 3.5 Mini (GGUF Q4_K_M)",
        description="""Microsoft Phi 3.5 Mini - efficient 3.8B model with strong reasoning.""",
        tags=["gguf", "mobile", "reasoning"],
        metadata=ModelMetadata(
            model_id=ModelId("bartowski/Phi-3.5-mini-instruct-GGUF"),
            pretty_name="Phi 3.5 Mini (GGUF)",
            storage_size=Memory.from_mb(2300),
            n_layers=32,
        ),
    ),
    # TinyLlama for extremely constrained devices
    "tinyllama-1.1b-gguf": ModelCard(
        short_id="tinyllama-1.1b-gguf",
        model_id=ModelId("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
        name="TinyLlama 1.1B (GGUF Q4_K_M)",
        description="""TinyLlama 1.1B - extremely efficient model for edge devices.""",
        tags=["gguf", "mobile", "edge", "tiny"],
        metadata=ModelMetadata(
            model_id=ModelId("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            pretty_name="TinyLlama 1.1B (GGUF)",
            storage_size=Memory.from_mb(700),
            n_layers=22,
        ),
    ),
    # AutoGLM-Phone vision model for mobile agents
    "autoglm-phone-9b-gguf": ModelCard(
        short_id="autoglm-phone-9b-gguf",
        model_id=ModelId("mradermacher/AutoGLM-Phone-9B-Multilingual-i1-GGUF"),
        name="AutoGLM Phone 9B (GGUF Q4_K_M)",
        description="""AutoGLM Phone 9B - multilingual vision model for mobile phone agents.""",
        tags=["gguf", "mobile", "vision", "agent", "multilingual"],
        metadata=ModelMetadata(
            model_id=ModelId("mradermacher/AutoGLM-Phone-9B-Multilingual-i1-GGUF"),
            pretty_name="AutoGLM Phone 9B (GGUF)",
            storage_size=Memory.from_mb(6170),
            n_layers=40,
        ),
    ),
}

# Combined model cards - currently GGUF only for Android focus
# To include MLX models: uncomment MODEL_CARDS above and change to:
#   ALL_MODEL_CARDS: dict[str, ModelCard] = {**MODEL_CARDS, **GGUF_MODEL_CARDS}
ALL_MODEL_CARDS: dict[str, ModelCard] = {**MODEL_CARDS, **GGUF_MODEL_CARDS}
