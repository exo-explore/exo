from pydantic import PositiveInt

from exo.shared.types.common import Id
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class ModelId(Id):
    def normalize(self) -> str:
        return self.replace("/", "--")

    def short(self) -> str:
        return self.split("/")[-1]


class ModelCard(CamelCaseModel):
    model_id: ModelId
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    supports_tensor: bool


MODEL_CARDS: dict[str, ModelCard] = {
    # deepseek v3
    "deepseek-v3.1-4bit": ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
        storage_size=Memory.from_gb(378),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
    ),
    "deepseek-v3.1-8bit": ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
        storage_size=Memory.from_gb(713),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
    ),
    # kimi k2
    "kimi-k2-instruct-4bit": ModelCard(
        model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
        storage_size=Memory.from_gb(578),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
    ),
    "kimi-k2-thinking": ModelCard(
        model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
        storage_size=Memory.from_gb(658),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
    ),
    # llama-3.1
    "llama-3.1-8b": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        storage_size=Memory.from_mb(4423),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
    ),
    "llama-3.1-8b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
        storage_size=Memory.from_mb(8540),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
    ),
    "llama-3.1-8b-bf16": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
        storage_size=Memory.from_mb(16100),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
    ),
    "llama-3.1-70b": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
        storage_size=Memory.from_mb(38769),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
    ),
    # llama-3.2
    "llama-3.2-1b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        storage_size=Memory.from_mb(696),
        n_layers=16,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "llama-3.2-3b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        storage_size=Memory.from_mb(1777),
        n_layers=28,
        hidden_size=3072,
        supports_tensor=True,
    ),
    "llama-3.2-3b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
        storage_size=Memory.from_mb(3339),
        n_layers=28,
        hidden_size=3072,
        supports_tensor=True,
    ),
    # llama-3.3
    "llama-3.3-70b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
        storage_size=Memory.from_mb(38769),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
    ),
    "llama-3.3-70b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
        storage_size=Memory.from_mb(73242),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
    ),
    "llama-3.3-70b-fp16": ModelCard(
        model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
        storage_size=Memory.from_mb(137695),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
    ),
    # qwen3
    "qwen3-0.6b": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
        storage_size=Memory.from_mb(327),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
    ),
    "qwen3-0.6b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_mb(666),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
    ),
    "qwen3-30b": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
        storage_size=Memory.from_mb(16797),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-30b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
        storage_size=Memory.from_mb(31738),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-80b-a3B-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
        storage_size=Memory.from_mb(44800),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-80b-a3B-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-80b-a3B-thinking-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-80b-a3B-thinking-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
    ),
    "qwen3-235b-a22b-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
        storage_size=Memory.from_gb(132),
        n_layers=94,
        hidden_size=4096,
        supports_tensor=True,
    ),
    "qwen3-235b-a22b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
        storage_size=Memory.from_gb(250),
        n_layers=94,
        hidden_size=4096,
        supports_tensor=True,
    ),
    "qwen3-coder-480b-a35b-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
        storage_size=Memory.from_gb(270),
        n_layers=62,
        hidden_size=6144,
        supports_tensor=True,
    ),
    "qwen3-coder-480b-a35b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
        storage_size=Memory.from_gb(540),
        n_layers=62,
        hidden_size=6144,
        supports_tensor=True,
    ),
    # gpt-oss
    "gpt-oss-120b-MXFP4-Q8": ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        storage_size=Memory.from_kb(68_996_301),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=True,
    ),
    "gpt-oss-20b-MXFP4-Q8": ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        storage_size=Memory.from_kb(11_744_051),
        n_layers=24,
        hidden_size=2880,
        supports_tensor=True,
    ),
    # glm 4.5
    "glm-4.5-air-8bit": ModelCard(
        # Needs to be quantized g32 or g16 to work with tensor parallel
        model_id=ModelId("mlx-community/GLM-4.5-Air-8bit"),
        storage_size=Memory.from_gb(114),
        n_layers=46,
        hidden_size=4096,
        supports_tensor=False,
    ),
    "glm-4.5-air-bf16": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.5-Air-bf16"),
        storage_size=Memory.from_gb(214),
        n_layers=46,
        hidden_size=4096,
        supports_tensor=True,
    ),
    # glm 4.7
    "glm-4.7-4bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-4bit"),
        storage_size=Memory.from_bytes(198556925568),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
    ),
    "glm-4.7-6bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-6bit"),
        storage_size=Memory.from_bytes(286737579648),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
    ),
    "glm-4.7-8bit-gs32": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-8bit-gs32"),
        storage_size=Memory.from_bytes(396963397248),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
    ),
    # minimax-m2
    "minimax-m2.1-8bit": ModelCard(
        model_id=ModelId("mlx-community/MiniMax-M2.1-8bit"),
        storage_size=Memory.from_bytes(242986745856),
        n_layers=61,
        hidden_size=3072,
        supports_tensor=True,
    ),
    "minimax-m2.1-3bit": ModelCard(
        model_id=ModelId("mlx-community/MiniMax-M2.1-3bit"),
        storage_size=Memory.from_bytes(100086644736),
        n_layers=61,
        hidden_size=3072,
        supports_tensor=True,
    ),
}
