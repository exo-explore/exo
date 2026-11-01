"""A tiny Nemotron-H configuration shared by the family's tests."""

from typing import Any

# A tiny Nemotron-H with every block type exo shards: Mamba-2, attention, LatentMoE
# (as in Nemotron 3 Super/Ultra and 3.5 Super-VL) and a dense MLP block.
CONFIG: dict[str, Any] = {
    "model_type": "nemotron_h",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 64,
    "max_position_embeddings": 4096,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "attention_bias": False,
    "mamba_num_heads": 8,
    "mamba_head_dim": 16,
    "mamba_proj_bias": False,
    "ssm_state_size": 16,
    "conv_kernel": 4,
    "n_groups": 2,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": True,
    "layers_block_type": ["mamba", "attention", "moe", "mamba", "mlp"],
    # 3 quantization groups of 32: two ranks can't split them evenly (as with Nemotron 3
    # Nano's and 3.5 Lightning's 29 groups of 64).
    "moe_intermediate_size": 96,
    "moe_shared_expert_intermediate_size": 64,
    "moe_latent_size": 64,
    "n_group": 1,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
}
PROMPT = [[1, 23, 45, 6, 17, 38, 52, 9, 11, 60, 2, 33]]
