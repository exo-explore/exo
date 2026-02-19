from . import ArchitectureSpec, register

QWEN_DENSE_SPEC = ArchitectureSpec(
    name = "llama",
    attention_type = "grouped_query",
    mlp_type = "swiglu",
    norm_type = "rms_norm",
    rope_type = "standard",
    layer_prefix = "model.layers.{layer_idx}",
    q_proj_key = "self_attn.q_proj",
    k_proj_key = "self_attn.k_proj",
    v_proj_key = "self_attn.v_proj",
    o_proj_key = "self_attn.o_proj",
    gate_proj_key = "mlp.gate_proj",
    up_proj_key = "mlp.up_proj",
    down_proj_key = "mlp.down_proj",
    input_norm_key = "input_layernorm",
    post_attn_norm_key = "post_attention_layernorm",
    embed_key = "model.embed_tokens",
    final_norm_key = "model.norm",
    lm_head_key = "lm_head",
)

register("Qwen2ForCausalLM", QWEN_DENSE_SPEC)
register("Qwen3ForCausalLM", QWEN_DENSE_SPEC)
