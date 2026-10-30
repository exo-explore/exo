from exo.worker.engines.mlx.models.spec import FamilySpec, PlacementRules

SPEC = FamilySpec(
    name="deepseek_v4",
    model_types=frozenset({"deepseek_v4"}),
    tensor_parallel_architectures=frozenset({"DeepseekV4ForCausalLM"}),
    placement=PlacementRules(
        # DeepSeek V4 is MQA (num_key_value_heads=1) but its sharding strategy
        # head-parallelises wq_b/wo_a and shards MoE experts instead of splitting
        # KV heads, so the kv-head divisibility check doesn't apply.
        tensor_parallel_splits_kv_heads=False,
    ),
)
