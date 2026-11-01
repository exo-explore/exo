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
        # The strategy gives each rank the same share of every output group's heads
        # (DeepseekV4ShardingStrategy). Flash (64 heads in 8 groups) and Pro (128 heads in
        # 16 groups) both have 8 heads per group, so the node count must divide 8. Without
        # this, V4-Pro passed placement on 7 nodes (7168 = 7 x 1024) and failed to shard.
        tensor_parallel_divides=(8,),
    ),
)
