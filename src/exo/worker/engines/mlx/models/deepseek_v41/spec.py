from exo.worker.engines.mlx.models.spec import FamilySpec, PlacementRules

SPEC = FamilySpec(
    name="deepseek_v41",
    model_types=frozenset({"deepseek_v41"}),
    tensor_parallel_architectures=frozenset({"DeepseekV41ForCausalLM"}),
    placement=PlacementRules(
        # One KV head (MQA over a latent): tensor parallelism splits the attention heads by
        # output group and every expert's intermediate dimension, not the KV heads.
        tensor_parallel_splits_kv_heads=False,
        # A rank keeps whole output groups (o_groups = 8).
        tensor_parallel_divides=(8,),
    ),
    # With fast synch, a tensor-parallel prefill over RDMA can stop making progress on
    # prompts of over about 1k tokens (where the sparse top-k selection starts).
    fast_synch=False,
)
