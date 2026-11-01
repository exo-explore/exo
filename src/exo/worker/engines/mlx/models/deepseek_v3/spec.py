from exo.worker.engines.mlx.models.spec import FamilySpec, PlacementRules

SPEC = FamilySpec(
    name="deepseek_v3",
    model_types=frozenset({"deepseek_v3", "deepseek_v32"}),
    tensor_parallel_architectures=frozenset(
        {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}
    ),
    placement=PlacementRules(
        pipeline_unsupported_model_ids=(
            (
                "mlx-community/DeepSeek-V3.1-8bit",
                "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)",
            ),
        ),
    ),
)
