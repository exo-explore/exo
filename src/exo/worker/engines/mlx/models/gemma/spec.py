from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec, PlacementRules

SPEC = FamilySpec(
    name="gemma",
    model_types=frozenset({"gemma4"}),
    tensor_parallel_architectures=frozenset({"Gemma4ForConditionalGeneration"}),
    placement=PlacementRules(
        multi_node_pipeline_error=(
            "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
        ),
    ),
    eos_rules=(EosRule(("gemma-4", "gemma-3"), (1, 106, 50)),),
)
