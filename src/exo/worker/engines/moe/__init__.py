"""MoE execution engines with elasticity and fault-tolerance."""

from exo.worker.engines.moe.elastic_fallback_executor import ElasticMoEFallbackExecutor

__all__ = [
    "ElasticMoEFallbackExecutor",
]
