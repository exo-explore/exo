from collections.abc import Generator
from typing import Any, Final

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import ExpertShardMetadata

try:
    import mlx.core as mx
    from mlx import nn
except ImportError:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

DEFAULT_TOP_K_EXPERTS: Final[int] = 2
DEFAULT_TOTAL_EXPERTS: Final[int] = 8


class ExpertRoutingDecision(BaseModel):
    """Immutable record of the sparse routing decision for a single token."""

    model_config = ConfigDict(frozen=True, strict=True)

    token_index: int
    selected_expert_indices: list[int]
    routing_weights: list[float]
    local_expert_indices: list[int]
    remote_expert_indices: list[int]


class ExpertParallelConfiguration(BaseModel):
    """Configuration settings for Mixture of Experts (MoE) parallelism."""

    model_config = ConfigDict(frozen=True, strict=True)

    total_experts: int = Field(ge=1, default=DEFAULT_TOTAL_EXPERTS)
    top_k_experts: int = Field(ge=1, default=DEFAULT_TOP_K_EXPERTS)
    hosted_expert_indices: list[int] = Field(default_factory=list)
    normalize_routing_weights: bool = True


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable Softmax over an array of logits."""
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exponential_values = np.exp(shifted_logits)
    sum_exponential_values = np.sum(exponential_values, axis=-1, keepdims=True)
    return exponential_values / (sum_exponential_values + 1e-9)


def compute_sparse_routing(
    gating_logits: np.ndarray,
    top_k: int,
    hosted_expert_indices: list[int],
    normalize_weights: bool = True,
) -> list[ExpertRoutingDecision]:
    """Compute top-k sparse gating decisions and partition into local and remote experts."""
    decisions: list[ExpertRoutingDecision] = []
    batch_size = gating_logits.shape[0]

    for token_index in range(batch_size):
        logits = gating_logits[token_index]
        # Find top-k expert indices
        candidate_indices = np.argsort(logits)[-top_k:][::-1].tolist()
        selected_logits = logits[candidate_indices]

        if normalize_weights:
            weights = softmax_stable(selected_logits).tolist()
        else:
            weights = selected_logits.tolist()

        local_indices: list[int] = []
        remote_indices: list[int] = []

        for expert_index in candidate_indices:
            if expert_index in hosted_expert_indices:
                local_indices.append(expert_index)
            else:
                remote_indices.append(expert_index)

        decisions.append(
            ExpertRoutingDecision(
                token_index=token_index,
                selected_expert_indices=candidate_indices,
                routing_weights=weights,
                local_expert_indices=local_indices,
                remote_expert_indices=remote_indices,
            )
        )

    return decisions


def combine_expert_contributions(
    routing_decision: ExpertRoutingDecision,
    expert_outputs: dict[int, np.ndarray],
    feature_dimension: int,
) -> np.ndarray:
    """Perform gating weighted accumulation of expert output representations."""
    aggregated_output = np.zeros(feature_dimension, dtype=np.float32)

    for expert_index, weight in zip(
        routing_decision.selected_expert_indices,
        routing_decision.routing_weights,
        strict=True,
    ):
        if expert_index in expert_outputs:
            output_slice = expert_outputs[expert_index]
            aggregated_output += float(weight) * output_slice.astype(np.float32)

    return aggregated_output


class SparseExpertRouter:
    """Evaluates router gate projections to route tokens across distributed experts."""

    def __init__(
        self,
        gate_weight: np.ndarray,
        configuration: ExpertParallelConfiguration,
    ) -> None:
        self.gate_weight = gate_weight
        self.configuration = configuration

    def route_tokens(
        self,
        hidden_states: np.ndarray,
    ) -> list[ExpertRoutingDecision]:
        """Compute routing decisions for a batch of input token vectors."""
        # hidden_states: (batch_size, hidden_dimension)
        # gate_weight: (hidden_dimension, total_experts)
        gating_logits = np.matmul(hidden_states, self.gate_weight)
        return compute_sparse_routing(
            gating_logits=gating_logits,
            top_k=self.configuration.top_k_experts,
            hosted_expert_indices=self.configuration.hosted_expert_indices,
            normalize_weights=self.configuration.normalize_routing_weights,
        )


class ShardedExpertBlock:
    """Manages execution of locally hosted experts in a Mixture of Experts layer."""

    def __init__(
        self,
        layer_index: int,
        configuration: ExpertParallelConfiguration,
        local_expert_callables: dict[int, Any],
    ) -> None:
        self.layer_index = layer_index
        self.configuration = configuration
        self.local_experts = local_expert_callables
        logger.info(
            f"Initialized ShardedExpertBlock for layer {layer_index} with "
            f"{len(self.local_experts)} / {configuration.total_experts} hosted experts "
            f"({list(self.local_experts.keys())})"
        )

    def execute_local_experts(
        self,
        token_vector: np.ndarray,
        required_local_indices: list[int],
    ) -> dict[int, np.ndarray]:
        """Execute all locally assigned experts for the given token vector."""
        results: dict[int, np.ndarray] = {}
        for expert_index in required_local_indices:
            if expert_index in self.local_experts:
                expert_callable = self.local_experts[expert_index]
                if mx is not None and isinstance(token_vector, mx.array):
                    output = expert_callable(token_vector)
                    results[expert_index] = np.asarray(output)
                else:
                    output = expert_callable(token_vector)
                    results[expert_index] = (
                        np.asarray(output)
                        if not isinstance(output, np.ndarray)
                        else output
                    )
        return results


class ExpertTokenDispatcher:
    """Orchestrates token dispatch across local and remote cluster expert nodes."""

    def __init__(
        self,
        router: SparseExpertRouter,
        sharded_block: ShardedExpertBlock,
    ) -> None:
        self.router = router
        self.sharded_block = sharded_block

    def dispatch_and_forward(
        self,
        input_tokens: np.ndarray,
        remote_expert_handler: Any | None = None,
    ) -> np.ndarray:
        """Forward a batch of token representations through local and remote experts."""
        batch_size, feature_dimension = input_tokens.shape
        routing_decisions = self.router.route_tokens(input_tokens)
        output_matrix = np.zeros((batch_size, feature_dimension), dtype=np.float32)

        for token_index, decision in enumerate(routing_decisions):
            token_vector = input_tokens[token_index]
            expert_outputs: dict[int, np.ndarray] = {}

            # Execute local experts
            if len(decision.local_expert_indices) > 0:
                local_results = self.sharded_block.execute_local_experts(
                    token_vector=token_vector,
                    required_local_indices=decision.local_expert_indices,
                )
                expert_outputs.update(local_results)

            # Execute remote experts if handler is provided
            if len(decision.remote_expert_indices) > 0 and remote_expert_handler is not None:
                remote_results = remote_expert_handler(
                    token_index=token_index,
                    token_vector=token_vector,
                    remote_indices=decision.remote_expert_indices,
                )
                expert_outputs.update(remote_results)

            # Combine weighted outputs
            combined_vector = combine_expert_contributions(
                routing_decision=decision,
                expert_outputs=expert_outputs,
                feature_dimension=feature_dimension,
            )
            output_matrix[token_index] = combined_vector

        return output_matrix


def expert_auto_parallel(
    model: Any,
    group: Any,
    shard_metadata: ExpertShardMetadata,
) -> Generator[ModelLoadingResponse, None, Any]:
    """
    Parallelize a Mixture of Experts (MoE) model across devices.
    Only experts allocated in shard_metadata.expert_indices are retained in memory.
    """
    inner_model: Any = getattr(model, "model", model)
    layers: list[Any] = getattr(inner_model, "layers", getattr(inner_model, "h", []))
    total_layers = len(layers)

    for layer_index, layer in enumerate(layers):
        moe_block = None
        if hasattr(layer, "block_sparse_moe"):
            moe_block = layer.block_sparse_moe
        elif hasattr(layer, "moe"):
            moe_block = layer.moe
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_block = layer.mlp.experts

        if moe_block is not None and hasattr(moe_block, "experts"):
            experts_collection = moe_block.experts
            if isinstance(experts_collection, list):
                for expert_index in range(len(experts_collection)):
                    if not shard_metadata.is_expert_assigned(expert_index):
                        experts_collection[expert_index] = None

        if mx is not None and hasattr(mx, "eval"):
            mx.eval(layer)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

        yield ModelLoadingResponse(layers_loaded=layer_index + 1, total=total_layers)

    return model

