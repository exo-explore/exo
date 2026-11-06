import math

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.common import NodeId
from exo.shared.types.moe_routing import (
    MoERoutingDecision,
    MoERoutingTelemetry,
    NodeComputeProfile,
)

DEFAULT_ALPHA: float = 0.5  # Queue penalty coefficient
DEFAULT_BETA: float = 0.5  # Compute weight reward coefficient
DEFAULT_GAMMA: float = 1.0  # Thermal throttling penalty coefficient
QUEUE_NORMALIZATION_BASE: float = 10.0
EPSILON: float = 1e-8


class DeviceAwareMoERouter:
    """MoE Gating Router dynamically balancing expert activations across heterogeneous edge devices."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        capacity_factor: float = 1.25,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.capacity_factor = capacity_factor

    def route_tokens(
        self,
        gating_logits: NDArray[np.floating],
        node_profiles: dict[NodeId, NodeComputeProfile],
        expert_placements: dict[int, NodeId],
        top_k: int = 2,
    ) -> tuple[MoERoutingDecision, MoERoutingTelemetry]:
        """Routes a batch of tokens to experts considering device compute, queues, and thermals.

        Args:
            gating_logits: Matrix of shape (num_tokens, num_experts).
            node_profiles: Mapping of NodeId to its dynamic hardware profile.
            expert_placements: Mapping of expert index (0..E-1) to host NodeId.
            top_k: Number of experts activated per token.

        Returns:
            Tuple of (MoERoutingDecision, MoERoutingTelemetry).
        """
        logits = np.asarray(gating_logits, dtype=np.float32)
        num_tokens, num_experts = logits.shape
        top_k = min(top_k, num_experts)

        # 1. Compute per-expert device bias
        bias = np.zeros(num_experts, dtype=np.float32)
        for expert_id, node_id in expert_placements.items():
            if 0 <= expert_id < num_experts and node_id in node_profiles:
                profile = node_profiles[node_id]
                queue_penalty = self.alpha * (profile.queue_depth / QUEUE_NORMALIZATION_BASE)
                compute_bonus = self.beta * (profile.compute_weight - 1.0)
                throttle_penalty = self.gamma if profile.is_throttled else 0.0
                bias[expert_id] = compute_bonus - queue_penalty - throttle_penalty

        # 2. Adjust logits and compute Softmax probabilities
        adjusted_logits = logits + bias.reshape(1, -1)
        exp_logits = np.exp(adjusted_logits - np.max(adjusted_logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + EPSILON)

        # 3. Dynamic node capacity calculation
        num_nodes = max(1, len(node_profiles))
        node_capacities: dict[NodeId, int] = {}
        for nid, prof in node_profiles.items():
            scaled_cap = math.ceil(
                (num_tokens * top_k / num_nodes) * self.capacity_factor * max(0.2, prof.compute_weight)
            )
            node_capacities[nid] = min(prof.max_capacity, max(1, scaled_cap))

        # 4. Dispatch with capacity constraint & overflow fallback
        selected_experts: list[list[int]] = []
        routing_weights: list[list[float]] = []
        overflow_tokens: list[int] = []

        node_token_counts: dict[str, int] = {str(nid): 0 for nid in node_profiles}
        expert_token_counts: dict[int, int] = {e: 0 for e in range(num_experts)}

        # Sort all experts by descending probability for each token
        ranked_experts_per_token = np.argsort(-probs, axis=-1)

        for tok_idx in range(num_tokens):
            tok_selected: list[int] = []
            tok_weights: list[float] = []
            has_overflow = False

            for candidate_expert in ranked_experts_per_token[tok_idx]:
                if len(tok_selected) >= top_k:
                    break

                target_node = expert_placements.get(int(candidate_expert))
                if target_node is None:
                    continue

                curr_node_load = node_token_counts.get(str(target_node), 0)
                cap = node_capacities.get(target_node, num_tokens)

                if curr_node_load < cap:
                    tok_selected.append(int(candidate_expert))
                    tok_weights.append(float(probs[tok_idx, candidate_expert]))
                    node_token_counts[str(target_node)] = curr_node_load + 1
                    expert_token_counts[int(candidate_expert)] += 1
                else:
                    has_overflow = True

            if has_overflow and len(tok_selected) < top_k:
                overflow_tokens.append(tok_idx)

            # Re-normalize selected weights to sum to 1.0
            sum_w = sum(tok_weights)
            if sum_w > 0:
                normalized_w = [w / sum_w for w in tok_weights]
            else:
                normalized_w = [1.0 / len(tok_selected)] * len(tok_selected) if tok_selected else []

            selected_experts.append(tok_selected)
            routing_weights.append(normalized_w)

        # 5. Build Telemetry
        node_loads = list(node_token_counts.values())
        mean_load = float(np.mean(node_loads)) if node_loads else 0.0
        std_load = float(np.std(node_loads)) if node_loads else 0.0
        load_cv = (std_load / mean_load) if mean_load > 0 else 0.0

        decision = MoERoutingDecision(
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            overflow_tokens=overflow_tokens,
            expert_token_counts=expert_token_counts,
            node_token_counts=node_token_counts,
        )

        telemetry = MoERoutingTelemetry(
            total_tokens=num_tokens,
            num_experts=num_experts,
            overflow_count=len(overflow_tokens),
            overflow_rate=len(overflow_tokens) / max(1, num_tokens),
            load_balance_cv=load_cv,
        )

        return decision, telemetry
