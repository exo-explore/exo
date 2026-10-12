"""Parallel DeepSeek V4 MoE router kernel.

mlx_lm's fused gate kernel runs one GPU thread per token: that thread computes
sqrtsoftplus for all routed experts and then a serial top-k scan. At decode
(one token) the whole router is a single thread, which costs ~50us per layer.

This kernel uses one threadgroup per token and one thread per expert: every
thread scores its own expert, then top-k is TOP_K rounds of a threadgroup
argmax. Selection, weights and renormalization match the original kernel;
indices come out in descending score order (ties to the lower expert index).
"""

import mlx.core as mx
from mlx_lm.models import deepseek_v4
from mlx_lm.models.deepseek_v4 import MoEGate

_THREADS_PER_SIMDGROUP = 32

_SOURCE = """
    uint token = threadgroup_position_in_grid.x;
    uint expert = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup = simdgroup_index_in_threadgroup;
    constexpr uint NUM_SIMDGROUPS = N_ROUTED / 32;

    threadgroup float group_values[NUM_SIMDGROUPS];
    threadgroup int group_indices[NUM_SIMDGROUPS];
    threadgroup int winners[TOP_K];
    threadgroup float winner_activations[TOP_K];

    float v = static_cast<float>(scores[token * N_ROUTED + expert]);
    float sp = (v > 20.0f) ? v : metal::fast::log(1.0f + metal::fast::exp(v));
    float activated = metal::sqrt(sp);
    float candidate = activated + static_cast<float>(bias[expert]);

    for (int k = 0; k < TOP_K; ++k) {
        float best = simd_max(candidate);
        int best_index = simd_min(candidate == best ? int(expert) : int(N_ROUTED));
        if (lane == 0) {
            group_values[simdgroup] = best;
            group_indices[simdgroup] = best_index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            float value = lane < NUM_SIMDGROUPS ? group_values[lane] : -INFINITY;
            int index = lane < NUM_SIMDGROUPS ? group_indices[lane] : int(N_ROUTED);
            float overall = simd_max(value);
            int overall_index = simd_min(value == overall ? index : int(N_ROUTED));
            if (lane == 0) {
                winners[k] = overall_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(expert) == winners[k]) {
            winner_activations[k] = activated;
            candidate = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (expert == 0) {
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            sum += winner_activations[k];
        }
        float scale_factor = route_scale[0] / (sum + 1e-20f);
        for (int k = 0; k < TOP_K; ++k) {
            weights[token * TOP_K + k] = static_cast<OUT_T>(winner_activations[k] * scale_factor);
            inds[token * TOP_K + k] = winners[k];
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="dsv4_moe_gate_parallel",
    input_names=["scores", "bias", "route_scale"],
    output_names=["inds", "weights"],
    source=_SOURCE,
)

_original_call = MoEGate.__call__


def _supports_parallel_gate(gate: MoEGate) -> bool:
    return (
        deepseek_v4._moe_gate_kernel is not None  # pyright: ignore[reportPrivateUsage]
        and not gate.hash
        and gate.score_func == "sqrtsoftplus"
        and gate.norm_topk_prob
        and gate.n_routed % _THREADS_PER_SIMDGROUP == 0
        and gate.n_routed <= 1024
        and gate.top_k <= gate.n_routed
    )


def _patched_call(
    self: MoEGate, x: mx.array, input_ids: mx.array | None = None
) -> tuple[mx.array, mx.array]:
    if not _supports_parallel_gate(self):
        return _original_call(self, x, input_ids)
    scores = x @ self.weight.T
    batch, sequence, _ = x.shape
    total = batch * sequence
    indices, weights = _kernel(
        inputs=[scores, self.e_score_correction_bias, self._route_scale_arr],
        template=[
            ("N_ROUTED", self.n_routed),
            ("TOP_K", self.top_k),
            ("OUT_T", x.dtype),
        ],
        grid=(total * self.n_routed, 1, 1),
        threadgroup=(self.n_routed, 1, 1),
        output_shapes=[(batch, sequence, self.top_k), (batch, sequence, self.top_k)],
        output_dtypes=[mx.int32, x.dtype],
    )
    return indices, weights


def patch_deepseek_v4_moe_gate() -> None:
    MoEGate.__call__ = _patched_call
