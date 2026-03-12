import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch._ops import OpOverload as OpOverload
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_aiter_sparse_attn_indexer as rocm_aiter_sparse_attn_indexer,
    rocm_aiter_sparse_attn_indexer_fake as rocm_aiter_sparse_attn_indexer_fake,
)

FP8_DTYPE: Incomplete

def is_aiter_found() -> bool: ...

IS_AITER_FOUND: Incomplete

def is_aiter_found_and_supported() -> bool: ...
def if_aiter_supported(func: Callable) -> Callable: ...

class rocm_aiter_ops:
    @classmethod
    def refresh_env_variables(cls) -> None: ...
    @staticmethod
    def get_aiter_activation_type(activation_str: str): ...
    @staticmethod
    def get_aiter_quant_type(quant_type_str: str): ...
    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_linear_fp8_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_rmsnorm_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_fused_moe_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_fusion_moe_shared_experts_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_mla_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_mha_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_shuffle_kv_cache_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_triton_unified_attn_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_fp8bmm_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_fp4bmm_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_asm_fp4_gemm_dynamic_quant_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_triton_rotary_embed_enabled(cls) -> bool: ...
    @classmethod
    @if_aiter_supported
    def is_triton_gemm_enabled(cls) -> bool: ...
    @staticmethod
    @if_aiter_supported
    def register_ops_once() -> None: ...
    @staticmethod
    def get_rmsnorm_fused_add_op() -> OpOverload: ...
    @staticmethod
    def get_rmsnorm_op() -> OpOverload: ...
    @staticmethod
    def get_rmsnorm_fused_add_dynamic_quant_op() -> OpOverload: ...
    @staticmethod
    def get_rmsnorm_fused_dynamic_quant_op() -> OpOverload: ...
    @staticmethod
    def get_rmsnorm_group_fused_quant_op() -> OpOverload: ...
    @staticmethod
    def get_rmsnorm_group_add_fused_quant_op() -> OpOverload: ...
    @staticmethod
    def get_per_token_quant_op() -> OpOverload: ...
    @staticmethod
    def get_group_quant_op() -> OpOverload: ...
    @staticmethod
    def get_act_mul_fused_fp8_group_quant_op() -> OpOverload: ...
    @staticmethod
    def get_triton_add_rmsnorm_pad_op() -> OpOverload: ...
    @staticmethod
    def get_triton_rotary_embedding_op() -> OpOverload: ...
    @staticmethod
    def rms_norm(
        x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
    ) -> torch.Tensor: ...
    @staticmethod
    def rms_norm2d_with_add(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def gemm_a8w8(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    def triton_gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    def gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    def fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_mask: torch.Tensor | None = None,
        activation_method: int = 0,
        quant_method: int = 0,
        doweight_stage1: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        num_local_tokens: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def asm_moe_tkw1(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: torch.Tensor | None = None,
        fc2_scale: torch.Tensor | None = None,
        fc1_smooth_scale: torch.Tensor | None = None,
        fc2_smooth_scale: torch.Tensor | None = None,
        a16: bool = False,
        per_tensor_quant_scale: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
        activation_method: int = 0,
    ) -> torch.Tensor: ...
    @staticmethod
    def topk_softmax(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> tuple[torch.Tensor, ...]: ...
    @staticmethod
    def topk_sigmoid(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> tuple[torch.Tensor, ...]: ...
    @staticmethod
    def biased_grouped_topk(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0,
    ) -> None: ...
    @staticmethod
    def grouped_topk(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ) -> None: ...
    @staticmethod
    def fused_topk(
        x: torch.Tensor, router_logits: torch.Tensor, top_k: int, gate_up: bool
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def mla_decode_fwd(
        q: torch.Tensor,
        kv_buffer: torch.Tensor,
        o: torch.Tensor,
        sm_scale: float,
        qo_indptr: torch.Tensor,
        max_seqlen_qo: int,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        kv_last_page_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
        q_scale: torch.Tensor | None = None,
        kv_scale: torch.Tensor | None = None,
    ): ...
    @staticmethod
    def per_tensor_quant(
        x: torch.Tensor, quant_dtype: torch.dtype, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def per_token_quant(
        x: torch.Tensor, quant_dtype: torch.dtype, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def triton_fp4_gemm_dynamic_qaunt(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = ...,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def triton_rope_and_cache(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        flash_layout: bool,
        apply_scale: bool,
    ): ...
    @staticmethod
    def batched_gemm_a16wfp4(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale: torch.Tensor,
        Y: torch.Tensor,
        transpose_bm: bool | None = False,
        prequant: bool | None = False,
        y_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def triton_fp8_bmm(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        group_size: int = 128,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype | None = ...,
        splitK: int | None = None,
        YQ: torch.Tensor | None = None,
        transpose_bm: bool | None = False,
        config: dict | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def group_fp8_quant(
        input_2d: torch.Tensor, group_size: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def is_triton_gemm_w8a8_tuned(n: int, k: int) -> bool: ...
    @staticmethod
    def is_triton_gemm_afp4wfp4_presh_ws_tuned(n: int, k: int) -> bool: ...
    @staticmethod
    def shuffle_weight(
        self, tensor: torch.Tensor, layout: tuple[int, int] = (16, 16)
    ) -> torch.Tensor: ...
    @staticmethod
    def shuffle_weight_a16w4(
        tensor: torch.Tensor, nLane: int, gate_up: bool
    ) -> torch.Tensor: ...
    @staticmethod
    def shuffle_scale_a16w4(
        tensor: torch.Tensor, num_experts: int, gate_up: bool
    ) -> torch.Tensor: ...
    @staticmethod
    def shuffle_weights(
        *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
    ) -> tuple[torch.Tensor, ...]: ...
    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] | None = None,
        alibi_slopes: torch.Tensor | None = None,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ): ...
    @staticmethod
    def pa_fwd_asm(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables_stride0: int,
        K_QScale: torch.Tensor,
        V_QScale: torch.Tensor,
        out_: torch.Tensor,
    ): ...
