import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from vllm.forward_context import (
    get_forward_context as get_forward_context,
    is_forward_context_available as is_forward_context_available,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
    apply_moe_activation as apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.utils import (
    count_expert_num_tokens as count_expert_num_tokens,
    disable_inplace as disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.worker.ubatching import (
    dbo_enabled as dbo_enabled,
    dbo_maybe_run_recv_hook as dbo_maybe_run_recv_hook,
    dbo_register_recv_hook as dbo_register_recv_hook,
    dbo_yield as dbo_yield,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager as current_workspace_manager,
)

logger: Incomplete

class FusedMoEActivationFormat(Enum):
    Standard = ("standard",)
    BatchedExperts = ("batched_experts",)

@dataclass
class ExpertTokensMetadata:
    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None
    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> ExpertTokensMetadata: ...

class TopKWeightAndReduce(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...

PrepareResultType: Incomplete
PrepareMonolithicResultType: Incomplete
ReceiverType = Callable[[], PrepareResultType]

class FusedMoEPrepareAndFinalize(ABC, metaclass=abc.ABCMeta):
    def post_init_setup(self, fused_experts: FusedMoEExperts): ...
    @property
    @abstractmethod
    def activation_format(self) -> FusedMoEActivationFormat: ...
    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None: ...
    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None: ...
    @abstractmethod
    def num_dispatchers(self) -> int: ...
    @abstractmethod
    def output_is_reduced(self) -> bool: ...

class FusedMoEPrepareAndFinalizeModular(
    FusedMoEPrepareAndFinalize, metaclass=abc.ABCMeta
):
    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> PrepareResultType: ...
    def supports_async(self) -> bool: ...
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[Callable, ReceiverType] | ReceiverType: ...
    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None: ...
    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable: ...

class FusedMoEPrepareAndFinalizeMonolithic(
    FusedMoEPrepareAndFinalize, metaclass=abc.ABCMeta
):
    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType: ...
    @abstractmethod
    def finalize(self, fused_expert_output: torch.Tensor) -> torch.Tensor: ...

class FusedMoEExperts(ABC, metaclass=abc.ABCMeta):
    moe_config: Incomplete
    quant_config: Incomplete
    max_num_tokens: Incomplete
    num_dispatchers: Incomplete
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None: ...
    @staticmethod
    def is_monolithic() -> bool: ...
    @property
    def expects_unquantized_inputs(self) -> bool: ...
    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat: ...
    @staticmethod
    def is_supported_config(
        cls,
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]: ...
    @property
    def quant_dtype(self) -> torch.dtype | str | None: ...
    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None: ...
    @property
    def block_shape(self) -> list[int] | None: ...
    @property
    def per_act_token_quant(self) -> bool: ...
    @property
    def per_out_ch_quant(self) -> bool: ...
    @property
    def a1_scale(self) -> torch.Tensor | None: ...
    @property
    def a2_scale(self) -> torch.Tensor | None: ...
    @property
    def a1_gscale(self) -> torch.Tensor | None: ...
    @property
    def a2_gscale(self) -> torch.Tensor | None: ...
    @property
    def w1_scale(self) -> torch.Tensor | None: ...
    @property
    def w2_scale(self) -> torch.Tensor | None: ...
    @property
    def w1_zp(self) -> torch.Tensor | None: ...
    @property
    def w2_zp(self) -> torch.Tensor | None: ...
    @property
    def w1_bias(self) -> torch.Tensor | None: ...
    @property
    def w2_bias(self) -> torch.Tensor | None: ...
    @property
    def g1_alphas(self) -> torch.Tensor | None: ...
    @property
    def g2_alphas(self) -> torch.Tensor | None: ...
    @abstractmethod
    def supports_chunking(self) -> bool: ...
    @abstractmethod
    def supports_expert_map(self) -> bool: ...
    def supports_packed_ue8m0_act_scales(self) -> bool: ...
    def enable_chunking(self): ...

class FusedMoEExpertsModular(FusedMoEExperts, metaclass=abc.ABCMeta):
    @staticmethod
    def is_monolithic() -> bool: ...
    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]: ...
    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype: ...
    @abstractmethod
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    @staticmethod
    def adjust_N_for_activation(N: int, activation: MoEActivation) -> int: ...
    def activation(
        self, activation: MoEActivation, output: torch.Tensor, input: torch.Tensor
    ) -> None: ...
    @abstractmethod
    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce: ...
    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None: ...

class FusedMoEExpertsMonolithic(FusedMoEExperts, metaclass=abc.ABCMeta):
    @staticmethod
    def is_monolithic() -> bool: ...
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor: ...

class FusedMoEKernelModularImpl:
    prepare_finalize: Incomplete
    fused_experts: Incomplete
    shared_experts: Incomplete
    moe_parallel_config: Incomplete
    inplace: Incomplete
    is_dp_ep: Incomplete
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        fused_experts: FusedMoEExpertsModular,
        shared_experts: torch.nn.Module | None = None,
        moe_parallel_config: FusedMoEParallelConfig | None = None,
        inplace: bool = False,
    ) -> None: ...
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation = ...,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class FusedMoEKernelMonolithicImpl:
    prepare_finalize: Incomplete
    fused_experts: Incomplete
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeMonolithic,
        fused_experts: FusedMoEExpertsMonolithic,
    ) -> None: ...
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor: ...

class FusedMoEKernel:
    shared_experts: Incomplete
    impl: FusedMoEKernelModularImpl | FusedMoEKernelMonolithicImpl
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEExperts,
        shared_experts: torch.nn.Module | None = None,
        moe_parallel_config: FusedMoEParallelConfig | None = None,
        inplace: bool = False,
    ) -> None: ...
    @property
    def is_monolithic(self) -> bool: ...
    @property
    def prepare_finalize(self) -> FusedMoEPrepareAndFinalize: ...
    @property
    def fused_experts(self) -> FusedMoEExperts: ...
    def supports_expert_map(self) -> bool: ...
    def output_is_reduced(self) -> bool: ...
    def apply_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor: ...
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
