import torch
from .matmul_ogs_details._p_matmul_ogs import (
    get_per_device_per_stream_alloc_fn as get_per_device_per_stream_alloc_fn,
)
from .matmul_ogs_details.opt_flags import (
    make_opt_flags as make_opt_flags,
    update_opt_flags_constraints as update_opt_flags_constraints,
)
from .numerics_details.mxfp import MXFP_BLOCK_SIZE as MXFP_BLOCK_SIZE
from .reduce import reduce as reduce
from .specialize import (
    ClosureArg as ClosureArg,
    FnSpecs as FnSpecs,
    SpecializationModule as SpecializationModule,
)
from .tensor import (
    FP4 as FP4,
    RaggedTensorMetadata as RaggedTensorMetadata,
    Storage as Storage,
    Tensor as Tensor,
    bitwidth as bitwidth,
    wrap_torch_tensor as wrap_torch_tensor,
)
from .tensor_details.layout_details.strided import StridedLayout as StridedLayout
from _typeshed import Incomplete
from dataclasses import dataclass, field
from enum import Enum
from triton_kernels.numerics import InFlexData, OutFlexData

@dataclass
class GatherIndx:
    src_indx: torch.Tensor
    dst_indx: torch.Tensor

@dataclass
class ScatterIndx:
    src_indx: torch.Tensor
    dst_indx: torch.Tensor

@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: RaggedTensorMetadata = ...
    expected_tokens_per_expt: int = field(default=None)
    def n_blocks(self, n_rows, block_m): ...

@dataclass(frozen=True)
class FusedActivation:
    specs: FnSpecs = ...
    fn_args: tuple[object] = ...

@dataclass(frozen=True)
class Epilogue:
    specs: FnSpecs = ...
    fn_arg_values_matmul: tuple[object] = ...
    fn_arg_values_finalize: tuple[object] = ...
    effective_itemsize: float = ...

class FnName(Enum):
    QUANTIZE_MXFP8 = ...

@dataclass(frozen=True)
class FusedComm:
    out_handles: torch.Tensor
    scatter_shard_indx: torch.Tensor | None = ...
    reduce_rank: int = ...
    n_reduce_shards: int = ...

specializations: Incomplete

def can_overflow_int32(tensor: torch.Tensor): ...
def should_upcast_indices(*args): ...
@dataclass
class InnerRoutingData:
    base: RoutingData | None = ...
    block_k: int | None = ...
    x_is_padded: bool = ...
    w_is_padded: bool = ...
    @staticmethod
    def make_kernel_args(data, block_m): ...

@dataclass(frozen=True)
class FlexCtx:
    lhs_data: InFlexData = ...
    rhs_data: InFlexData = ...
    out_data: OutFlexData = ...
    acc_data: InFlexData = ...

@dataclass
class PrecisionConfig:
    max_num_imprecise_acc: int = ...
    allow_tf32: bool = ...
    flex_ctx: FlexCtx = ...
    acc_scale: int = ...
    flexpoint_saturate_inf: bool = ...
    report_quantization_err_fn: callable = ...
    act_scale: Tensor | None = ...
    weight_scale: Tensor | None = ...
    out_scale: Tensor | None = ...
    out_dtype: torch.dtype = ...
    enforce_bitwise_invariance: bool = ...

def get_swap_xw(precision_config, opt_flags): ...
@dataclass
class MatmulAllocation:
    device: str
    output: tuple[tuple[int], torch.dtype]
    scratchpads: dict[str, tuple]

def init_allocation(
    x,
    w,
    precision_config,
    fused_activation,
    routing_data,
    gather_indx,
    scatter_indx,
    inner_routing_data,
    n_reduce_shards,
    opt_flags,
): ...
def apply_allocation(allocation: MatmulAllocation, output): ...
def matmul_ogs_set_idle_sms(num_idle_sms) -> None: ...
def matmul_ogs(
    x,
    w,
    bias,
    routing_data: RoutingData | None = None,
    gather_indx: GatherIndx | None = None,
    scatter_indx: ScatterIndx | None = None,
    precision_config: PrecisionConfig | None = None,
    betas: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    out_alpha: float | None = None,
    y: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
    fused_activation: FusedActivation | None = None,
    epilogue: Epilogue | None = None,
    y_acc_in: torch.Tensor | None = None,
    inner_routing_data: InnerRoutingData | None = None,
): ...
def matmul_ogs_torch(
    x,
    w,
    bias,
    routing_data: RoutingData = None,
    gather_indx: GatherIndx = None,
    scatter_indx: ScatterIndx = None,
    precision_config: PrecisionConfig = None,
    betas=None,
    gammas=None,
    inner_routing_data: InnerRoutingData | None = None,
    round_x=None,
    round_y=None,
): ...
def post_matmul_comm_torch(
    y: torch.Tensor,
    rank: int,
    n_reduce_shards: int,
    world_size: int,
    scatter_shard_indx: torch.Tensor | None = None,
): ...
