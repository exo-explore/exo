import torch
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h as chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o as chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd as chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum as chunk_local_cumsum
from .l2norm import l2norm_fwd as l2norm_fwd
from .solve_tril import solve_tril as solve_tril
from .utils import SUPPRESS_LEVEL as SUPPRESS_LEVEL, input_guard as input_guard
from .wy_fast import recompute_w_u_fwd as recompute_w_u_fwd

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
): ...

class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ): ...

@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
): ...
