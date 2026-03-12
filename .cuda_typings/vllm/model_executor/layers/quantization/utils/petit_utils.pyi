import torch

def verify_petit_nvfp4_supported(quant_method: str, group_size: int | None) -> None: ...
def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None: ...
def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
