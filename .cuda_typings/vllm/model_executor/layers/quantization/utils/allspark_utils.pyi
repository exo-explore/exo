import torch
from _typeshed import Incomplete
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import ScalarType as ScalarType, scalar_types as scalar_types

ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD: int
ALLSPARK_SUPPORTED_QUANT_TYPES: Incomplete
ALLSPARK_AMPERE_N_ALIGN: int
ALLSPARK_AMPERE_K_ALIGN: int

def check_allspark_supported_dtype_shape(
    input_size_per_partition: int,
    output_size_per_partition: int,
    group_size: int,
    weight_dtype: ScalarType,
    act_dtype: torch.dtype,
): ...
