from _typeshed import Incomplete
from enum import Enum
from vllm.logger import init_logger as init_logger

logger: Incomplete
OCP_MX_BLOCK_SIZE: int
OCP_MX_DTYPES: Incomplete
SUPPORTED_OCP_MX_DTYPES: Incomplete

class OCP_MX_Scheme(str, Enum):
    w_mxfp4 = "w_mxfp4"
    w_mxfp4_a_mxfp4 = "w_mxfp4_a_mxfp4"
    w_mxfp4_a_mxfp6_e3m2 = "w_mxfp4_a_mxfp6_e3m2"
    w_mxfp4_a_mxfp6_e2m3 = "w_mxfp4_a_mxfp6_e2m3"
    w_mxfp4_a_fp8 = "w_mxfp4_a_fp8"
    w_mxfp6_e3m2 = "w_mxfp6_e3m2"
    w_mxfp6_e3m2_a_mxfp6_e3m2 = "w_mxfp6_e3m2_a_mxfp6_e3m2"
    w_mxfp6_e3m2_a_fp8 = "w_mxfp6_e3m2_a_fp8"
    w_mxfp6_e2m3 = "w_mxfp6_e2m3"
    w_mxfp6_e2m3_a_mxfp6_e2m3 = "w_mxfp6_e2m3_a_mxfp6_e2m3"
    w_mxfp6_e2m3_a_fp8 = "w_mxfp6_e2m3_a_fp8"
    @classmethod
    def from_quant_dtype(cls, input_dtype: str | None, weight_dtype: str | None): ...
