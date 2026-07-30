from enum import Enum


class Backend(str, Enum):
    MlxMetal = "MlxMetal"
    MlxCpu = "MlxCpu"
    MlxCuda = "MlxCuda"
    Vllm = "Vllm"
