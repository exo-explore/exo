from enum import Enum


class Backend(str, Enum):
    MlxMetal = "MlxMetal"
    MlxCpu = "MlxCpu"
    MlxCuda = "MlxCuda"
    Vllm = "Vllm"
    TinygradAmd = "TinygradAmd"
    TinygradMetal = "TinygradMetal"
    TinygradCuda = "TinygradCuda"
    TinygradCpu = "TinygradCpu"
    # Windows join identities. These are not MLX backends.
    WinAMD = "WinAMD"
    WinCUDA = "WinCUDA"
    WinCPU = "WinCPU"
