import torch

class _CompilationConfig:
    static_forward_context: dict[str, object]

class _ModelConfig:
    hf_config: object

class GPUModelRunner:
    kv_caches: list[torch.Tensor]
    compilation_config: _CompilationConfig
    model_config: _ModelConfig | None
    def _allocate_kv_cache_tensors(
        self, kv_cache_config: object
    ) -> dict[str, torch.Tensor]: ...
    def initialize_kv_cache_tensors(
        self, kv_cache_config: object, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]: ...
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: object,
        raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]: ...
