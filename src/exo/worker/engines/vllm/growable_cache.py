import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from exo.shared.logging import logger
from exo.worker.engines.mlx.cache import KVPrefixCache

INITIAL_FRACTION = 0.05
GROWTH_HEADROOM_BYTES = 512 * 1024 * 1024
MIN_GROWTH_BLOCKS = 16

_patched = False
_prefix_cache: KVPrefixCache | None = None
_model_runner: GPUModelRunner | None = None


def get_prefix_cache() -> KVPrefixCache | None:
    return _prefix_cache


def set_prefix_cache(cache: KVPrefixCache | None) -> None:
    global _prefix_cache
    _prefix_cache = cache


def get_model_runner() -> GPUModelRunner | None:
    return _model_runner


def set_model_runner(runner: GPUModelRunner | None) -> None:
    global _model_runner
    _model_runner = runner


def patch_vllm() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    _patch_determine_available_memory()
    _patch_check_enough_kv_cache_memory()
    _patch_initialize_kv_cache_tensors()
    _patch_initialize_from_config()
    _patch_kv_cache_manager_init()
    _patch_allocate_slots()
    _patch_get_computed_blocks()
    _patch_moe_sum()
    _patch_marlin_w2_thread_config()
    logger.info("vLLM growable KV cache patch applied")


def _patch_determine_available_memory() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.determine_available_memory

    @torch.inference_mode()
    def patched(self: "Worker") -> int:
        real_empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None  # type: ignore
        try:
            original(self)
        except (AssertionError, Exception):
            pass
        finally:
            torch.cuda.empty_cache = real_empty_cache  # type: ignore
        free_bytes, _ = torch.cuda.mem_get_info()
        initial = max(int(free_bytes * INITIAL_FRACTION), 1)
        self._growable_max_kv_bytes = free_bytes
        self.available_kv_cache_memory_bytes = initial
        logger.info(
            f"Growable KV cache: initial {initial / (1024**3):.2f} GiB "
            f"(max {free_bytes / (1024**3):.2f} GiB)"
        )
        return initial

    Worker.determine_available_memory = patched  # type: ignore


def _patch_check_enough_kv_cache_memory() -> None:
    from vllm.v1.core import kv_cache_utils

    def noop(*_args: "object", **_kwargs: "object") -> None:
        pass

    kv_cache_utils._check_enough_kv_cache_memory = noop  # type: ignore


def _patch_initialize_kv_cache_tensors() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def patched_alloc(
        self: "GPUModelRunner", kv_cache_config: "object"
    ) -> "dict[str, torch.Tensor]":
        raw_tensors = original_alloc(self, kv_cache_config)
        self._growable_raw_tensors = {name: t for name, t in raw_tensors.items()}
        return raw_tensors

    GPUModelRunner._allocate_kv_cache_tensors = patched_alloc  # type: ignore

    original_init_tensors = GPUModelRunner.initialize_kv_cache_tensors

    def patched_init_tensors(
        self: "GPUModelRunner",
        kv_cache_config: "object",
        kernel_block_sizes: "list[int]",
    ) -> "dict[str, torch.Tensor]":
        self._growable_kv_cache_config = kv_cache_config
        self._growable_kernel_block_sizes = kernel_block_sizes
        return original_init_tensors(self, kv_cache_config, kernel_block_sizes)

    GPUModelRunner.initialize_kv_cache_tensors = patched_init_tensors  # type: ignore


def _patch_initialize_from_config() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.initialize_from_config

    def patched(self: "Worker", kv_cache_config: "object") -> None:
        original(self, kv_cache_config)
        set_model_runner(self.model_runner)

    Worker.initialize_from_config = patched  # type: ignore


def _patch_kv_cache_manager_init() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_init = KVCacheManager.__init__

    def patched_init(
        self: "KVCacheManager", *args: "object", **kwargs: "object"
    ) -> None:
        original_init(self, *args, **kwargs)
        self._growable_model_runner = get_model_runner()

    KVCacheManager.__init__ = patched_init  # type: ignore


def _patch_allocate_slots() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original = KVCacheManager.allocate_slots

    def patched(
        self: "KVCacheManager",
        request: "object",
        num_new_tokens: int,
        *args: "object",
        **kwargs: "object",
    ) -> "object":
        result = original(self, request, num_new_tokens, *args, **kwargs)
        if result is None and _try_grow_cache(self):
            result = original(self, request, num_new_tokens, *args, **kwargs)
        return result

    KVCacheManager.allocate_slots = patched  # type: ignore


def _try_grow_cache(kv_cache_manager: "object") -> bool:
    block_pool = kv_cache_manager.block_pool  # type: ignore
    model_runner = kv_cache_manager._growable_model_runner  # type: ignore

    if model_runner is None:
        return False

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < GROWTH_HEADROOM_BYTES:
        return False

    kv_cache_config = model_runner._growable_kv_cache_config  # type: ignore
    old_num_blocks: int = kv_cache_config.num_blocks

    total_tensor_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    per_block_bytes = total_tensor_bytes // old_num_blocks

    usable_bytes = int(free_bytes * 0.8)
    growth_blocks = min(usable_bytes // per_block_bytes, old_num_blocks)

    if growth_blocks < MIN_GROWTH_BLOCKS:
        return False

    new_num_blocks = old_num_blocks + growth_blocks

    logger.info(
        f"Growing KV cache: {old_num_blocks} → {new_num_blocks} blocks "
        f"(+{growth_blocks * per_block_bytes / (1024**3):.2f} GiB)"
    )

    try:
        kv_cache_config.num_blocks = new_num_blocks
        for tensor_spec in kv_cache_config.kv_cache_tensors:
            tensor_spec.size = int(tensor_spec.size * new_num_blocks / old_num_blocks)
        _grow_tensors(model_runner, kv_cache_config, old_num_blocks, new_num_blocks)
        _grow_block_pool(block_pool, old_num_blocks, new_num_blocks)
        logger.info(f"KV cache grown successfully to {new_num_blocks} blocks")
        return True
    except Exception:
        logger.opt(exception=True).error("Failed to grow KV cache")
        return False




def _grow_tensors(
    model_runner: "object",
    kv_cache_config: "object",
    old_num_blocks: int,
    new_num_blocks: int,
) -> None:
    raw_tensors: dict[str, torch.Tensor] = model_runner._growable_raw_tensors  # type: ignore
    ratio = new_num_blocks / old_num_blocks

    already_grown: dict[int, torch.Tensor] = {}
    new_raw_tensors: dict[str, torch.Tensor] = {}

    for layer_name, old_raw in raw_tensors.items():
        storage_id = old_raw.data_ptr()
        if storage_id in already_grown:
            new_raw_tensors[layer_name] = already_grown[storage_id]
            continue

        old_size = old_raw.numel()
        new_size = int(old_size * ratio)
        new_raw = torch.zeros(new_size, dtype=torch.int8, device=old_raw.device)
        new_raw[:old_size] = old_raw
        already_grown[storage_id] = new_raw
        new_raw_tensors[layer_name] = new_raw

    model_runner._growable_raw_tensors = new_raw_tensors  # type: ignore

    kernel_block_sizes: list[int] = model_runner._growable_kernel_block_sizes  # type: ignore
    new_kv_caches: dict[str, torch.Tensor] = model_runner._reshape_kv_cache_tensors(  # type: ignore
        kv_cache_config,
        new_raw_tensors,
        kernel_block_sizes,
    )

    forward_context: dict[str, "object"] = (
        model_runner.compilation_config.static_forward_context
    )  # type: ignore
    runner_kv_caches: list[torch.Tensor] = model_runner.kv_caches  # type: ignore
    runner_kv_caches.clear()

    from collections import defaultdict

    from vllm.v1.worker.utils import extract_layer_index

    num_attn_module = 1
    hf_config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)  # type: ignore
    if getattr(hf_config, "model_type", "") == "longcat_flash":
        num_attn_module = 2

    index2name: dict[int, list[str]] = defaultdict(list)
    for ln in new_kv_caches:
        index2name[extract_layer_index(ln, num_attn_module)].append(ln)

    for layer_index in sorted(index2name.keys()):
        for ln in index2name[layer_index]:
            runner_kv_caches.append(new_kv_caches[ln])

    for layer_name, kv_cache in new_kv_caches.items():
        forward_context[layer_name].kv_cache = [kv_cache]  # type: ignore


def _grow_block_pool(
    block_pool: "object", old_num_blocks: int, new_num_blocks: int
) -> None:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    new_blocks: list["KVCacheBlock"] = []
    for idx in range(old_num_blocks, new_num_blocks):
        block = KVCacheBlock(idx)
        block_pool.blocks.append(block)  # type: ignore
        new_blocks.append(block)

    block_pool.free_block_queue.append_n(new_blocks)  # type: ignore
    block_pool.num_gpu_blocks = new_num_blocks  # type: ignore


def _patch_moe_sum() -> None:
    import vllm._custom_ops as ops  # type: ignore[reportMissingImports]

    def moe_sum_f32(x: "torch.Tensor", output: "torch.Tensor") -> None:
        output[:] = x.to(torch.float32).sum(dim=1).to(output.dtype)  # type: ignore

    ops.moe_sum = moe_sum_f32  # type: ignore


def _patch_marlin_w2_thread_config() -> None:
    try:
        import vllm._custom_ops as ops  # type: ignore[reportMissingImports]
    except ImportError:
        return

    original_gemm = ops.moe_wna16_marlin_gemm

    def patched_gemm(*args: "object", **kwargs: "object") -> "object":
        kwargs["thread_k"] = 64
        kwargs["thread_n"] = 128
        return original_gemm(*args, **kwargs)

    ops.moe_wna16_marlin_gemm = patched_gemm  # type: ignore


def _patch_get_computed_blocks() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

    original = KVCacheManager.get_computed_blocks

    def patched(
        self: KVCacheManager,
        request: Request,
    ) -> tuple[KVCacheBlocks, int]:
        prefix_cache = get_prefix_cache()
        if prefix_cache is None or request.prompt_token_ids is None:
            return original(self, request)

        from exo.worker.engines.vllm.kv_cache import (
            TorchKVCache as _TorchKVCache,  # noqa: F811
        )

        try:
            torch_cache, num_matched, _ = prefix_cache.lookup(
                list(request.prompt_token_ids)
            )  # type: ignore[reportUnknownMemberType]
        except Exception:
            return original(self, request)

        if (
            torch_cache is None
            or not isinstance(torch_cache, _TorchKVCache)
            or num_matched == 0
        ):
            return original(self, request)

        from vllm.utils.math_utils import cdiv  # type: ignore[reportMissingImports]

        from exo.worker.engines.vllm.vllm_generator import _build_layer_groups

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        null_block = self.block_pool.null_block
        save_offsets = torch_cache.token_offset_per_group or [0] * num_groups

        for gi in range(num_groups):
            save_off = save_offsets[gi] if gi < len(save_offsets) else 0
            if save_off > 0:
                spec = self.kv_cache_config.kv_cache_groups[gi].kv_cache_spec  # type: ignore
                window = getattr(spec, "sliding_window", 0) or 0
                if window > 0 and num_matched < save_off + window:
                    return original(self, request)

        real_block_counts: list[int] = []
        skipped_block_counts: list[int] = []
        total_needed = 0
        for gi in range(num_groups):
            mgr = self.coordinator.single_type_managers[gi]  # type: ignore
            block_size: int = self.kv_cache_config.kv_cache_groups[
                gi
            ].kv_cache_spec.block_size  # type: ignore
            num_skipped: int = mgr.get_num_skipped_tokens(num_matched)  # type: ignore
            num_skipped_blocks = num_skipped // block_size
            num_real = cdiv(num_matched, block_size) - num_skipped_blocks
            real_block_counts.append(num_real)
            skipped_block_counts.append(num_skipped_blocks)
            total_needed += num_real

        if self.block_pool.get_num_free_blocks() < total_needed:
            return original(self, request)

        blocks_per_group: list[list[KVCacheBlock]] = []
        token_offset_per_group: list[int] = []
        for gi in range(num_groups):
            mgr = self.coordinator.single_type_managers[gi]  # type: ignore
            block_size = self.kv_cache_config.kv_cache_groups[
                gi
            ].kv_cache_spec.block_size  # type: ignore
            real_blocks: list[KVCacheBlock] = self.block_pool.get_new_blocks(
                real_block_counts[gi]
            )  # type: ignore
            blocks_per_group.append(real_blocks)

            full_block_list = [null_block] * skipped_block_counts[gi] + list(
                real_blocks
            )
            req_blocks = mgr.req_to_blocks[request.request_id]  # type: ignore
            req_blocks.extend(full_block_list)  # type: ignore

            token_offset_per_group.append(skipped_block_counts[gi] * block_size)

        block_ids_per_group = [[b.block_id for b in grp] for grp in blocks_per_group]
        layer_to_group = _build_layer_groups(self.kv_cache_config)
        model_runner = self._growable_model_runner  # type: ignore[reportAttributeAccessIssue]
        if model_runner is not None:
            torch_cache.write_to_vllm_blocks(  # type: ignore
                model_runner.kv_caches,
                block_ids_per_group,
                layer_to_group,  # type: ignore
                token_offset_per_group,
            )

        total_blocks = sum(len(g) for g in blocks_per_group)
        logger.info(
            f"Prefix cache hit: {num_matched} tokens, {total_blocks} blocks ({num_groups} groups)"
        )
        return self.empty_kv_cache_blocks, num_matched

    KVCacheManager.get_computed_blocks = patched  # type: ignore[reportAttributeAccessIssue]
