from __future__ import annotations

import torch

from exo.shared.logging import logger

INITIAL_FRACTION = 0.05
GROWTH_HEADROOM_BYTES = 512 * 1024 * 1024
MIN_GROWTH_BLOCKS = 16

_patched = False


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
    logger.info("vLLM growable KV cache patch applied")


def _patch_determine_available_memory() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.determine_available_memory

    @torch.inference_mode()
    def patched(self: "Worker") -> int:
        try:
            original(self)
        except AssertionError:
            logger.warning(
                "vLLM memory profiling assertion failed (free memory changed during init, "
                "likely another process released GPU memory). Continuing with growable cache."
            )
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        initial = max(int(free_bytes * INITIAL_FRACTION), 1)
        self._growable_max_kv_bytes = free_bytes
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
        _growable_model_runner_ref[0] = self.model_runner

    Worker.initialize_from_config = patched  # type: ignore


_growable_model_runner_ref: list["object | None"] = [None]


def _patch_kv_cache_manager_init() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_init = KVCacheManager.__init__

    def patched_init(
        self: "KVCacheManager", *args: "object", **kwargs: "object"
    ) -> None:
        original_init(self, *args, **kwargs)
        self._growable_model_runner = _growable_model_runner_ref[0]

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
        if result is not None:
            return result

        if _try_grow_cache(self):
            return original(self, request, num_new_tokens, *args, **kwargs)
        return None

    KVCacheManager.allocate_slots = patched  # type: ignore


def _try_grow_cache(kv_cache_manager: "object") -> bool:
    block_pool = kv_cache_manager.block_pool  # type: ignore
    model_runner = kv_cache_manager._growable_model_runner  # type: ignore

    if model_runner is None:
        logger.debug("No model_runner reference — cannot grow cache")
        return False

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < GROWTH_HEADROOM_BYTES:
        logger.debug(f"Only {free_bytes / (1024**3):.2f} GiB free — not enough to grow")
        return False

    kv_cache_config = model_runner._growable_kv_cache_config  # type: ignore
    old_num_blocks: int = kv_cache_config.num_blocks

    total_tensor_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    per_block_bytes = total_tensor_bytes // old_num_blocks

    usable_bytes = int(free_bytes * 0.8)
    growth_blocks = min(usable_bytes // per_block_bytes, old_num_blocks)

    if growth_blocks < MIN_GROWTH_BLOCKS:
        logger.debug(f"Growth too small ({growth_blocks} blocks)")
        return False

    new_num_blocks = old_num_blocks + growth_blocks

    logger.info(
        f"Growing KV cache: {old_num_blocks} → {new_num_blocks} blocks "
        f"(+{growth_blocks * per_block_bytes / (1024**3):.2f} GiB)"
    )

    try:
        _grow_tensors(model_runner, kv_cache_config, old_num_blocks, new_num_blocks)
        _grow_block_pool(block_pool, old_num_blocks, new_num_blocks)
        kv_cache_config.num_blocks = new_num_blocks
        for tensor_spec in kv_cache_config.kv_cache_tensors:
            tensor_spec.size = int(tensor_spec.size * new_num_blocks / old_num_blocks)
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
