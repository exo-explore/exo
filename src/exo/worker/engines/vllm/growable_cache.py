# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from exo.shared.logging import logger
from exo.worker.engines.mlx.cache import KVPrefixCache

INITIAL_FRACTION = 0.05
GROWTH_HEADROOM_BYTES = 512 * 1024 * 1024
MIN_GROWTH_BLOCKS = 16

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager


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

    _patch_nogds()
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


def _patch_nogds() -> None:
    from vllm.model_executor.model_loader import weight_utils

    original = weight_utils._init_fastsafetensors_loader

    def patched(
        pg: "torch.distributed.ProcessGroup",
        device: "torch.device",
        f_list: list[str],
        *,
        nogds: bool = False,
    ) -> object:
        return original(pg, device, f_list, nogds=True)

    weight_utils._init_fastsafetensors_loader = patched


def _patch_determine_available_memory() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    # original = Worker.determine_available_memory

    @torch.inference_mode()
    def patched(self: "Worker") -> int:
        import pathlib
        import shutil

        compile_cache = pathlib.Path.home() / ".cache" / "vllm" / "torch_compile_cache"
        if compile_cache.exists():
            shutil.rmtree(compile_cache, ignore_errors=True)

        free_bytes, _ = torch.cuda.mem_get_info()
        initial = max(int(free_bytes * INITIAL_FRACTION), 1)
        self._growable_max_kv_bytes = free_bytes
        self.available_kv_cache_memory_bytes = initial
        logger.info(
            f"Growable KV cache: initial {initial / (1024**3):.2f} GiB "
            f"(max {free_bytes / (1024**3):.2f} GiB)"
        )
        return initial

    Worker.determine_available_memory = patched


def _patch_check_enough_kv_cache_memory() -> None:
    from vllm.v1.core import kv_cache_utils

    def noop(*_args: "object", **_kwargs: "object") -> None:
        pass

    kv_cache_utils._check_enough_kv_cache_memory = noop


def _patch_initialize_kv_cache_tensors() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def patched_alloc(
        self: GPUModelRunner, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        raw_tensors = original_alloc(self, kv_cache_config)
        self._growable_raw_tensors = {name: t for name, t in raw_tensors.items()}
        return raw_tensors

    GPUModelRunner._allocate_kv_cache_tensors = patched_alloc

    original_init_tensors = GPUModelRunner.initialize_kv_cache_tensors

    def patched_init_tensors(
        self: GPUModelRunner,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        self._growable_kv_cache_config = kv_cache_config
        self._growable_kernel_block_sizes = kernel_block_sizes
        return original_init_tensors(self, kv_cache_config, kernel_block_sizes)

    GPUModelRunner.initialize_kv_cache_tensors = patched_init_tensors


def _patch_initialize_from_config() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    original_init_attn = GPUModelRunner.initialize_attn_backend

    def clear_and_reinit_attn(
        self: GPUModelRunner,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self.attn_groups.clear()
        original_init_attn(self, kv_cache_config)

    GPUModelRunner.initialize_attn_backend = clear_and_reinit_attn

    original = Worker.initialize_from_config

    def patched(self: Worker, kv_cache_config: KVCacheConfig) -> None:
        original(self, kv_cache_config)
        set_model_runner(self.model_runner)

    Worker.initialize_from_config = patched


def _patch_kv_cache_manager_init() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_init = KVCacheManager.__init__

    def patched_init(
        self: KVCacheManager,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        original_init(
            self,
            kv_cache_config,
            max_model_len,
            hash_block_size,
            enable_caching,
            use_eagle,
            log_stats,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            metrics_collector,
        )
        self._growable_model_runner = get_model_runner()

    KVCacheManager.__init__ = patched_init


def _patch_allocate_slots() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

    original = KVCacheManager.allocate_slots

    def patched(
        self: KVCacheManager,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        result = original(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
        )
        while result is None and _try_grow_cache(self):
            result = original(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens,
                num_external_computed_tokens,
                delay_cache_blocks,
                num_encoder_tokens,
            )

        return result

    KVCacheManager.allocate_slots = patched

    if hasattr(KVCacheManager, "can_fit_full_sequence"):
        original_can_fit = cast(
            Callable[..., bool],
            KVCacheManager.can_fit_full_sequence,
        )

        def patched_can_fit(
            self: KVCacheManager,
            request: Request,
            num_new_computed_tokens: int = 0,
            new_computed_blocks: KVCacheBlocks | None = None,
            num_external_computed_tokens: int = 0,
            num_encoder_tokens: int = 0,
        ) -> bool:
            result: bool = original_can_fit(
                self,
                request,
                num_new_computed_tokens,
                new_computed_blocks,
                num_external_computed_tokens,
                num_encoder_tokens,
            )
            while not result and _try_grow_cache(self):
                result = original_can_fit(
                    self,
                    request,
                    num_new_computed_tokens,
                    new_computed_blocks,
                    num_external_computed_tokens,
                    num_encoder_tokens,
                )
            return result

        KVCacheManager.can_fit_full_sequence = patched_can_fit


def _try_grow_cache(kv_cache_manager: "KVCacheManager") -> bool:
    block_pool = kv_cache_manager.block_pool
    model_runner = cast(GPUModelRunner | None, kv_cache_manager._growable_model_runner)

    if model_runner is None:
        return False

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < GROWTH_HEADROOM_BYTES:
        return False

    kv_cache_config = cast(KVCacheConfig, model_runner._growable_kv_cache_config)
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
    model_runner: GPUModelRunner,
    kv_cache_config: KVCacheConfig,
    old_num_blocks: int,
    new_num_blocks: int,
) -> None:
    raw_tensors: dict[str, torch.Tensor] = cast(
        dict[str, torch.Tensor], model_runner._growable_raw_tensors
    )
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

    model_runner._growable_raw_tensors = new_raw_tensors

    kernel_block_sizes: list[int] = cast(
        list[int], model_runner._growable_kernel_block_sizes
    )
    new_kv_caches: dict[str, torch.Tensor] = model_runner._reshape_kv_cache_tensors(
        kv_cache_config,
        new_raw_tensors,
        kernel_block_sizes,
    )

    forward_context: dict[str, Any] = (
        model_runner.compilation_config.static_forward_context
    )
    runner_kv_caches: list[torch.Tensor] = model_runner.kv_caches

    from collections import defaultdict

    from vllm.model_executor.models.utils import extract_layer_index

    num_attn_module = 1
    hf_config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)
    if getattr(hf_config, "model_type", "") == "longcat_flash":
        num_attn_module = 2

    index2name: dict[int, list[str]] = defaultdict(list)
    for ln in new_kv_caches:
        index2name[extract_layer_index(ln, num_attn_module)].append(ln)

    new_ordered: list[torch.Tensor] = []
    for layer_index in sorted(index2name.keys()):
        for ln in index2name[layer_index]:
            new_ordered.append(new_kv_caches[ln])

    for i, new_kv in enumerate(new_ordered):
        if i < len(runner_kv_caches):
            runner_kv_caches[i] = new_kv
        else:
            runner_kv_caches.append(new_kv)

    new_kv_typed = cast("dict[str, torch.Tensor | list[torch.Tensor]]", new_kv_caches)
    for layer_name, new_kv in new_kv_typed.items():
        old_kv_list = cast(
            "list[torch.Tensor | list[torch.Tensor]] | torch.Tensor | None",
            forward_context[layer_name].kv_cache,
        )
        if old_kv_list is not None and (
            not isinstance(old_kv_list, torch.Tensor) or old_kv_list.numel() > 0
        ):
            old_entry = old_kv_list[0]
            if isinstance(old_entry, list) and isinstance(new_kv, list):
                for old_t, new_t in zip(old_entry, new_kv, strict=True):
                    old_t.set_(
                        new_t.storage(),
                        new_t.storage_offset(),
                        new_t.shape,
                        new_t.stride(),
                    )
            elif isinstance(old_entry, torch.Tensor) and isinstance(
                new_kv, torch.Tensor
            ):
                old_entry.set_(
                    new_kv.storage(),
                    new_kv.storage_offset(),
                    new_kv.shape,
                    new_kv.stride(),
                )
            else:
                forward_context[layer_name].kv_cache = [new_kv]
        else:
            forward_context[layer_name].kv_cache = [new_kv]


def _grow_block_pool(
    block_pool: BlockPool, old_num_blocks: int, new_num_blocks: int
) -> None:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    new_blocks: list[KVCacheBlock] = []
    for idx in range(old_num_blocks, new_num_blocks):
        block = KVCacheBlock(idx)
        block_pool.blocks.append(block)
        new_blocks.append(block)

    block_pool.free_block_queue.append_n(new_blocks)
    block_pool.num_gpu_blocks = new_num_blocks


def _patch_moe_sum() -> None:
    import vllm._custom_ops as ops

    def moe_sum_f32(x: "torch.Tensor", output: "torch.Tensor") -> None:
        output[:] = x.to(torch.float32).sum(dim=1).to(output.dtype)

    ops.moe_sum = moe_sum_f32


def _patch_marlin_w2_thread_config() -> None:
    try:
        import vllm._custom_ops as ops
    except ImportError:
        return

    original_gemm = cast(Callable[..., object], ops.moe_wna16_marlin_gemm)

    def patched_gemm(*args: object, **kwargs: object) -> object:
        kwargs["thread_k"] = 64
        kwargs["thread_n"] = 128
        return original_gemm(*args, **kwargs)

    ops.moe_wna16_marlin_gemm = patched_gemm


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

        try:
            torch_cache, num_matched, _ = prefix_cache.lookup(
                list(request.prompt_token_ids)
            )
        except Exception:
            return original(self, request)

        if torch_cache is None or num_matched == 0:
            return original(self, request)

        from vllm.utils.math_utils import cdiv

        from exo.worker.engines.vllm.generator import build_layer_groups

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        null_block = self.block_pool.null_block
        save_offsets = torch_cache.token_offset_per_group or [0] * num_groups

        for gi in range(num_groups):
            save_off = save_offsets[gi] if gi < len(save_offsets) else 0
            if save_off > 0:
                spec = self.kv_cache_config.kv_cache_groups[gi].kv_cache_spec
                window = getattr(spec, "sliding_window", 0) or 0
                if window > 0 and num_matched < save_off + window:
                    return original(self, request)

        real_block_counts: list[int] = []
        skipped_block_counts: list[int] = []
        total_needed = 0
        for gi in range(num_groups):
            mgr = self.coordinator.single_type_managers[gi]
            block_size: int = self.kv_cache_config.kv_cache_groups[
                gi
            ].kv_cache_spec.block_size
            num_skipped: int = mgr.get_num_skipped_tokens(num_matched)
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
            mgr = self.coordinator.single_type_managers[gi]
            block_size = self.kv_cache_config.kv_cache_groups[
                gi
            ].kv_cache_spec.block_size
            real_blocks: list[KVCacheBlock] = self.block_pool.get_new_blocks(
                real_block_counts[gi]
            )
            blocks_per_group.append(real_blocks)

            full_block_list = [null_block] * skipped_block_counts[gi] + list(
                real_blocks
            )
            req_blocks = mgr.req_to_blocks[request.request_id]
            req_blocks.extend(full_block_list)

            token_offset_per_group.append(skipped_block_counts[gi] * block_size)

        block_ids_per_group = [[b.block_id for b in grp] for grp in blocks_per_group]
        layer_to_group = build_layer_groups(self.kv_cache_config)
        model_runner = cast(GPUModelRunner | None, self._growable_model_runner)
        if model_runner is not None:
            torch_cache.write_to_vllm_blocks(
                cast(
                    "list[torch.Tensor | list[torch.Tensor]]",
                    model_runner.kv_caches,
                ),
                block_ids_per_group,
                layer_to_group,
                token_offset_per_group,
            )

        total_blocks = sum(len(g) for g in blocks_per_group)
        logger.info(
            f"Prefix cache hit: {num_matched} tokens, {total_blocks} blocks ({num_groups} groups)"
        )
        return self.empty_kv_cache_blocks, num_matched

    KVCacheManager.get_computed_blocks = patched
