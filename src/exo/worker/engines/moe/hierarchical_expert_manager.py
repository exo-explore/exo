import concurrent.futures
import time
from collections.abc import Callable
from typing import Any

from exo.shared.types.moe_offload import (
    DEFAULT_LOOKAHEAD_LAYERS,
    MoEOffloadTelemetry,
    OffloadedExpertWeight,
    StorageTier,
)


class EarlyLayerPrefetchPredictor:
    """Predicts future MoE expert activations across early Transformer layers."""

    def __init__(self, lookahead_layers: int = DEFAULT_LOOKAHEAD_LAYERS) -> None:
        self.lookahead_layers = lookahead_layers

    def predict_future_experts(
        self,
        current_layer_id: int,
        router_probabilities: list[float] | dict[int, float],
        top_k: int = 2,
    ) -> list[tuple[int, int]]:
        """Predicts Top-K candidate experts for layer (current + lookahead)."""
        target_layer = current_layer_id + self.lookahead_layers

        # Sort router probabilities to extract top_k expert IDs
        if isinstance(router_probabilities, dict):
            sorted_experts = sorted(router_probabilities.items(), key=lambda item: item[1], reverse=True)
            top_expert_ids = [exp_id for exp_id, _ in sorted_experts[:top_k]]
        else:
            indexed_probs = list(enumerate(router_probabilities))
            indexed_probs.sort(key=lambda item: item[1], reverse=True)
            top_expert_ids = [exp_id for exp_id, _ in indexed_probs[:top_k]]

        return [(target_layer, exp_id) for exp_id in top_expert_ids]


class HierarchicalExpertEngine:
    """Manages multi-tier MoE expert caching (VRAM/RAM/SSD) with non-blocking speculative prefetching."""

    def __init__(
        self,
        fast_capacity_slots: int = 4,
        simulate_ssd_latency_sec: float = 0.005,
        max_workers: int = 2,
    ) -> None:
        self.fast_capacity_slots = fast_capacity_slots
        self.simulate_ssd_latency_sec = simulate_ssd_latency_sec

        self.fast_pool: dict[str, OffloadedExpertWeight] = {}
        self.cold_storage: dict[str, OffloadedExpertWeight] = {}
        self.prefetch_buffer: dict[str, OffloadedExpertWeight] = {}

        self.predictor = EarlyLayerPrefetchPredictor()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: dict[str, concurrent.futures.Future[OffloadedExpertWeight | None]] = {}

        self._total_requests = 0
        self._hot_hits = 0
        self._prefetch_hits = 0
        self._cold_misses = 0
        self._swap_in_bytes = 0
        self._io_time_masked_ms = 0.0
        self._total_io_latency_ms = 0.0

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        weights_data: tuple[float, ...] | list[float],
        tier: StorageTier = "nvme_ssd",
        is_pinned_hot: bool = False,
    ) -> OffloadedExpertWeight:
        """Registers an expert into either fast pool or cold secondary storage."""
        weights_tuple = tuple(weights_data)
        num_params = len(weights_tuple)
        size_bytes = num_params * 4  # float32

        expert = OffloadedExpertWeight(
            expert_id=expert_id,
            layer_id=layer_id,
            num_params=num_params,
            size_in_bytes=size_bytes,
            current_tier=tier,
            is_pinned_hot=is_pinned_hot,
            access_count=0,
            last_accessed_timestamp=time.time(),
            weights_data=weights_tuple,
        )

        key = expert.expert_key
        if tier in ("vram", "host_ram"):
            self.fast_pool[key] = expert
        else:
            self.cold_storage[key] = expert

        return expert

    def _load_from_cold_storage(self, key: str) -> OffloadedExpertWeight | None:
        """Simulates asynchronous DMA/NVMe transfer of an expert into memory."""
        if key not in self.cold_storage:
            return None

        if self.simulate_ssd_latency_sec > 0:
            time.sleep(self.simulate_ssd_latency_sec)

        cold_expert = self.cold_storage[key]
        upgraded_expert = OffloadedExpertWeight(
            expert_id=cold_expert.expert_id,
            layer_id=cold_expert.layer_id,
            num_params=cold_expert.num_params,
            size_in_bytes=cold_expert.size_in_bytes,
            current_tier="host_ram",
            is_pinned_hot=cold_expert.is_pinned_hot,
            access_count=cold_expert.access_count + 1,
            last_accessed_timestamp=time.time(),
            weights_data=cold_expert.weights_data,
        )
        return upgraded_expert

    def start_speculative_prefetch(self, layer_id: int, expert_id: int) -> None:
        """Asynchronously trigger background SSD-to-RAM prefetch for candidate expert."""
        key = f"l{layer_id}_e{expert_id}"
        if key in self.fast_pool or key in self.prefetch_buffer or key in self.pending_futures:
            return

        future = self.executor.submit(self._load_from_cold_storage, key)
        self.pending_futures[key] = future

    def _evict_lru_unpinned_if_needed(self) -> None:
        """Evicts unpinned fast pool experts back to cold storage if slots exceeded."""
        if len(self.fast_pool) <= self.fast_capacity_slots:
            return

        # Find evictable candidates
        candidates = [exp for exp in self.fast_pool.values() if not exp.is_pinned_hot]
        if not candidates:
            return

        # Sort by oldest accessed
        candidates.sort(key=lambda e: e.last_accessed_timestamp)
        victim = candidates[0]

        demoted = OffloadedExpertWeight(
            expert_id=victim.expert_id,
            layer_id=victim.layer_id,
            num_params=victim.num_params,
            size_in_bytes=victim.size_in_bytes,
            current_tier="nvme_ssd",
            is_pinned_hot=False,
            access_count=victim.access_count,
            last_accessed_timestamp=victim.last_accessed_timestamp,
            weights_data=victim.weights_data,
        )
        self.fast_pool.pop(victim.expert_key, None)
        self.cold_storage[victim.expert_key] = demoted

    def request_expert(self, layer_id: int, expert_id: int) -> tuple[OffloadedExpertWeight | None, str]:
        """Requests an expert for current execution. Returns (expert, hit_status)."""
        key = f"l{layer_id}_e{expert_id}"
        self._total_requests += 1

        # 1. Hot Fast Pool Hit
        if key in self.fast_pool:
            self._hot_hits += 1
            expert = self.fast_pool[key]
            updated = OffloadedExpertWeight(
                expert_id=expert.expert_id,
                layer_id=expert.layer_id,
                num_params=expert.num_params,
                size_in_bytes=expert.size_in_bytes,
                current_tier=expert.current_tier,
                is_pinned_hot=expert.is_pinned_hot,
                access_count=expert.access_count + 1,
                last_accessed_timestamp=time.time(),
                weights_data=expert.weights_data,
            )
            self.fast_pool[key] = updated
            return updated, "hot_hit"

        # 2. Prefetch Buffer Hit
        if key in self.prefetch_buffer:
            self._prefetch_hits += 1
            expert = self.prefetch_buffer.pop(key)
            self._swap_in_bytes += expert.size_in_bytes
            self.fast_pool[key] = expert
            self._evict_lru_unpinned_if_needed()
            return expert, "prefetch_hit"

        # 3. In-flight Future Resolution
        if key in self.pending_futures:
            try:
                expert = self.pending_futures[key].result(timeout=1.0)
                self.pending_futures.pop(key, None)
                if expert:
                    self._prefetch_hits += 1
                    self._swap_in_bytes += expert.size_in_bytes
                    self.fast_pool[key] = expert
                    self._evict_lru_unpinned_if_needed()
                    return expert, "prefetch_hit"
            except (concurrent.futures.CancelledError, TimeoutError, OSError, RuntimeError):
                self.pending_futures.pop(key, None)

        # 4. Cold SSD Miss
        self._cold_misses += 1
        start_io = time.perf_counter()
        loaded_expert = self._load_from_cold_storage(key)
        elapsed_io_ms = (time.perf_counter() - start_io) * 1000.0
        self._total_io_latency_ms += elapsed_io_ms

        if loaded_expert:
            self._swap_in_bytes += loaded_expert.size_in_bytes
            self.fast_pool[key] = loaded_expert
            self._evict_lru_unpinned_if_needed()

        return loaded_expert, "cold_miss"

    def step_forward_with_early_prefetch(
        self,
        current_layer_id: int,
        current_expert_id: int,
        router_probabilities: list[float] | dict[int, float],
        compute_fn: Callable[..., Any] | None = None,
        compute_args: tuple[Any, ...] = (),
    ) -> tuple[OffloadedExpertWeight | None, MoEOffloadTelemetry]:
        """Executes forward layer compute while triggering early prefetch for predicted upcoming layers."""
        # 1. Predict and prefetch for early upcoming layers
        candidates = self.predictor.predict_future_experts(
            current_layer_id=current_layer_id,
            router_probabilities=router_probabilities,
            top_k=2,
        )

        io_start = time.perf_counter()
        for cand_layer, cand_exp in candidates:
            self.start_speculative_prefetch(cand_layer, cand_exp)

        # 2. Concurrently execute current compute workload
        compute_start = time.perf_counter()
        if compute_fn:
            compute_fn(*compute_args)
        else:
            time.sleep(0.008)  # default simulated forward layer work (8ms)
        compute_end = time.perf_counter()
        compute_time_ms = (compute_end - compute_start) * 1000.0

        # 3. Synchronize in-flight prefetch completions
        for key, future in list(self.pending_futures.items()):
            if future.done():
                try:
                    res = future.result()
                    if res:
                        self.prefetch_buffer[key] = res
                except (concurrent.futures.CancelledError, TimeoutError, OSError, RuntimeError):
                    pass
                self.pending_futures.pop(key, None)

        io_end = time.perf_counter()
        total_io_ms = (io_end - io_start) * 1000.0
        masked_io_ms = min(compute_time_ms, total_io_ms)

        self._io_time_masked_ms += masked_io_ms
        self._total_io_latency_ms += masked_io_ms

        # 4. Fulfill request for current active expert
        expert, _ = self.request_expert(current_layer_id, current_expert_id)
        return expert, self.get_telemetry()

    def get_telemetry(self) -> MoEOffloadTelemetry:
        return MoEOffloadTelemetry(
            total_expert_requests=self._total_requests,
            hot_cache_hits=self._hot_hits,
            speculative_prefetch_hits=self._prefetch_hits,
            cold_ssd_misses=self._cold_misses,
            total_swap_in_bytes=self._swap_in_bytes,
            io_time_masked_ms=self._io_time_masked_ms,
            total_io_latency_ms=self._total_io_latency_ms,
        )

    def close(self) -> None:
        self.executor.shutdown(wait=False)
