"""exo's sampler and logits processors: mlx-lm's, with the same results for less work per token.

Sampler: mlx-lm's sampler chain, with its top-k computed by an exact radix selection and the
sort inside its top-p by a radix sort.

mlx-lm's ``apply_top_k`` finds the k best log-probs with ``mx.argpartition``, which MLX's GPU
backend runs as a full merge sort of the vocabulary, and then scatters -inf over all the other
tokens. On a 100k-250k vocabulary that is most of the sampler's time.

Here the k best are found by a radix selection: every threadgroup ranks one chunk of the
vocabulary in threadgroup memory and keeps the chunk's k best, then one threadgroup per row
selects the row's k best among those candidates, and a last pass writes the log-probs with every
other token set to -inf. The order is the one MLX's stable sort gives ``apply_top_k``: by value
(-0.0 equal to 0.0, NaN below everything), ties to the lower token id. So the filtered log-probs
are bit for bit the ones ``apply_top_k`` returns, and every other step of the chain (top-p, min-p,
the categorical draw and its use of the random state) is mlx-lm's own: the sampled tokens are the
same.

The kernels need a Metal GPU and float32 log-probs (exo's batch step normalises in float32). In
any other case, or for a top-k above ``_MAX_TOP_K`` or not below the vocabulary size, mlx-lm's
``apply_top_k`` runs, with its own errors for invalid values.

Top-p: ``apply_top_p`` sorts the log-probs with ``mx.sort`` (MLX's multi-block merge sort), sums
the sorted probabilities with a cumulative sum and cuts where the mass above a token reaches top_p
of the total. The cut depends on the float32 sums in the sorted order, so only a full sort keeps it
exact: ``sorted_top_p`` runs the same operations on the output of a radix sort, which sorts the
same values (see radix_sort.py), and gives the same filtered log-probs bit for bit.

Logits processors: mlx-lm's repetition, presence and frequency penalties, without the ones that
can't change the logits (a repetition penalty of 1.0 multiplies or divides them by 1.0), and
each marked with how many of the most recent tokens it reads, so that the batch step converts
only those to an array instead of the whole history, every step.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final, final

import mlx.core as mx
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
)
from mlx_lm.sample_utils import (
    make_logits_processors as mlx_make_logits_processors,
)
from mlx_lm.sample_utils import (
    make_sampler as mlx_make_sampler,
)

from exo.worker.engines.mlx.radix_sort import can_sort_rows, sort_rows

Sampler = Callable[[mx.array], mx.array]
LogitsProcessor = Callable[[mx.array, mx.array], mx.array]

# Each threadgroup of the candidate kernel ranks one chunk of the vocabulary in threadgroup
# memory and keeps its k best: the row's k best are among them.
_CHUNK: Final = 4096
_CHUNK_THREADS: Final = 256
_MAX_TOP_K: Final = 1024
_SELECT_THREADS: Final = 1024
_MASK_THREADS: Final = 256
# Below this many tokens MLX sorts a row in one threadgroup, faster than the radix sort.
_MIN_SORTED_TOP_P: Final = 4096
# mlx-lm's presence_context_size and frequency_context_size defaults.
_MLX_PENALTY_CONTEXT: Final = 20

_HEADER = """
// The order mlx-lm's apply_top_k ranks log-probs in: MLX's stable GPU sort of -logprobs puts
// larger values first, -0.0 level with 0.0 and NaN last. Larger key = ranked first.
inline uint order_key(float f) {
  if (metal::isnan(f)) {
    return 0u;
  }
  uint u = as_type<uint>(f);
  if (u == 0x80000000u) {
    u = 0u;
  }
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One radix step of selecting the remaining-th largest key: hist holds the counts of the 256
// values of the current digit among the keys that match the digits chosen so far. Writes the
// digit holding it, its rank inside that digit and the digit's count to result. NT >= 256.
template <int NT>
inline void select_digit(
    threadgroup atomic_uint* hist,
    threadgroup uint* sg_totals,
    threadgroup uint* result,
    uint remaining,
    uint tid,
    uint lane,
    uint sg) {
  uint h = 0;
  if (tid < 256) {
    h = atomic_load_explicit(&hist[255 - tid], memory_order_relaxed);
  }
  uint incl = simd_prefix_inclusive_sum(h);
  if (lane == 31 && sg < 8) {
    sg_totals[sg] = incl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0;
    for (int s = 0; s < 8; ++s) {
      uint t = sg_totals[s];
      sg_totals[s] = acc;
      acc += t;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 256) {
    incl += sg_totals[sg];
    uint excl = incl - h;
    if (excl < remaining && incl >= remaining) {
      result[0] = 255 - tid;
      result[1] = remaining - excl;
      result[2] = h;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# One threadgroup per (chunk, row): writes the chunk's K best as (token id, log-prob). A chunk
# of K tokens or fewer is copied whole and padded with id 0xFFFFFFFF and NaN, which rank below
# every real token.
_CANDIDATES_SOURCE = """
  constexpr int NT = CHUNK_THREADS;
  constexpr int PER_THREAD = CHUNK / NT;
  threadgroup uint keys[CHUNK];
  threadgroup atomic_uint hist[256];
  threadgroup uint sg_totals[NT / 32];
  threadgroup uint result[3];
  threadgroup atomic_uint slot;

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint chunk = threadgroup_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  const uint start = chunk * CHUNK;
  const uint len = min(uint(CHUNK), uint(V) - start);
  auto xr = x + size_t(row) * V;
  auto out_idx = cand_idx + (size_t(row) * NCHUNKS + chunk) * K;
  auto out_val = cand_val + (size_t(row) * NCHUNKS + chunk) * K;

  if (len <= uint(K)) {
    for (uint i = tid; i < uint(K); i += NT) {
      if (i < len) {
        out_idx[i] = start + i;
        out_val[i] = xr[start + i];
      } else {
        out_idx[i] = 0xFFFFFFFFu;
        out_val[i] = as_type<float>(0x7FC00000u);
      }
    }
    return;
  }

  for (uint i = tid; i < len; i += NT) {
    keys[i] = order_key(xr[start + i]);
  }
  if (tid == 0) {
    atomic_store_explicit(&slot, 0u, memory_order_relaxed);
  }

  uint prefix = 0;
  uint prefix_mask = 0;
  uint remaining = K;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = tid; i < 256; i += NT) {
      atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < len; i += NT) {
      uint k = keys[i];
      if ((k & prefix_mask) == prefix) {
        atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    select_digit<NT>(hist, sg_totals, result, remaining, tid, lane, sg);
    prefix |= result[0] << shift;
    prefix_mask |= 255u << shift;
    remaining = result[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint kth = prefix;
  const uint ties_kept = remaining;

  // Ties with the K-th key are kept in token order: each thread owns PER_THREAD consecutive
  // keys, so a prefix sum of the per-thread tie counts ranks every tie.
  const uint base = tid * PER_THREAD;
  uint my_ties = 0;
  for (int j = 0; j < PER_THREAD; ++j) {
    uint i = base + j;
    if (i < len && keys[i] == kth) {
      my_ties++;
    }
  }
  uint incl = simd_prefix_inclusive_sum(my_ties);
  if (lane == 31) {
    sg_totals[sg] = incl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0;
    for (int s = 0; s < NT / 32; ++s) {
      uint t = sg_totals[s];
      sg_totals[s] = acc;
      acc += t;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rank = incl - my_ties + sg_totals[sg];
  for (int j = 0; j < PER_THREAD; ++j) {
    uint i = base + j;
    if (i < len) {
      uint k = keys[i];
      bool keep = k > kth || (k == kth && rank < ties_kept);
      if (k == kth) {
        rank++;
      }
      if (keep) {
        uint s = atomic_fetch_add_explicit(&slot, 1u, memory_order_relaxed);
        out_idx[s] = start + i;
        out_val[s] = xr[start + i];
      }
    }
  }
"""

# One threadgroup per row: the row's K-th best key among the chunks' candidates, and the
# highest token id kept among the tokens tied with it.
_SELECT_SOURCE = """
  constexpr int NT = SELECT_THREADS;
  threadgroup atomic_uint hist[256];
  threadgroup uint sg_totals[NT / 32];
  threadgroup uint result[3];

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint row = threadgroup_position_in_grid.y;
  auto ci = cand_idx + size_t(row) * M;
  auto cv = cand_val + size_t(row) * M;

  uint prefix = 0;
  uint prefix_mask = 0;
  uint remaining = K;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) {
      atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(M); i += NT) {
      uint k = order_key(cv[i]);
      if ((k & prefix_mask) == prefix) {
        atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    select_digit<NT>(hist, sg_totals, result, remaining, tid, lane, sg);
    prefix |= result[0] << shift;
    prefix_mask |= 255u << shift;
    remaining = result[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint kth = prefix;
  const uint ties_kept = remaining;

  // The kept ties are the ones with the lowest token ids, and all of them are candidates (a
  // chunk drops a tie only after keeping its lower ids). Select the ties_kept-th largest of ~id
  // among the tied candidates: the highest kept id.
  uint max_tie_id;
  {
    uint id_prefix = 0;
    uint id_mask = 0;
    uint id_remaining = ties_kept;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < uint(M); i += NT) {
        if (order_key(cv[i]) == kth) {
          uint k = ~ci[i];
          if ((k & id_mask) == id_prefix) {
            atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      select_digit<NT>(hist, sg_totals, result, id_remaining, tid, lane, sg);
      id_prefix |= result[0] << shift;
      id_mask |= 255u << shift;
      id_remaining = result[1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_tie_id = ~id_prefix;
  }
  if (tid == 0) {
    cut[2 * row] = kth;
    cut[2 * row + 1] = max_tie_id;
  }
"""

# One thread per log-prob: keep it if it ranks among the row's K best, else -inf.
_MASK_SOURCE = """
  const uint i = thread_position_in_grid.x;
  const uint row = thread_position_in_grid.y;
  if (i >= uint(V)) {
    return;
  }
  const size_t at = size_t(row) * V + i;
  const float v = x[at];
  const uint k = order_key(v);
  const uint kth = cut[2 * row];
  const bool keep = k > kth || (k == kth && i <= cut[2 * row + 1]);
  out[at] = keep ? v : -INFINITY;
"""


@cache
def _kernels() -> tuple[
    "mx.fast.MetalKernel", "mx.fast.MetalKernel", "mx.fast.MetalKernel"
]:
    candidates = mx.fast.metal_kernel(
        name="exo_top_k_candidates",
        input_names=["x"],
        output_names=["cand_idx", "cand_val"],
        source=_CANDIDATES_SOURCE,
        header=_HEADER,
    )
    select = mx.fast.metal_kernel(
        name="exo_top_k_select",
        input_names=["cand_idx", "cand_val"],
        output_names=["cut"],
        source=_SELECT_SOURCE,
        header=_HEADER,
    )
    mask = mx.fast.metal_kernel(
        name="exo_top_k_mask",
        input_names=["x", "cut"],
        output_names=["out"],
        source=_MASK_SOURCE,
        header=_HEADER,
    )
    return candidates, select, mask


def can_select_top_k(logprobs: mx.array, top_k: int) -> bool:
    return (
        mx.metal.is_available()
        and logprobs.dtype == mx.float32
        and logprobs.ndim == 2
        and 0 < top_k <= _MAX_TOP_K
        and top_k < logprobs.shape[-1]
    )


def select_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """apply_top_k(logprobs, top_k), bit for bit, for [rows, vocab] float32 log-probs when
    can_select_top_k(logprobs, top_k)."""
    rows, vocab = logprobs.shape
    candidates, select, mask = _kernels()
    n_chunks = (vocab + _CHUNK - 1) // _CHUNK
    cand_idx, cand_val = candidates(
        inputs=[logprobs],
        output_shapes=[(rows, n_chunks * top_k), (rows, n_chunks * top_k)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(n_chunks * _CHUNK_THREADS, rows, 1),
        threadgroup=(_CHUNK_THREADS, 1, 1),
        template=[
            ("V", vocab),
            ("K", top_k),
            ("NCHUNKS", n_chunks),
            ("CHUNK", _CHUNK),
            ("CHUNK_THREADS", _CHUNK_THREADS),
        ],
    )
    (cut,) = select(
        inputs=[cand_idx, cand_val],
        output_shapes=[(rows, 2)],
        output_dtypes=[mx.uint32],
        grid=(_SELECT_THREADS, rows, 1),
        threadgroup=(_SELECT_THREADS, 1, 1),
        template=[
            ("K", top_k),
            ("M", n_chunks * top_k),
            ("SELECT_THREADS", _SELECT_THREADS),
        ],
    )
    (out,) = mask(
        inputs=[logprobs, cut],
        output_shapes=[(rows, vocab)],
        output_dtypes=[mx.float32],
        grid=(vocab, rows, 1),
        threadgroup=(_MASK_THREADS, 1, 1),
        template=[("V", vocab)],
    )
    return out


@mx.compile
def _top_p_cut(logprobs: mx.array, sorted_logprobs: mx.array, top_p: float) -> mx.array:
    # mlx_lm.sample_utils.apply_top_p after its sort, operation for operation.
    sorted_probs = mx.exp(sorted_logprobs.astype(mx.float32))
    mass_above = mx.cumsum(sorted_probs, axis=-1, reverse=True, inclusive=False)
    total_mass = mass_above[..., :1] + sorted_probs[..., :1]
    num_dropped = (mass_above >= top_p * total_mass).sum(axis=-1, keepdims=True)
    threshold = mx.take_along_axis(sorted_logprobs, num_dropped, axis=-1)
    return mx.where(logprobs < threshold, -float("inf"), logprobs)


def sorted_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """apply_top_p(logprobs, top_p), bit for bit, with the radix sort in place of mx.sort, for
    [rows, vocab] float32 log-probs when can_sort_rows(logprobs).

    The sorted log-probs differ from mx.sort's at most in the order of -0.0 and 0.0 and in NaN
    bit patterns. Neither changes the result: exp(-0.0) == exp(0.0), x < -0.0 exactly when
    x < 0.0, and NaN compares false whatever its bits."""
    return _top_p_cut(logprobs, sort_rows(logprobs), top_p)


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
) -> Sampler:
    """mlx_lm.sample_utils.make_sampler(temp, top_p, min_p, top_k=top_k): the same chain in
    the same order, with select_top_k in place of apply_top_k where it applies."""
    if temp == 0 or (top_k <= 0 and not 0 < top_p < 1.0):
        return mlx_make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)

    def sampler(logprobs: mx.array) -> mx.array:
        if 0 < top_p < 1.0:
            if can_sort_rows(logprobs) and logprobs.shape[-1] > _MIN_SORTED_TOP_P:
                logprobs = sorted_top_p(logprobs, top_p)
            else:
                logprobs = apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = apply_min_p(logprobs, min_p, 1)
        if top_k > 0:
            if can_select_top_k(logprobs, top_k):
                logprobs = select_top_k(logprobs, top_k)
            else:
                logprobs = apply_top_k(logprobs, top_k)
        return categorical_sampling(logprobs, temp)

    return sampler


@final
@dataclass(frozen=True)
class RecentTokensProcessor:
    """A logits processor that reads at most the last ``window`` tokens of the history it is
    given, so that it returns the same logits given only those."""

    processor: LogitsProcessor
    window: int

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return self.processor(tokens, logits)


def history_window(processors: Sequence[LogitsProcessor]) -> int | None:
    """How many of the most recent tokens these processors read, or None for all of them."""
    windows = [
        p.window if isinstance(p, RecentTokensProcessor) else None for p in processors
    ]
    if any(w is None for w in windows):
        return None
    return max((w for w in windows if w is not None), default=0)


def recent_tokens(tokens: list[int], window: int | None) -> list[int]:
    if window is None or window >= len(tokens):
        return tokens
    return tokens[len(tokens) - window :]


def make_logits_processors(
    repetition_penalty: float | None,
    repetition_context_size: int,
    presence_penalty: float | None,
    frequency_penalty: float | None,
) -> list[LogitsProcessor]:
    """mlx_lm.sample_utils.make_logits_processors with these arguments (presence and frequency
    penalties over mlx-lm's default 20 tokens), returning the same logits.

    A repetition penalty of 1.0 is left out: x * 1.0 and x / 1.0 are x for every float, so its
    gather and scatter return the logits unchanged. mlx-lm already leaves out penalties of 0."""
    penalties: list[tuple[float | None, int, str]] = [
        (
            None if repetition_penalty == 1.0 else repetition_penalty,
            repetition_context_size,
            "repetition",
        ),
        (presence_penalty, _MLX_PENALTY_CONTEXT, "presence"),
        (frequency_penalty, _MLX_PENALTY_CONTEXT, "frequency"),
    ]
    processors: list[LogitsProcessor] = []
    for penalty, context_size, kind in penalties:
        built = mlx_make_logits_processors(
            repetition_penalty=penalty if kind == "repetition" else None,
            repetition_context_size=context_size,
            presence_penalty=penalty if kind == "presence" else None,
            frequency_penalty=penalty if kind == "frequency" else None,
        )
        # Each reads tokens[-context_size:]; a context size of 0 or less reads them all.
        processors.extend(
            RecentTokensProcessor(p, context_size) if context_size > 0 else p
            for p in built
        )
    return processors
