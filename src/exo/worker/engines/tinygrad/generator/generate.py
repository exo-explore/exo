import struct
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from tinygrad import TinyJit
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig
from exo.shared.models.model_cards import ModelId
from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model
from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.tinygrad.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_LOGPROBS,
    DEFAULT_TOP_P,
)

from ..cache import KVCache
from ..forward import forward_pass
from ..sampling import sample_token
from ..weight_loader import TransformerWeights


@dataclass
class _JitState:
    """Persistent decode state reused across requests."""
    jit_decode: Callable[..., tuple[Tensor, ...]]
    cache: KVCache
    input_buffer: Tensor
    position_buffer: Tensor

_jit_registry: dict[int, _JitState] = {}

def cleanup_jit_state() -> None:
    """Called by engine cleanup to free all JIT state."""
    _jit_registry.clear()


def _build_jit_decode(
    weights: TransformerWeights,
    cache: KVCache,
) -> Callable[..., tuple[Tensor, ...]]:
    num_layers = len(weights.layers)

    @TinyJit
    def decode(
        input_ids: Tensor, position: Tensor,
        rope_cos_table: Tensor, rope_sin_table: Tensor,
        *cache_kv: Tensor,
    ) -> tuple[Tensor, ...]:
        for i in range(num_layers):
            cache._keys[i] = cache_kv[i]
            cache._values[i] = cache_kv[num_layers + i]

        logits, _ = forward_pass(
            weights, input_ids, cache,
            position_offset=position,
            rope_cos=rope_cos_table, rope_sin=rope_sin_table,
        )

        # Realize everything at once — JIT captures one fused kernel schedule
        # instead of 32 fragmented ones.
        logits = logits.realize(*cache._keys, *cache._values)

        return (logits, *cache._keys, *cache._values)

    return decode

def tinygrad_generate(
    model: TransformerWeights,
    tokenizer: Any,  # pyright: ignore[reportAny]
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: Any = None,  # pyright: ignore[reportAny]
    on_prefill_progress: Callable[[int, int], None] | None = None,
    group: None = None,
) -> Generator[GenerationResponse]:
    input_ids = _encode_prompt(tokenizer, prompt)

    max_tokens = task.max_output_tokens or DEFAULT_MAX_TOKENS
    temperature = task.temperature or DEFAULT_TEMPERATURE
    top_p = task.top_p or DEFAULT_TOP_P

    request_logprobs = task.logprobs
    top_logprobs_count = task.top_logprobs or 0

    eos_ids = _get_eos_ids(tokenizer, model.config)
    print(f"[DEBUG] eos_ids={eos_ids}")
    prompt_tokens = len(input_ids)

    model_key = id(model)
    num_layers = len(model.layers)
    state = _jit_registry.get(model_key)

    if state is None:
        # First request: create cache, JIT, and pre-allocate buffers
        config = model.config
        cache = KVCache(
            num_layers=len(model.layers),
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=min(config.max_position_embeddings, 4096),
        )
        # Realize cache tensors — Tensor.zeros() produces lazy/const tensors
        # that TinyJit rejects as inputs. Force device buffer allocation.
        for i in range(len(model.layers)):
            cache._keys[i] = cache._keys[i].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
            cache._values[i] = cache._values[i].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        jit_decode = _build_jit_decode(model, cache)
        input_buffer = Tensor.empty(1, 1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        position_buffer = Tensor.empty(1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        state = _JitState(
            jit_decode=jit_decode,
            cache=cache,
            input_buffer=input_buffer,
            position_buffer=position_buffer,
        )
        _jit_registry[model_key] = state

    cache = state.cache
    jit_decode = state.jit_decode

    # Sequential prefill: process each prompt token through the JIT decode path.
    # Always seq_len=1, reuses the same JIT graph — zero compilation after warmup.
    if not input_ids:
        raise ValueError("Prompt must contain at least one token")

    prefill_start = time.time()
    for i, token_id in enumerate(input_ids):
        state.input_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', token_id))))  # pyright: ignore[reportPrivateUsage]
        state.position_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', i))))  # pyright: ignore[reportPrivateUsage]
        results = jit_decode(
            state.input_buffer, state.position_buffer,
            model.rope_cos, model.rope_sin,
            *cache._keys, *cache._values,
        )
        logits = results[0]
        for j in range(num_layers):
            cache._keys[j] = results[1 + j]
            cache._values[j] = results[1 + num_layers + j]

    prefill_time = time.time() - prefill_start
    prompt_tps = prompt_tokens / max(prefill_time, 1e-9)

    position = prompt_tokens

    # Decode
    generation_start = time.time()
    for token_idx in range(max_tokens):
        result = sample_token(
            logits, temperature=temperature, top_p=top_p,  # pyright: ignore[reportPossiblyUnboundVariable]
            top_logprobs_count=top_logprobs_count,
            request_logprobs=request_logprobs,
        )

        token_text: str = tokenizer.decode([result.token_id])  # pyright: ignore[reportAny]

        is_eos = result.token_id in eos_ids
        if is_eos:
            print(f"[DEBUG] EOS detected: token_id={result.token_id}, text={token_text!r}")
        if "<|eot_id|>" in token_text or "<|end" in token_text:
            print(f"[DEBUG] Special token in text but is_eos={is_eos}, token_id={result.token_id}")
        tokens_generated = token_idx + 1
        elapsed = time.time() - generation_start
        generation_tps = tokens_generated / max(elapsed, 1e-9)

        finish_reason = None
        stats = None
        usage = None

        if is_eos:
            finish_reason = "stop"
        elif token_idx == max_tokens - 1:
            finish_reason = "length"

        if finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=prompt_tps, generation_tps=generation_tps,
                prompt_tokens=prompt_tokens,
                generation_tokens=tokens_generated,
                peak_memory_usage=Memory.from_bytes(0),
            )

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_generated,
                total_tokens=prompt_tokens + tokens_generated,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails(),
            )

        logprob = result.logprob if request_logprobs else None
        top_lps = None
        if request_logprobs and task.top_logprobs:
            top_lps = [
                TopLogprobItem(
                    token=str(tokenizer.decode([tok_id])),  # pyright: ignore[reportAny]
                    logprob=lp,
                    bytes=list(str(tokenizer.decode([tok_id])).encode("utf-8")),  # pyright: ignore[reportAny]
                )
                for tok_id, lp in result.top_logprobs
            ]

        if is_eos:
            token_text = ""

        yield GenerationResponse(
            text=token_text, token=result.token_id,
            logprob=logprob, top_logprobs=top_lps,
            finish_reason=finish_reason, stats=stats, usage=usage,
        )

        if finish_reason is not None:
            break

        state.input_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', result.token_id))))  # pyright: ignore[reportPrivateUsage]
        state.position_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', position))))  # pyright: ignore[reportPrivateUsage]
        results = jit_decode(
            state.input_buffer, state.position_buffer,
            model.rope_cos, model.rope_sin,
            *cache._keys, *cache._values,
        )

        logits = results[0]
        for i in range(num_layers):
            cache._keys[i] = results[1 + i]
            cache._values[i] = results[1 + num_layers + i]
        position += 1

def warmup_inference(model: TransformerWeights, tokenizer: Any, group: None = None) -> int:  # pyright: ignore[reportAny]
    """Run a full generation loop to warm up forward pass, KV cache, and sampling."""
    from exo.shared.tokenizer.chat_template import apply_chat_template
    from exo.shared.types.common import ModelId as CommonModelId
    from exo.shared.types.text_generation import InputMessage

    warmup_task = TextGenerationTaskParams(
        model=CommonModelId("warmup"),
        input=[InputMessage(role="user", content="Time to warm up!")],
        logprobs=True,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
    )

    prompt: str = apply_chat_template(tokenizer, warmup_task)
    tokens_generated = 0

    for _ in tinygrad_generate(model, tokenizer, warmup_task, prompt):
        tokens_generated += 1
        if tokens_generated >= 5:
            break

    return tokens_generated

def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:  # pyright: ignore[reportAny]
    result: Any = tokenizer.encode(prompt)  # pyright: ignore[reportAny]

    return result.ids if hasattr(result, "ids") else result  # pyright: ignore[reportAny]

def _get_eos_ids(tokenizer: Any, config: ModelConfig) -> set[int]:  # pyright: ignore[reportAny]
    eos_ids: set[int] = set()

    model_eos = get_eos_token_ids_for_model(ModelId(config.architecture_spec.name))

    if model_eos:
        eos_ids.update(model_eos)

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]

    if not eos_ids:
        eos_ids.add(2)

    return eos_ids
