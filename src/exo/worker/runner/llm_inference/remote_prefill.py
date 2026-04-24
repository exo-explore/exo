import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.disaggregated.mlx_adapter import serialize_mlx_cache_to_payload
from exo.disaggregated.server import PrefillPayloadLookup
from exo.shared.types.events import RemotePrefillReady
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import RemotePrefill
from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import prefill as mlx_prefill
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
)
from exo.worker.runner.bootstrap import logger


def _wire_dtype_from_model(model: Model) -> str:
    dtype = getattr(model, "dtype", None)
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.float32:
        return "float32"
    return "bfloat16"


def serve_remote_prefill(
    task: RemotePrefill,
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    payload_lookup: PrefillPayloadLookup,
    endpoint: str,
) -> RemotePrefillReady:
    request_id = str(task.task_id)

    prompt_str = apply_chat_template(tokenizer, task.task_params)
    prompt_tokens = encode_prompt(tokenizer, prompt_str)
    prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
    logger.info(
        f"RemotePrefill: serving {int(prompt_tokens.shape[0])} tokens "
        f"for task_id={task.task_id} request_id={request_id}"
    )

    cache = make_kv_cache(model)
    sampler = make_sampler(temp=1.0)
    _ = mlx_prefill(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        prompt_tokens=prompt_tokens,
        cache=cache,
        group=group,
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    dtype = _wire_dtype_from_model(model)
    payload = serialize_mlx_cache_to_payload(
        cache,
        dtype=dtype,
        model_id=str(task.model_id),
        request_id=request_id,
        start_pos=task.start_pos,
    )
    payload_lookup.register(request_id, payload)

    num_tokens = int(prompt_tokens.shape[0]) - task.start_pos
    return RemotePrefillReady(
        task_id=task.task_id,
        endpoint=endpoint,
        request_id=request_id,
        num_tokens=max(num_tokens, 0),
    )
