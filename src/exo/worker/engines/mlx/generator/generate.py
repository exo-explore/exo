from typing import Any, Callable, Generator, cast, get_args, Protocol, Union, runtime_checkable
import base64
import io

try:
    from PIL import Image
except ImportError:
    Image = None

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

# from exo.engines.mlx.cache import KVPrefixCache
from exo.shared.types.api import (
    BenchChatCompletionTaskParams,
    ChatCompletionMessage,
    FinishReason,
    GenerationStats,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    make_kv_cache,
    mx_barrier,
    extract_images_from_messages,
)
from exo.worker.runner.bootstrap import logger
from mlx_vlm import generate

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[KVCache | Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if (
            hasattr(c, "to_quantized") and c.offset >= quantized_kv_start  # type: ignore
        ):
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)

@runtime_checkable
class TextTokenizer(Protocol):
    """Protocol for text-only tokenizers"""
    eos_token_id: int

@runtime_checkable
class VisionProcessor(Protocol):
    """Protocol for vision model processors that wrap a tokenizer"""
    tokenizer: TextTokenizer

def extract_tokenizer(tokenizer_or_processor: Union[TextTokenizer, VisionProcessor]) -> tuple[bool, TextTokenizer]:
    """Extract the text tokenizer from either a tokenizer or vision processor.
    
    Returns:
        tuple: (is_vision_processor, actual_tokenizer)
    """
    # More robust check: use isinstance if VisionProcessor is available
    # Otherwise fall back to attribute checking
    try:
        from mlx_vlm.utils import VisionProcessor
        is_vision_processor = isinstance(tokenizer_or_processor, VisionProcessor)
    except ImportError:
        # Fallback to attribute checking if VisionProcessor not available
        is_vision_processor = (
            not hasattr(tokenizer_or_processor, "eos_token_id") 
            and hasattr(tokenizer_or_processor, "tokenizer")
        )
    
    actual_tokenizer = tokenizer_or_processor.tokenizer if is_vision_processor else tokenizer_or_processor
    return is_vision_processor, actual_tokenizer


def warmup_inference(
    model: Model,
    tokenizer: Union[TextTokenizer, VisionProcessor],
    sampler: Callable[[mx.array], mx.array],
) -> int:
    """Warm up the inference engine with a simple generation task."""
    content = "Prompt to warm up the inference engine. Repeat this."

    messages = [
        ChatCompletionMessage(
            role="user",
            content=content
        )
    ]

    # If vision model, add a dummy image to warm up the vision encoder
    is_vision_processor, actual_tokenizer = extract_tokenizer(tokenizer)
    
    if is_vision_processor and Image is not None:
        try:
            # Create a small black image
            img = Image.new("RGB", (32, 32), color="black")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_str}"
            
            # Update to multimodal message
            messages = [
                ChatCompletionMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]
            logger.info("Added dummy image to vision model warmup")
        except Exception as e:
            logger.warning(f"Failed to create dummy image for warmup: {e}")
            raise

    warmup_task = ChatCompletionTaskParams(
        model="",
        messages=messages,
        max_tokens=50,
    )
    
    logger.info("Generating warmup tokens")
    
    tokens_generated = 0
    
    # Use the unified mlx_generate function for both text and vision models
    for _ in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        task=warmup_task,
    ):
        tokens_generated += 1

    
    
    logger.info("Generated ALL warmup tokens")

    # TODO: Do we want an mx_barrier?
    #  At least this version is actively incorrect, as it should use mx_barrier(group)
    mx_barrier()
    
    return tokens_generated


def mlx_generate(
    model: Model,
    tokenizer: Union[TextTokenizer, VisionProcessor],
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation

    mx.reset_peak_memory()
    is_bench: bool = isinstance(task, BenchChatCompletionTaskParams)

    logger.info(f"task_params: {task}")
    
    # Check if this is a vision model processor
    is_vision_processor, actual_tokenizer = extract_tokenizer(tokenizer)

    # Route to vision generator if it's a vision model processor
    if is_vision_processor:
        logger.info(f"Routing to vision generator for model {task.model}")
        
        # Extract images and prompt
        _, images = extract_images_from_messages(task.messages)
        prompt = apply_chat_template(tokenizer, task)
        
        if not images:
            logger.info("No images found in messages for vision model. Proceeding with text-only inference.")
        else:
            logger.info(f"Generating with prompt: {prompt[:100]}... and {len(images)} image(s)")
        
        # Generate response
        max_tokens = task.max_tokens or MAX_TOKENS
        temperature = task.temperature or 0.7
        
        # try:
        output = generate(
            model,
            tokenizer,
            prompt,
            images[0] if images else None,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        
        text_output = output.text
        
        # Encode the text to get token IDs using the processor's tokenizer
        token_ids = tokenizer.tokenizer.encode(text_output)
        
        # Detokenize each token individually
        for idx, token_id in enumerate(token_ids):
            token_text = tokenizer.tokenizer.decode([token_id])
            
            yield GenerationResponse(
                text=token_text,
                token=idx,
                finish_reason=None,
            )
        
        # Final response with finish reason
        stats: GenerationStats | None = None
        if hasattr(output, "prompt_tps"):
            stats = GenerationStats(
                prompt_tps=float(output.prompt_tps),
                generation_tps=float(output.generation_tps),
                prompt_tokens=int(output.prompt_tokens),
                generation_tokens=int(output.generation_tokens),
                peak_memory_usage=Memory.from_gb(output.peak_memory),
            )

        yield GenerationResponse(
            text="",
            token=len(token_ids),
            finish_reason="stop",
            stats=stats,
        )

        return
    
    # Standard text-only generation for non-vision models
    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    caches = make_kv_cache(model=model)

    max_tokens = task.max_tokens or MAX_TOKENS
    for out in stream_generate(
        model=model,
        tokenizer=actual_tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=caches,
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info(out.text)


        stats: GenerationStats | None = None
        if out.finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=float(out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )

        if out.finish_reason not in get_args(FinishReason):
                # We don't throw here as this failure case is really not all that bad
                # Just log the error and move on
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=cast(FinishReason | None, out.finish_reason),
            stats=stats,
        )

        if out.finish_reason is not None:
            break
