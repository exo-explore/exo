from typing import Callable, Generator, Any
import base64
import io
from PIL import Image

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.runner.bootstrap import logger
from exo.worker.engines.mlx.utils_mlx import apply_chat_template


def extract_images_from_messages(
    messages: list[ChatCompletionMessage],
) -> tuple[str, list[Image.Image]]:
    """
    Extract text prompt and images from multimodal messages.
    
    Returns:
        tuple: (text_prompt, list_of_pil_images)
    """
    text_parts = []
    images = []
    
    for message in messages:
        content = message.content
        
        if content is None:
            continue
            
        # Standardize content to a list of parts
        parts = []
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = content
        else:
            # Single object (text or image_url model)
            parts = [content]

        for part in parts:
            # Handle both Pydantic models and dictionaries
            p_type = getattr(part, 'type', part.get('type') if isinstance(part, dict) else None)
            
            if p_type == "text":
                text = getattr(part, 'text', part.get('text') if isinstance(part, dict) else "")
                text_parts.append(text)
            elif p_type == "image_url":
                image_url_obj = getattr(part, 'image_url', part.get('image_url') if isinstance(part, dict) else {})
                url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else getattr(image_url_obj, 'url', "")
                
                if url.startswith("data:image"):
                    try:
                        base64_data = url.split(",", 1)[1]
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append(image)
                        logger.info(f"Successfully extracted image. Current count: {len(images)}")
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")
    
    prompt = " ".join(text_parts)
    logger.info(f"Extraction complete. Total text length: {len(prompt)}, Total images: {len(images)}")
    return prompt, images


def mlx_vlm_generate(
    model: Any,  # Pre-loaded vision model
    processor: Any,  # Pre-loaded processor (acts as tokenizer)
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate responses using MLX vision-language models.
    
    Args:
        model: Pre-loaded vision model from mlx_vlm.load()
        processor: Pre-loaded processor from mlx_vlm.load()
        task: Chat completion task parameters
    
    Yields:
        GenerationResponse objects with generated tokens
    """
    logger.info(f"Generating with vision model")
    
    # Extract images and generate formatted prompt
    _, images = extract_images_from_messages(task.messages)
    prompt = apply_chat_template(processor, task)
    
    if not images:
        logger.info("No images found in messages for vision model. Proceeding with text-only inference.")
    else:
        logger.info(f"Generating with prompt: {prompt[:100]}... and {len(images)} image(s)")
    
    # Generate response
    max_tokens = task.max_tokens or 512
    temperature = task.temperature or 0.7
    
    try:
        # Use mlx_vlm's generate function
        output = generate(
            model,
            processor,
            prompt,
            images[0] if images else None,  # Qwen2-VL takes single image
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        
        # mlx_vlm.generate can return a string or a GenerationResult object
        text_output = output.text if hasattr(output, "text") else output
        
        # Yield it as word-level chunks for better UX
        tokens = text_output.split()
        for idx, token in enumerate(tokens):
            yield GenerationResponse(
                text=token + " ",
                token=idx,
                finish_reason=None,
            )
        
        # Final token with finish reason
        yield GenerationResponse(
            text="",
            token=len(tokens),
            finish_reason="stop",
        )
        
    except Exception as e:
        logger.error(f"Error during vision model generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Yield error message
        yield GenerationResponse(
            text=f"[Error generating response: {str(e)}]",
            token=0,
            finish_reason="stop",
        )
