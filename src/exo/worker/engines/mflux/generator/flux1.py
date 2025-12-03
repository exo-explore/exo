from typing import Literal
import mlx.core as mx
from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.array_util import ArrayUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from PIL import Image
from tqdm import tqdm

from exo.shared.types.api import ImageGenerationTaskParams


def _generate_image(model: Flux1, settings: Config, prompt: str, seed: int):
    # 0. Create a new runtime config based on the model type and input parameters
    config = RuntimeConfig(settings, model.model_config)
    time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))

    # 1. Create the initial latents
    latents = LatentCreator.create_for_txt2img_or_img2img(
        seed=seed,
        height=config.height,
        width=config.width,
        img2img=Img2Img(
            vae=model.vae,
            latent_creator=FluxLatentCreator,
            image_path=config.image_path,
            sigmas=config.scheduler.sigmas,
            init_time_step=config.init_time_step,
        ),
    )

    # 2. Encode the prompt
    prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
        prompt=prompt,
        prompt_cache=model.prompt_cache,
        t5_tokenizer=model.t5_tokenizer,
        clip_tokenizer=model.clip_tokenizer,
        t5_text_encoder=model.t5_text_encoder,
        clip_text_encoder=model.clip_text_encoder,
    )

    # (Optional) Call subscribers for beginning of loop
    Callbacks.before_loop(
        seed=seed,
        prompt=prompt,
        latents=latents,
        config=config,
    )

    for t in time_steps:
        try:
            # Scale model input if needed by the scheduler
            latents = config.scheduler.scale_model_input(latents, t)

            # 3.t Predict the noise
            noise = model.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )

            # 4.t Take one denoise step
            latents = config.scheduler.step(
                model_output=noise,
                timestep=t,
                sample=latents,
            )

            # (Optional) Call subscribers in-loop
            Callbacks.in_loop(
                t=t,
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )

            # (Optional) Evaluate to enable progress tracking
            mx.eval(latents)

        except KeyboardInterrupt:  # noqa: PERF203
            Callbacks.interruption(
                t=t,
                seed=seed,
                prompt=prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )
            raise StopImageGenerationException(
                f"Stopping image generation at step {t + 1}/{len(time_steps)}"
            ) from None

    # (Optional) Call subscribers after loop
    Callbacks.after_loop(
        seed=seed,
        prompt=prompt,
        latents=latents,
        config=config,
    )

    # 7. Decode the latent array and return the image
    latents = ArrayUtil.unpack_latents(
        latents=latents, height=config.height, width=config.width
    )
    decoded = model.vae.decode(latents)
    return ImageUtil.to_image(
        decoded_latents=decoded,
        config=config,
        seed=seed,
        prompt=prompt,
        quantization=model.bits,
        lora_paths=model.lora_paths,
        lora_scales=model.lora_scales,
        image_path=config.image_path,
        image_strength=config.image_strength,
        generation_time=time_steps.format_dict["elapsed"],
    )


def generate_image(
    model: Flux1,
    prompt: str,
    height: int,
    width: int,
    quality: Literal["low", "medium", "high"],
    seed: int,
) -> Image.Image:
    # Parse parameters

    # TODO: Flux1 only
    steps = 2
    if quality == "low":
        steps = 1
    elif quality == "high":
        steps = 4

    seed = 2  # TODO: not in OAI API?
    config = Config(num_inference_steps=steps, height=height, width=width)

    image = _generate_image(model=model, settings=config, prompt=prompt, seed=seed)
    return image.image
