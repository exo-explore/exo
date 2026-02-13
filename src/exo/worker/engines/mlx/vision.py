"""Vision support for MLX engine.

Model-agnostic vision pipeline for exo. All model-specific values (HF repo,
token IDs, mlx-vlm model type) come from the model card's [vision] section.

The pipeline:
  1. Decode base64 images to PIL
  2. Preprocess via HuggingFace image processor (trust_remote_code for Kimi,
     Qwen2VLImageProcessor for Qwen3-VL, etc.)
  3. Encode through vision tower [+ projector] → (N, hidden_dim) features
  4. Build prompt with correct placeholder token count per image
  5. Splice vision features into LM token embeddings
  6. Prefill KV cache with the spliced embeddings

Only steps 3-6 require model-specific code. Steps 1-2 are generic.
The VisionPipeline class orchestrates the full flow, returning a VisionResult
that generate.py can consume without knowing any vision internals.
"""

import base64
import hashlib
import importlib
import inspect
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from PIL import Image

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import VisionCardConfig
from exo.shared.types.common import ModelId

logger = logging.getLogger(__name__)


def _filter_config(cls, d: dict) -> dict:
    """Filter a dict to only keys accepted by *cls.__init__*."""
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in d.items() if k in valid}


def decode_base64_image(b64_data: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw))


def build_vision_prompt(
    tokenizer,
    chat_template_messages: list[dict],
    n_tokens_per_image: list[int],
    image_token: str,
) -> str:
    """Build a prompt with the correct number of image placeholder tokens.

    Uses the tokenizer's Jinja chat_template to produce the base prompt
    (with 1 placeholder per image), then expands each occurrence to
    the actual vision-token count for that image.

    Args:
        tokenizer: Tokenizer with a chat_template attribute.
        chat_template_messages: Message dicts with 'role' and 'content'.
        n_tokens_per_image: Token counts per image, in order of appearance.
        image_token: The placeholder token string (e.g. "<|media_pad|>").

    Returns:
        Formatted prompt string with expanded pad tokens.
    """
    # apply_chat_template produces 1 placeholder per image
    prompt = tokenizer.apply_chat_template(
        chat_template_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Expand each single placeholder to the correct count
    image_idx = 0
    result = []
    i = 0
    pad_len = len(image_token)
    while i < len(prompt):
        if prompt[i : i + pad_len] == image_token:
            n = (
                n_tokens_per_image[image_idx]
                if image_idx < len(n_tokens_per_image)
                else 1
            )
            result.append(image_token * n)
            image_idx += 1
            i += pad_len
        else:
            result.append(prompt[i])
            i += 1

    return "".join(result)


# ──────────────────────────────────────────────────────────────────────
# VisionResult — returned by VisionPipeline.process()
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MediaRegion:
    """Describes a contiguous run of media tokens (image or video frame).

    Used by KVPrefixCache to detect when a cached KV entry's image content
    doesn't match a new query — even though the token IDs (all pad tokens)
    are identical. Generalises to video: each frame is its own region.
    """

    content_hash: str  # sha256 hex digest of the raw media bytes
    start_pos: int  # first pad-token position (inclusive)
    end_pos: int  # last pad-token position (exclusive)


@dataclass
class VisionResult:
    """Complete result of vision preprocessing, ready for generate.py.

    generate.py should only need to look at these fields — it doesn't
    need to know about image preprocessing, media tokens, or embedding
    splicing.
    """

    prompt: str  # Prompt with expanded placeholder tokens
    prompt_tokens: mx.array  # Encoded token IDs
    embeddings: mx.array  # [1, seq_len, hidden_dim] with vision features spliced in
    media_regions: list[MediaRegion]  # One per image/frame, for KV cache validation


# ──────────────────────────────────────────────────────────────────────
# VisionEncoder — loads and runs a specific vision model
# ──────────────────────────────────────────────────────────────────────


class VisionEncoder:
    """Loads a vision encoder + optional projector and processes images.

    Uses HF image processors for all models (trust_remote_code=True).
    Model-specific values come from the model card's [vision] section.
    """

    def __init__(self, config: VisionCardConfig, model_path: Optional[str] = None):
        self._config = config
        self._model_path = Path(
            model_path or str(build_model_path(ModelId(config.weights_repo)))
        )
        self._vision_tower: Optional[nn.Module] = None
        self._projector: Optional[nn.Module] = None
        self._processor = None  # HF image processor
        self._spatial_merge_size: int = 2  # For token counting (Qwen-style)
        self._merge_kernel_size: Optional[list[int]] = (
            None  # For token counting (Kimi-style)
        )
        self._needs_nhwc: bool = False  # Whether vision tower expects NHWC layout
        self._loaded = False

    def _load_config_json(self) -> dict:
        """Load and return config.json from the model directory."""
        path = self._model_path / "config.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _import_mlx_vlm(self, *submodules: str):
        """Import mlx-vlm submodules for the current model_type.

        Returns imported modules in the order requested.
        e.g. _import_mlx_vlm("config", "vision") → (config_mod, vision_mod)
        """
        mt = self._config.model_type
        results = []
        for sub in submodules:
            name = f"mlx_vlm.models.{mt}.{sub}"
            try:
                results.append(importlib.import_module(name))
            except ModuleNotFoundError as e:
                raise ImportError(
                    f"mlx-vlm is required for vision support but '{e.name}' was not found. "
                    "Install it with: pip install mlx-vlm"
                ) from e
        return results[0] if len(results) == 1 else tuple(results)

    def _ensure_loaded(self) -> None:
        """Lazy-load vision encoder weights on first use."""
        if self._loaded:
            return
        self._load_weights()
        self._loaded = True

    # ── Weight loading ───────────────────────────────────────────────

    def _load_weights(self) -> None:
        """Load vision encoder, optional projector, and HF image processor."""
        logger.info(f"Loading vision weights from {self._model_path}")
        config = self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        mt = self._config.model_type
        vision_cfg = config.get("vision_config", {})

        # Import mlx-vlm modules for this model type
        config_mod, vision_mod = self._import_mlx_vlm("config", "vision")
        vision_config_cls = config_mod.VisionConfig
        vision_model_cls = vision_mod.VisionModel

        vision_config = vision_config_cls(
            **_filter_config(vision_config_cls, vision_cfg)
        )
        self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self._vision_tower = vision_model_cls(vision_config)

        # ── Detect and build projector if the model has one ──
        # Look in the mlx-vlm model module for a class containing "Projector"
        try:
            model_mod = self._import_mlx_vlm(mt)
        except ImportError:
            model_mod = None

        projector_cls = None
        if model_mod is not None:
            for attr_name in dir(model_mod):
                obj = getattr(model_mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and "Projector" in attr_name
                ):
                    projector_cls = obj
                    break

        if projector_cls is not None:
            text_config = config_mod.TextConfig(
                **_filter_config(config_mod.TextConfig, config.get("text_config", {}))
            )
            model_config = config_mod.ModelConfig(
                text_config=text_config,
                vision_config=vision_config,
                **_filter_config(
                    config_mod.ModelConfig,
                    {
                        k: v
                        for k, v in config.items()
                        if k not in ("text_config", "vision_config", "model_type")
                    },
                ),
            )
            self._projector = projector_cls(model_config)

        # ── Load weights ──
        processor_repo = self._config.processor_repo
        if processor_repo:
            # Separate vision weights repo (e.g. Kimi K2.5)
            self._load_weights_from_separate_repo()
        else:
            # Bundled vision weights (e.g. Qwen3-VL)
            self._load_weights_from_model_repo()

        # ── Load HF image processor ──
        from transformers import AutoImageProcessor

        repo = processor_repo or str(self._model_path)
        self._processor = AutoImageProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
        )
        if processor_repo:
            self._merge_kernel_size = vision_cfg.get("merge_kernel_size", [2, 2])
            self._needs_nhwc = True
        logger.info(f"HF image processor loaded from {repo}")

    def _load_weights_from_separate_repo(self) -> None:
        """Load vision + projector weights from separate repo (PyTorch safetensors)."""
        from safetensors import safe_open

        safetensors_files = list(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)
                    np_tensor = tensor.float().numpy()
                    # Use bfloat16 to match the LM's native dtype.
                    # The original weights are bfloat16; using float16 here
                    # causes mx.where to promote to float32 when mixing
                    # with the LM's bfloat16 embeddings → 10x slowdown.
                    weights[key] = mx.array(np_tensor, dtype=mx.bfloat16)

        # Separate and remap vision_tower and projector weights
        vision_weights = {}
        projector_weights = {}
        for key, val in weights.items():
            if key.startswith("vision_tower."):
                short_key = key[len("vision_tower.") :]
                if short_key.startswith("encoder."):
                    short_key = short_key[len("encoder.") :]
                # PyTorch → MLX naming: blocks.N.wqkv → blocks.N.attn.wqkv
                m = re.match(r"^(blocks\.\d+)\.(wqkv|wo)\.(weight|bias)$", short_key)
                if m:
                    short_key = f"{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
                # Conv weights: PyTorch OIHW → MLX OHWI
                if short_key == "patch_embed.proj.weight" and val.ndim == 4:
                    val = val.transpose(0, 2, 3, 1)
                vision_weights[short_key] = val
            elif key.startswith(("mm_projector.", "multi_modal_projector.")):
                if key.startswith("multi_modal_projector."):
                    short_key = key[len("multi_modal_projector.") :]
                    if short_key.startswith("mm_projector."):
                        short_key = short_key[len("mm_projector.") :]
                else:
                    short_key = key[len("mm_projector.") :]
                short_key = short_key.replace("proj.0.", "linear_1.").replace(
                    "proj.2.", "linear_2."
                )
                projector_weights[short_key] = val

        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            self._projector.load_weights(list(projector_weights.items()))
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def _load_weights_from_model_repo(self) -> None:
        """Load vision weights bundled in the main model repo (MLX format)."""
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        # Try both prefixes: MLX-community uses "vision_tower.*", HF uses "model.visual.*"
        vision_prefixes = ["vision_tower.", "model.visual."]
        vision_weights: dict[str, mx.array] = {}
        found_raw_prefix = False
        for sf_path in safetensors_files:
            file_weights = mx.load(str(sf_path))
            for key, val in file_weights.items():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        short_key = key[len(prefix) :]
                        vision_weights[short_key] = val
                        if prefix == "model.visual.":
                            found_raw_prefix = True
                        break

        if not vision_weights:
            raise ValueError(
                f"No vision weights found with prefixes {vision_prefixes} in {self._model_path}. "
                "Ensure the model repo contains bundled vision weights."
            )

        if found_raw_prefix and hasattr(self._vision_tower, "sanitize"):
            vision_weights = self._vision_tower.sanitize(vision_weights)

        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())
        logger.info(f"Vision encoder loaded: {n_vision / 1e6:.1f}M params")

    # ── Image encoding ───────────────────────────────────────────────

    def encode_images(self, images: list[str]) -> tuple[mx.array, list[int]]:
        """Encode base64 images to vision features.

        Args:
            images: List of base64-encoded image strings.

        Returns:
            image_features: [total_tokens, hidden_dim] — concatenated
                vision embeddings for all images.
            n_tokens_per_image: List of token counts per image.
        """
        self._ensure_loaded()
        assert self._vision_tower is not None
        assert self._processor is not None

        pil_images = [decode_base64_image(b64) for b64 in images]

        # ── Preprocess with HF image processor ──
        if self._config.processor_repo:
            # Kimi-style: processor expects list of dicts
            processed = self._processor.preprocess(
                [{"type": "image", "image": img} for img in pil_images],
                return_tensors="np",
            )
            pixel_values = mx.array(processed["pixel_values"])
            grid_thw = mx.array(processed["grid_thws"])
            # Token count via merge kernel
            merge_length = int(np.prod(self._merge_kernel_size))
            n_tokens_per_image = [
                int(mx.prod(grid_thw[i]).item()) // merge_length
                for i in range(grid_thw.shape[0])
            ]
        else:
            # Standard HF image processor (Qwen2VLImageProcessor API)
            processed = self._processor(
                images=pil_images,
                return_tensors="np",
            )
            pixel_values = mx.array(processed["pixel_values"])
            grid_thw = mx.array(processed["image_grid_thw"])
            # Token count via spatial merge
            merge_unit = self._spatial_merge_size**2
            n_tokens_per_image = [
                int(
                    grid_thw[i, 0].item()
                    * grid_thw[i, 1].item()
                    * grid_thw[i, 2].item()
                )
                // merge_unit
                for i in range(grid_thw.shape[0])
            ]

        # ── Run vision tower ──
        if self._needs_nhwc:
            # Kimi expects NHWC layout: [patches, C, pH, pW] → [patches, pH, pW, C]
            # Kimi VisionModel expects grid as [n, 2] (h, w only) — strip temporal dim
            grid_hw = grid_thw[:, 1:] if grid_thw.shape[-1] == 3 else grid_thw
            hidden_states = self._vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                output_hidden_states=True,
                grid_thw=grid_hw,
            )
        else:
            # Qwen-style: returns (hidden_states, deepstack_features)
            result = self._vision_tower(pixel_values, grid_thw)
            hidden_states = result[0] if isinstance(result, tuple) else result

        # ── Run projector if present ──
        if self._projector is not None:
            image_features = self._projector(hidden_states)
        else:
            image_features = hidden_states

        return image_features, n_tokens_per_image


# ──────────────────────────────────────────────────────────────────────
# Vision-aware prefill — bypasses stream_generate to splice embeddings
# ──────────────────────────────────────────────────────────────────────


def _get_inner_model(model: nn.Module) -> nn.Module:
    """Navigate to the inner transformer model (the one with embed_tokens + layers).

    Known patterns:
      - DeepSeek V3 / Kimi:  model.model  → DeepseekV3Model
      - Qwen3-VL:            model.language_model.model → Qwen3Model
    """
    for candidate in (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "embed_tokens"):
            return candidate

    raise ValueError(
        f"Could not find inner transformer (embed_tokens) in {type(model).__name__}. "
        "Add a new pattern to _get_inner_model() for this architecture."
    )


def vision_prefill(
    model: nn.Module,
    input_embeddings: mx.array,
    cache: list,
) -> tuple[float, int]:
    """Prefill KV cache using pre-computed vision-augmented embeddings.

    Instead of using stream_generate (which doesn't support
    input_embeddings for all model types), we directly run the
    transformer layers with the pre-computed embeddings.

    Args:
        model: The language model (e.g., DeepSeek V3 Model).
        input_embeddings: [1, seq_len, hidden_dim] — text embeddings
                         with vision features already spliced in.
        cache: KV cache list to fill.

    Returns:
        (tokens_per_sec, num_tokens) — prefill speed and prompt length.
    """
    import time

    num_tokens = input_embeddings.shape[1]
    logger.info(f"Vision prefill: {num_tokens} tokens")
    start = time.perf_counter()

    # Access the inner transformer model (model-agnostic navigation)
    inner = _get_inner_model(model)
    h = input_embeddings

    # Get the layer list — handle pipeline parallelism
    layers = getattr(inner, "pipeline_layers", inner.layers)

    # Create causal attention mask
    mask = create_attention_mask(h, cache[0], return_array=True)

    # Handle pipeline receive (no-op for single device)
    if (
        hasattr(inner, "pipeline_rank")
        and hasattr(inner, "pipeline_size")
        and inner.pipeline_rank < inner.pipeline_size - 1
    ):
        h = mx.distributed.recv_like(h, (inner.pipeline_rank + 1))

    # Run transformer layers
    for layer, c in zip(layers, cache, strict=False):
        h = layer(h, mask, cache=c)

    # Handle pipeline send (no-op for single device)
    if hasattr(inner, "pipeline_rank") and hasattr(inner, "pipeline_size"):
        if inner.pipeline_rank != 0:
            h = mx.distributed.send(h, (inner.pipeline_rank - 1) % inner.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)
        if inner.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

    h = inner.norm(h)

    # We don't need logits, just cache filling — but evaluate to materialize
    mx.eval([c.state for c in cache])

    elapsed = time.perf_counter() - start
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Vision prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tps:.1f} tok/s)"
    )
    return tps, num_tokens


def create_vision_embeddings(
    model: nn.Module,
    prompt_tokens: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    """Create input embeddings with vision features spliced in.

    Embeds all prompt tokens using the LM's embedding layer, then
    replaces the positions of image placeholder tokens with the
    pre-computed vision features.

    Args:
        model: The language model.
        prompt_tokens: [seq_len] — token IDs including placeholder tokens.
        image_features: [n_image_tokens, hidden_dim] — vision embeddings.
        image_token_id: The placeholder token ID to replace.

    Returns:
        input_embeddings: [1, seq_len, hidden_dim] — ready for transformer.
    """
    # Get the embedding layer from the model (model-agnostic navigation)
    inner = _get_inner_model(model)
    embed_tokens = inner.embed_tokens

    # Embed all tokens
    input_embeddings = embed_tokens(prompt_tokens[None])  # [1, seq_len, hidden_dim]

    # Build boolean mask of image-placeholder positions (pure MLX, no numpy)
    is_image = prompt_tokens == image_token_id  # [seq_len]
    n_placeholders = int(mx.sum(is_image).item())

    if n_placeholders > 0:
        if n_placeholders != image_features.shape[0]:
            logger.warning(
                f"Placeholder count ({n_placeholders}) != image features "
                f"({image_features.shape[0]}). Using min of both."
            )
            n = min(n_placeholders, image_features.shape[0])
            image_features = image_features[:n]

        # Map each sequence position to an index into image_features via cumsum.
        # Non-image positions get index 0 but are masked out by mx.where below.
        image_indices = mx.cumsum(is_image.astype(mx.int32)) - 1  # [seq_len]
        image_indices = mx.clip(image_indices, 0, image_features.shape[0] - 1)

        # Cast vision features to match text embedding dtype (avoids float32 promotion)
        gathered = image_features[image_indices].astype(
            input_embeddings.dtype
        )  # [seq_len, hidden_dim]
        result = mx.where(is_image[:, None], gathered, input_embeddings[0])
        input_embeddings = result[None]  # [1, seq_len, hidden_dim]

    return input_embeddings


# ──────────────────────────────────────────────────────────────────────
# VisionPipeline — orchestrates the full vision flow for generate.py
# ──────────────────────────────────────────────────────────────────────


def _find_media_regions(
    prompt_tokens: mx.array,
    images: list[str],
    image_token_id: int,
) -> list[MediaRegion]:
    """Find contiguous runs of media-pad tokens and pair with image hashes.

    Each contiguous run of ``image_token_id`` in *prompt_tokens* corresponds
    to one image (or video frame), in order.  We hash each raw image and
    record [start, end) so the KV prefix cache can validate whether a
    cached entry's vision content matches a new query.

    Generalises to video: just pass more ``images`` — each frame gets its
    own region.
    """
    tokens_np = np.array(prompt_tokens)  # fast boolean ops
    is_pad = tokens_np == image_token_id

    # Walk the boolean mask to find contiguous runs
    regions: list[MediaRegion] = []
    in_run = False
    run_start = 0
    for pos, pad in enumerate(is_pad):
        if pad and not in_run:
            run_start = pos
            in_run = True
        elif not pad and in_run:
            regions.append(
                MediaRegion(content_hash="", start_pos=run_start, end_pos=pos)
            )
            in_run = False
    if in_run:
        regions.append(
            MediaRegion(content_hash="", start_pos=run_start, end_pos=len(tokens_np))
        )

    # Pair each run with its image hash (same order as chat template)
    for i, region in enumerate(regions):
        if i < len(images):
            region.content_hash = hashlib.sha256(images[i].encode("ascii")).hexdigest()
        else:
            logger.warning(f"Media region {i} has no corresponding image")

    return regions


class VisionPipeline:
    """Orchestrates vision preprocessing: encode → prompt → embed → result.

    generate.py creates one of these (lazily) and calls process().
    All model-specific details are encapsulated here — generate.py only
    sees VisionResult.
    """

    def __init__(self, config: VisionCardConfig):
        self._config = config
        self._encoder = VisionEncoder(config)

    def process(
        self,
        images: list[str],
        chat_template_messages: list[dict],
        tokenizer,
        model: nn.Module,
        encode_prompt_fn,
        fix_think_fn,
    ) -> VisionResult:
        """Run the full vision pipeline.

        Args:
            images: List of base64-encoded image strings.
            chat_template_messages: Message dicts for chat_template.
            tokenizer: Tokenizer with chat_template support.
            model: The language model (for embedding layer access).
            encode_prompt_fn: Function to encode prompt string → mx.array.
            fix_think_fn: Function to fix unmatched think end tokens.

        Returns:
            VisionResult with prompt, tokens, embeddings, and cache key.
        """
        logger.info(f"Vision pipeline: {len(images)} image(s)")

        # 1. Encode images through vision tower + projector
        image_features, n_tokens_per_image = self._encoder.encode_images(images)
        logger.info(
            f"Vision features: {image_features.shape} "
            f"({image_features.shape[0]} tokens, per-image: {n_tokens_per_image})"
        )

        # 2. Build prompt with correct placeholder token counts
        prompt = build_vision_prompt(
            tokenizer,
            chat_template_messages,
            n_tokens_per_image,
            self._config.image_token,
        )
        logger.info(f"Vision prompt (first 200 chars): {prompt[:200]}")

        # 3. Encode prompt to tokens
        prompt_tokens = encode_prompt_fn(tokenizer, prompt)
        prompt_tokens = fix_think_fn(prompt_tokens, tokenizer)

        # 4. Create embeddings with vision features spliced in
        embeddings = create_vision_embeddings(
            model,
            prompt_tokens,
            image_features,
            self._config.image_token_id,
        )

        # 5. Compute media regions for KV cache matching
        media_regions = _find_media_regions(
            prompt_tokens,
            images,
            self._config.image_token_id,
        )

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            media_regions=media_regions,
        )


# ──────────────────────────────────────────────────────────────────────
# Public API for generate.py — thin wrappers that keep vision details
# out of the generation loop (minimises merge-conflict surface).
# ──────────────────────────────────────────────────────────────────────

# Lazily initialised VisionPipeline singleton (one per model type).
_vision_pipeline: VisionPipeline | None = None
_vision_pipeline_model_type: str | None = None


def prepare_vision(
    images: list[str] | None,
    chat_template_messages: list[dict] | None,
    vision_config: VisionCardConfig | None,
    tokenizer,
    model: nn.Module,
    encode_prompt_fn,
    fix_think_fn,
) -> VisionResult | None:
    """Run the full vision pipeline if images are present.

    Returns *None* for text-only requests.  Raises on invalid inputs.
    Manages the VisionPipeline singleton internally so generate.py
    doesn't need to.
    """
    if not images:
        return None

    if vision_config is None:
        raise ValueError(
            "This model does not support image inputs. "
            "Use a model with vision capabilities (e.g. Kimi K2.5 or Qwen3-VL)."
        )
    if chat_template_messages is None:
        raise ValueError("Vision requests must include chat_template_messages")

    global _vision_pipeline, _vision_pipeline_model_type
    if (
        _vision_pipeline is None
        or _vision_pipeline_model_type != vision_config.model_type
    ):
        _vision_pipeline = VisionPipeline(vision_config)
        _vision_pipeline_model_type = vision_config.model_type

    return _vision_pipeline.process(
        images=images,
        chat_template_messages=chat_template_messages,
        tokenizer=tokenizer,
        model=model,
        encode_prompt_fn=encode_prompt_fn,
        fix_think_fn=fix_think_fn,
    )


def vision_prefill_cached(
    model: nn.Module,
    vision: VisionResult,
    prefix_hit_length: int,
    cache: list,
) -> tuple[float, int]:
    """Prefill KV cache from a VisionResult, respecting prefix cache hits.

    Handles the embedding slicing (skip cached prefix, trim last 2 tokens
    so stream_generate can re-process them).  Returns (tps, n_tokens).
    """
    embed_start = prefix_hit_length
    embed_end = vision.embeddings.shape[1] - 2  # trim last 2 like normal prefill
    if embed_end > embed_start:
        return vision_prefill(
            model,
            vision.embeddings[:, embed_start:embed_end, :],
            cache,
        )
    # Entire vision prompt was already cached
    return 0.0, 0
