import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mlx_vlm.utils import ImageProcessor
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import load_image_processor
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import VisionCardConfig
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    render_chat_template,
)
from exo.worker.runner.bootstrap import logger

_video_processor_patched = False


def _filter_config(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in d.items() if k in valid}  # type: ignore


_ProcessorOutput = dict[str, np.ndarray] | tuple[dict[str, np.ndarray], list[int]]


def _run_processor(
    processor: "ImageProcessor",
    pil_images: list[Image.Image],
) -> tuple[dict[str, np.ndarray], list[int] | None]:
    """
    Image processors split into two families by how they report per-image
    token counts:

    1. Variable-resolution patch models (Qwen3-VL, Llama 4 vision, ...):
       return a `BatchFeature` dict containing `pixel_values` and
       `image_grid_thw` — an (n_images, 3) array of (temporal, height, width)
       patch counts. The caller multiplies the three to get the per-image
       token count, so no override is needed.

    2. Fixed-token-budget models (Gemma 4): every image collapses to a fixed
       number of soft tokens, so there's no grid to report. These processors
       return `(batch_feature_dict, [n_tokens_per_image])` instead.

    We normalize both into (dict, optional tokens override).
    """
    raw = cast(_ProcessorOutput, processor(images=pil_images, return_tensors="np"))
    if isinstance(raw, tuple):
        batch, tokens = raw
        return batch, [int(n) for n in tokens]
    return raw, None


def _instantiate_projector(
    cls: type,
    model_config: Any,  # pyright: ignore[reportAny]
    vision_config: Any,  # pyright: ignore[reportAny]
    text_config: Any,  # pyright: ignore[reportAny]
) -> nn.Module:
    """
    Instantiate projector/embedding classes with any missing values
    """
    init_sig = inspect.signature(cls.__init__)
    params = {n: p for n, p in init_sig.parameters.items() if n != "self"}
    kwargs: dict[str, Any] = {}

    if "embedding_dim" in params:
        kwargs["embedding_dim"] = vision_config.hidden_size  # pyright: ignore[reportAny]
    if "text_hidden_size" in params:
        kwargs["text_hidden_size"] = text_config.hidden_size  # pyright: ignore[reportAny]
    if "eps" in params:
        kwargs["eps"] = getattr(vision_config, "rms_norm_eps", 1e-6)  # pyright: ignore[reportAny]
    return cls(**kwargs)  # type: ignore


def _patch_video_processor() -> None:
    """Patch so we don't crash horribly when torch vision isn't installed"""
    # TODO: Update if we add torch vision.
    global _video_processor_patched
    if _video_processor_patched:
        return
    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES  # type: ignore
        mapping.pop("video_processor", None)
    except (ImportError, AttributeError):
        pass
    _video_processor_patched = True


def decode_base64_image(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def _format_vlm_messages(
    messages: list[dict[str, Any]],
    model_type: str,
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        role: str = str(msg.get("role", "user"))  # type: ignore
        content: Any = msg.get("content")
        if not isinstance(content, list):
            formatted.append(msg)
            continue
        parts: list[dict[str, Any]] = content  # type: ignore
        text_parts = [str(p["text"]) for p in parts if p.get("type") == "text"]  # type: ignore
        n_images = sum(1 for p in parts if p.get("type") in ("image", "image_url"))
        result: dict[str, Any] = get_message_json(
            model_type, " ".join(text_parts), role, num_images=n_images
        )
        formatted.append(result)
    return formatted


def build_vision_prompt(
    tokenizer: TokenizerWrapper,
    chat_template_messages: list[dict[str, Any]],
    n_tokens_per_image: list[int],
    image_token: str,
    task_params: TextGenerationTaskParams,
) -> str:
    prompt = render_chat_template(tokenizer, chat_template_messages, task_params)

    image_idx = 0
    result: list[str] = []
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


@dataclass
class MediaRegion:
    content_hash: str
    start_pos: int
    end_pos: int


@dataclass
class VisionResult:
    prompt: str
    prompt_tokens: mx.array
    embeddings: mx.array
    media_regions: list[MediaRegion]
    image_token_id: int


class VisionEncoder:
    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self._config = config
        self._main_model_path = build_model_path(model_id)
        self._model_path = build_model_path(ModelId(config.weights_repo))
        self._vision_tower: nn.Module | None = None
        self._projector: nn.Module | None = None
        self._processor: "ImageProcessor | None" = None
        self._spatial_merge_size: int = 2
        self._merge_kernel_size: list[int] | None = None
        self._needs_nhwc: bool = False
        self._loaded = False

    def _load_config_json(self) -> dict[str, Any]:
        for candidate in (self._main_model_path, self._model_path):
            path = candidate / "config.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)  # type: ignore
        return {}

    def _import_mlx_vlm(self, *submodules: str) -> Any:  # type: ignore
        mt = self._config.model_type
        results: list[Any] = []
        for sub in submodules:
            name = f"mlx_vlm.models.{mt}.{sub}"
            results.append(importlib.import_module(name))
        return results[0] if len(results) == 1 else tuple(results)

    def _apply_projector_quantization_if_needed(
        self, projector_weights: dict[str, mx.array]
    ) -> None:
        # Quantized models ship the projector's Linear layers as packed uint32
        # weights plus `.scales`/`.biases`. Our now instantiated projector
        # uses plain nn.Linear, so we must mirror the packing via nn.quantize
        # before load_weights, otherwise MLX rejects the extra parameters.
        if self._projector is None:
            return
        has_quantized_tensors = any(
            key.endswith((".scales", ".biases")) or val.dtype == mx.uint32
            for key, val in projector_weights.items()
        )
        if not has_quantized_tensors:
            return
        config = self._load_config_json()
        quant_cfg = cast(dict[str, Any], config.get("quantization") or {})
        if not quant_cfg:
            return
        group_size = int(cast(int, quant_cfg.get("group_size", 64)))
        bits = int(cast(int, quant_cfg.get("bits", 4)))
        nn.quantize(self._projector, group_size=group_size, bits=bits)

    def _load_image_processor_from_module(self, repo: str) -> "ImageProcessor | None":
        # mlx_vlm.utils.load_image_processor only works for models that set
        # `Model.ImageProcessor = <cls>`, but Gemma4 just uses
        # `Gemma4ImageProcessor` from the package `__init__.py`
        try:
            pkg: Any = importlib.import_module(
                f"mlx_vlm.models.{self._config.model_type}"
            )
        except ImportError:
            return None
        for attr in dir(pkg):  # pyright: ignore[reportAny]
            cls = getattr(pkg, attr)  # pyright: ignore[reportAny]
            if isinstance(cls, type) and attr.endswith("ImageProcessor"):
                return cls.from_pretrained(repo)  # type: ignore
        return None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_weights()
        self._loaded = True

    def _load_weights(self) -> None:
        _patch_video_processor()
        logger.info(f"Loading vision weights from {self._model_path}")
        config = self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        vision_cfg = config.get("vision_config", {})  # type: ignore

        config_mod, vision_mod = self._import_mlx_vlm("config", "vision")  # type: ignore
        vision_config_cls = config_mod.VisionConfig  # type: ignore
        vision_model_cls = vision_mod.VisionModel  # type: ignore

        vision_config = vision_config_cls(  # type: ignore
            **_filter_config(vision_config_cls, vision_cfg)  # type: ignore
        )
        self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)  # type: ignore
        self._vision_tower = vision_model_cls(vision_config)
        model_mod: Any = None
        with contextlib.suppress(ImportError):
            model_mod = self._import_mlx_vlm(self._config.model_type)  # type: ignore

        projector_cls = None
        if model_mod is not None:
            for attr_name in dir(model_mod):  # type: ignore
                obj = getattr(model_mod, attr_name)  # type: ignore
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and ("Projector" in attr_name or "Embedder" in attr_name)
                ):
                    projector_cls = obj
                    break

        if projector_cls is not None:
            text_config = config_mod.TextConfig(  # type: ignore
                **_filter_config(config_mod.TextConfig, config.get("text_config", {}))  # type: ignore
            )
            extra = {
                k: v
                for k, v in config.items()  # type: ignore
                if k not in ("text_config", "vision_config")
            }
            extra.setdefault("model_type", self._config.model_type)
            model_config = config_mod.ModelConfig(  # type: ignore
                text_config=text_config,
                vision_config=vision_config,
                **_filter_config(config_mod.ModelConfig, extra),  # type: ignore
            )
            self._projector = _instantiate_projector(
                projector_cls,
                model_config,
                vision_config,
                text_config,
            )

        processor_repo = self._config.processor_repo
        if processor_repo:
            self._load_weights_from_separate_repo()
        else:
            self._load_weights_from_model_repo()

        repo = processor_repo or str(self._model_path)
        image_proc = load_image_processor(
            repo
        ) or self._load_image_processor_from_module(repo)
        if image_proc is not None:
            self._processor = image_proc
        else:
            self._processor = AutoImageProcessor.from_pretrained(  # type: ignore
                repo, trust_remote_code=True
            )
        if processor_repo:
            self._merge_kernel_size = vision_cfg.get("merge_kernel_size", [2, 2])  # type: ignore
            self._needs_nhwc = True
        logger.info(f"HF image processor loaded from {repo}")

    def _load_weights_from_separate_repo(self) -> None:
        safetensors_files = list(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)  # type: ignore
                    np_tensor = tensor.float().numpy()  # type: ignore
                    weights[key] = mx.array(np_tensor, dtype=mx.bfloat16)  # type: ignore

        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        for key, val in weights.items():
            if key.startswith("vision_tower."):
                short_key = key[len("vision_tower.") :]
                if short_key.startswith("encoder."):
                    short_key = short_key[len("encoder.") :]
                m = re.match(r"^(blocks\.\d+)\.(wqkv|wo)\.(weight|bias)$", short_key)
                if m:
                    short_key = f"{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
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

        assert self._vision_tower is not None
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
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        vision_prefixes = ["vision_tower.", "model.visual."]
        projector_prefixes = [
            "embed_vision.",
            "multi_modal_projector.",
            "mm_projector.",
        ]
        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}

        # If weights under `model.visual.`, we need to call mlx_vlm's VisionModel.sanitize()
        # to remap into its own keys.
        needs_sanitize = False

        for sf_path in safetensors_files:
            file_weights: dict[str, mx.array] = mx.load(str(sf_path))  # type: ignore
            for key, val in file_weights.items():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        vision_weights[key[len(prefix) :]] = val
                        if prefix == "model.visual.":
                            needs_sanitize = True
                        break
                else:
                    for prefix in projector_prefixes:
                        if key.startswith(prefix):
                            projector_weights[key[len(prefix) :]] = val
                            break

        if not vision_weights:
            raise ValueError(
                f"No vision weights found with prefixes {vision_prefixes} in {self._model_path}. "
                "Ensure the model repo contains bundled vision weights."
            )

        assert self._vision_tower is not None
        if needs_sanitize:
            sanitize: Callable[[dict[str, mx.array]], dict[str, mx.array]] | None = (
                getattr(self._vision_tower, "sanitize", None)
            )
            if sanitize is not None:
                vision_weights = sanitize(vision_weights)

        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            self._apply_projector_quantization_if_needed(projector_weights)
            self._projector.load_weights(list(projector_weights.items()))
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for v in vision_weights.values())
        n_proj = sum(v.size for v in projector_weights.values())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f" (+ projector {n_proj / 1e6:.1f}M)" if n_proj else "")
        )

    def encode_images(self, images: list[str]) -> tuple[mx.array, list[int]]:
        self.ensure_loaded()
        assert self._vision_tower is not None
        assert self._processor is not None

        pil_images = [decode_base64_image(b64) for b64 in images]
        for idx, img in enumerate(pil_images):
            logger.info(f"Image {idx}: {img.width}x{img.height} mode={img.mode}")

        per_image_pixels: list[mx.array]
        grid_thw: mx.array | None
        n_tokens_per_image: list[int]

        if self._config.processor_repo:
            processed = self._processor.preprocess(
                [{"type": "image", "image": img} for img in pil_images],
                return_tensors="np",
            )
            stacked_pixels = mx.array(processed["pixel_values"])  # type: ignore
            if stacked_pixels.ndim == 3:
                stacked_pixels = stacked_pixels[None]
            per_image_pixels = [
                stacked_pixels[i : i + 1] for i in range(stacked_pixels.shape[0])
            ]
            grid_thw = mx.array(processed["grid_thws"])  # type: ignore
            assert self._merge_kernel_size is not None
            merge_length = int(np.prod(self._merge_kernel_size))
            n_tokens_per_image = [
                int(mx.prod(grid_thw[i]).item()) // merge_length
                for i in range(grid_thw.shape[0])
            ]
        else:
            batch, tokens_override = _run_processor(self._processor, pil_images)
            # `Gemma4ImageProcessor` returns pixel_values as a plain ndarray
            # when all images resize to the same shape, or as a Python list of
            # per-image (C, H, W) ndarrays when they differ. Treat it as the
            # union here.
            raw_pixel_values: np.ndarray | list[np.ndarray] = cast(
                "np.ndarray | list[np.ndarray]", batch["pixel_values"]
            )
            raw_grid = batch.get("image_grid_thw")
            grid_thw = mx.array(raw_grid) if raw_grid is not None else None
            if tokens_override is not None:
                n_tokens_per_image = tokens_override
            else:
                assert grid_thw is not None
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

            if isinstance(raw_pixel_values, list):
                per_image_pixels = [
                    # (C, H, W) -> (1, C, H, W)
                    mx.array(p)[None] if p.ndim == 3 else mx.array(p)
                    for p in raw_pixel_values
                ]
            else:
                stacked = mx.array(raw_pixel_values)
                per_image_pixels = [stacked[i : i + 1] for i in range(stacked.shape[0])]

        if self._needs_nhwc:
            assert grid_thw is not None
            pixel_values = mx.concatenate(per_image_pixels, axis=0)
            grid_hw = grid_thw[:, 1:] if grid_thw.shape[-1] == 3 else grid_thw
            hidden_states = self._vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                output_hidden_states=True,
                grid_thw=grid_hw,
            )
        elif grid_thw is None:
            # Fixed-token-budget models (gemma4): run each image separately
            # since they can have different spatial shapes *and* different
            # soft-token counts, then flatten each to (n_tokens_i, hidden)
            # and concatenate along the token axis.
            per_image_hidden: list[mx.array] = []
            for pv in per_image_pixels:
                result = self._vision_tower(pv)
                h = result[0] if isinstance(result, tuple) else result
                if h.ndim == 3:
                    h = h.reshape(-1, h.shape[-1])
                per_image_hidden.append(h)
            hidden_states = mx.concatenate(per_image_hidden, axis=0)
        else:
            pixel_values = mx.concatenate(per_image_pixels, axis=0)
            result = self._vision_tower(pixel_values, grid_thw)
            hidden_states = result[0] if isinstance(result, tuple) else result

        if self._projector is not None:
            image_features: mx.array = self._projector(hidden_states)
        else:
            image_features = hidden_states

        # `create_vision_embeddings` expects a 2D (total_tokens, hidden) view,
        # but fixed-token-budget models (gemma4) return (n_images, tokens, hidden).
        if image_features.ndim == 3:
            image_features = image_features.reshape(-1, image_features.shape[-1])

        return image_features, n_tokens_per_image


def get_inner_model(model: nn.Module) -> Any:  # type: ignore
    for candidate in (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "embed_tokens"):  # type: ignore
            return candidate  # type: ignore

    raise ValueError(
        f"Could not find inner transformer (embed_tokens) in {type(model).__name__}. "
        "Add a new pattern to _get_inner_model() for this architecture."
    )


def create_vision_embeddings(
    model: Model,
    prompt_tokens: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    inner = get_inner_model(model)  # type: ignore
    embed_tokens = inner.embed_tokens  # type: ignore

    input_embeddings: mx.array = embed_tokens(prompt_tokens[None])  # type: ignore

    is_image: mx.array = mx.equal(prompt_tokens, image_token_id)
    n_placeholders = int(mx.sum(is_image).item())

    if n_placeholders > 0:
        if n_placeholders != image_features.shape[0]:
            logger.warning(
                f"Placeholder count ({n_placeholders}) != image features "
                f"({image_features.shape[0]}). Using min of both."
            )
            n = min(n_placeholders, image_features.shape[0])
            image_features = image_features[:n]

        # Gemma-family models apply `h = input_embeddings * embed_scale` inside
        # the inner model's forward pass. That scale is appropriate for text
        # tokens (which come out of a raw `embed_tokens(id)` lookup) but not
        # for our pre-projected image features. Pre-divide by `embed_scale`
        # so that after the model multiplies, image features are unchanged
        # while text positions remain correctly scaled.
        if hasattr(inner, "embed_scale"):  # type: ignore
            embed_scale = float(inner.embed_scale)  # type: ignore
            image_features = image_features / embed_scale

        image_indices = mx.cumsum(is_image.astype(mx.int32)) - 1
        image_indices = mx.clip(image_indices, 0, image_features.shape[0] - 1)

        gathered = image_features[image_indices].astype(input_embeddings.dtype)
        result = mx.where(is_image[:, None], gathered, input_embeddings[0])
        input_embeddings = result[None]

    return input_embeddings


def _find_media_regions(
    prompt_tokens: mx.array,
    images: list[str],
    image_token_id: int,
) -> list[MediaRegion]:
    tokens_np = np.array(prompt_tokens)
    is_pad = tokens_np == image_token_id  # type: ignore

    regions: list[MediaRegion] = []
    in_run = False
    run_start = 0
    for pos, pad in enumerate(is_pad):  # type: ignore
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

    for i, region in enumerate(regions):
        if i < len(images):
            img = decode_base64_image(images[i])
            region.content_hash = hashlib.sha256(img.tobytes()).hexdigest()
        else:
            logger.warning(f"Media region {i} has no corresponding image")

    return regions


class VisionProcessor:
    """
    Pipeline for vision models:
    1. Encode images into features (or grab from cache)
    2. Replace image placeholders with the features
    3. Build vision prompt
    4. Provide media regions for prefix caching
    """

    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self.vision_config = config
        self._encoder = VisionEncoder(config, model_id)
        self._feature_cache: dict[str, tuple[mx.array, list[int]]] = {}
        self._feature_cache_max = 32

    def load(self) -> None:
        self._encoder.ensure_loaded()

    def _image_cache_key(self, images: list[str]) -> str:
        h = hashlib.sha256()
        for img in images:
            pil = decode_base64_image(img)
            h.update(pil.tobytes())
        return h.hexdigest()

    def process(
        self,
        images: list[str],
        chat_template_messages: list[dict[str, Any]],
        tokenizer: TokenizerWrapper,
        model: Model,
        task_params: TextGenerationTaskParams,
    ) -> VisionResult:
        logger.info(f"Vision pipeline: {len(images)} image(s)")

        cache_key = self._image_cache_key(images)
        cached = self._feature_cache.pop(cache_key, None)
        if cached is not None:
            self._feature_cache[cache_key] = cached
            image_features, n_tokens_per_image = cached
        else:
            image_features, n_tokens_per_image = self._encoder.encode_images(images)
            self._feature_cache[cache_key] = (image_features, n_tokens_per_image)
            while len(self._feature_cache) > self._feature_cache_max:
                del self._feature_cache[next(iter(self._feature_cache))]
        logger.info(
            f"Vision features: {image_features.shape} "
            f"({image_features.shape[0]} tokens, per-image: {n_tokens_per_image})"
        )

        image_token = self.vision_config.image_token
        if image_token is None:
            image_token = tokenizer.decode([self.vision_config.image_token_id])

        formatted_messages = _format_vlm_messages(
            chat_template_messages, self.vision_config.model_type
        )

        prompt = build_vision_prompt(
            tokenizer,
            formatted_messages,
            n_tokens_per_image,
            image_token,
            task_params,
        )

        logger.info(
            f"Expanded prompt has {prompt.count(image_token)} image_token occurrences, total len={len(prompt)}"
        )

        prompt_tokens: mx.array = encode_prompt(tokenizer, prompt)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_image_tokens = int(
            mx.sum(mx.equal(prompt_tokens, self.vision_config.image_token_id)).item()
        )
        logger.info(
            f"Encoded prompt: {len(prompt_tokens)} tokens, {n_image_tokens} image pad tokens"
        )

        embeddings = create_vision_embeddings(
            model,
            prompt_tokens,
            image_features,
            self.vision_config.image_token_id,
        )
        mx.eval(embeddings)

        media_regions = _find_media_regions(
            prompt_tokens,
            images,
            self.vision_config.image_token_id,
        )

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            media_regions=media_regions,
            image_token_id=self.vision_config.image_token_id,
        )


def prepare_vision(
    images: list[str] | None,
    chat_template_messages: list[dict[str, Any]] | None,
    vision_processor: VisionProcessor,
    tokenizer: TokenizerWrapper,
    model: Model,
    model_id: ModelId,
    task_params: TextGenerationTaskParams,
) -> VisionResult | None:
    if not images:
        return None
    if chat_template_messages is None:
        logger.warning(
            "Vision request missing chat_template_messages — ignoring images"
        )
        return None

    return vision_processor.process(
        images=images,
        chat_template_messages=chat_template_messages,
        tokenizer=tokenizer,
        model=model,
        task_params=task_params,
    )
