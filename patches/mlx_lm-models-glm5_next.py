"""mlx_lm bridge for the glm5_next architecture (GLM-5.3-Flash).

mlx_lm resolves an architecture by importing ``mlx_lm.models.<model_type>`` and
reading ``Model``/``ModelArgs`` from it; it ships no ``glm5_next`` module, so
loading GLM-5.3-Flash raises "Model type glm5_next not supported."

mlx_vlm 0.6.17 does ship a complete implementation. EXO always loads the text
backbone through ``mlx_lm.load_model`` -- even for multimodal checkpoints, where
the vision tower is only attached afterwards as a side-car processor -- so this
module adapts mlx_vlm's language model to mlx_lm's interface.

The checkpoint stores the backbone under ``language_model.*`` and the vision
tower under ``vision_tower.*``. The attribute below is therefore named
``language_model`` so module paths line up with the weight names, and sanitize()
drops the vision tensors, which this text path does not own.
"""

import inspect
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

_VISION_PREFIXES = ("vision_tower.", "vision_model.", "model.visual.", "visual.")


def _renest_forget_gate_quant(weights):
    """Move the forget-gate projections' quantization tensors under forget_gate.

    mlx_vlm's LanguageModel.sanitize re-nests only ``f_a_proj.weight`` and
    ``f_b_proj.weight`` (its fg_parts tuple lists just the weights), so on a
    quantized checkpoint the matching ``.scales``/``.biases`` are left at the
    un-nested ``self_attn.f_*_proj.*`` path. mlx_lm decides what to quantize by
    testing ``f"{module_path}.scales" in weights``; with those keys missing it
    leaves f_a_proj/f_b_proj as plain nn.Linear, and the fused linear-attention
    matmul then raises "'Linear' object has no attribute 'scales'" at the first
    forward pass.
    """
    out = {}
    for k, v in weights.items():
        for proj in ("f_a_proj", "f_b_proj"):
            for suf in ("scales", "biases"):
                tail = f".self_attn.{proj}.{suf}"
                if k.endswith(tail):
                    k = k[: -len(f"{proj}.{suf}")] + f"forget_gate.{proj}.{suf}"
                    break
            else:
                continue
            break
        out[k] = v
    return out


@dataclass
class ModelArgs(TextConfig):
    """mlx_lm hands the whole config.json to from_dict; the backbone fields for
    a multimodal checkpoint live under ``text_config``."""

    @classmethod
    def from_dict(cls, params):
        src = dict(params.get("text_config") or params)
        src.setdefault("model_type", params.get("model_type", "glm5_next"))
        if "tie_word_embeddings" in params and "tie_word_embeddings" not in src:
            src["tie_word_embeddings"] = params["tie_word_embeddings"]
        # MTP checkpoints (num_nextn_predict_layers >= 1) describe one extra
        # layer in the per-layer type arrays than the backbone actually has.
        # The multi-token-prediction head's weights are dropped by
        # LanguageModel.sanitize, so trim its entries to keep the arrays
        # consistent with num_hidden_layers.
        n = src.get("num_hidden_layers")
        if isinstance(n, int):
            for key in ("layer_types", "mlp_layer_types", "indexer_types"):
                v = src.get(key)
                if isinstance(v, list) and len(v) > n:
                    src[key] = v[:n]
        allowed = set(inspect.signature(cls).parameters)
        return cls(**{k: v for k, v in src.items() if k in allowed})


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args)

    def __call__(self, inputs, cache=None, mask=None, **kwargs):
        return self.language_model(inputs, cache=cache, mask=mask, **kwargs).logits

    @property
    def layers(self):
        return self.language_model.model.layers

    def sanitize(self, weights):
        lang = {}
        for k, v in weights.items():
            if k.startswith(_VISION_PREFIXES):
                continue
            # Two checkpoint lineages exist for glm5_next:
            #  * MLX-native (Vontra, grant-ai, ...): language_model.model.layers.N,
            #    experts already stacked into switch_mlp, kv_b_proj already absorbed.
            #  * upstream-HF layout quantised in place (orcarouter, ...):
            #    model.language_model.layers.N, per-expert mlp.experts.N.*, and a raw
            #    kv_b_proj. mlx_vlm's inherited DeepSeek-V32 sanitize stacks the
            #    experts and absorbs kv_b_proj into embed_q/unembed_out, but only
            #    when the keys start with "model.layers.". Canonicalise to that.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            lang[k[len("language_model.") :] if k.startswith("language_model.") else k] = v
        lang = self.language_model.sanitize(lang)
        lang = _renest_forget_gate_quant(lang)
        # MLX promotes on the scales dtype: quantized_matmul / gather_qmm with
        # bfloat16 activations and float16 scales return float32 (bfloat16
        # scales return bfloat16). Checkpoints that store scales/biases as
        # float16 therefore flip the whole residual stream to float32 at the
        # first quantized MLP and never come back -- which upcasts every
        # remaining bf16 weight on each call, doubles the KV cache, and makes
        # MLA prefill ask for a float32 SDPA kernel at head_dim 256 that needs
        # more threadgroup memory than Metal allows, so any prompt beyond a
        # handful of tokens fails to load a kernel at all.
        lang = {
            k: (v.astype(mx.bfloat16) if v.dtype == mx.float16 else v)
            for k, v in lang.items()
        }
        return {f"language_model.{k}": v for k, v in lang.items()}

    def make_cache(self):
        # Build caches from mlx_lm's cache classes, not mlx_vlm's. The two
        # packages ship API-identical duplicates of ArraysCache/CacheList/
        # KVCache, but EXO's runner detects recurrent (SSM) layers with
        # isinstance() against mlx_lm.models.cache.ArraysCache - in
        # has_non_kv_caches(), the prefill rollback, and cache.py's trim path.
        # Caches built by mlx_vlm's make_cache fail all of those checks, so
        # EXO treats the recurrent state as a KV cache and calls .trim() on it:
        #   AttributeError: 'ArraysCache' object has no attribute 'trim'
        from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
        caches = []
        for layer in self.language_model.model.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
