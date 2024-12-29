from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.exaone import TransformerBlock, ModelArgs
from ...shard import Shard
from .base import IdentityBlock


@dataclass
class ModelArgs(ModelArgs):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        # super().__post_init__()  # Ensure parent initializations are respected

        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        self.shard = Shard(**self.shard)


class ExaoneModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)
        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            h = layer(h, mask, cache=c)

        return self.ln_f(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = ExaoneModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.wte.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.transformer.h

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads