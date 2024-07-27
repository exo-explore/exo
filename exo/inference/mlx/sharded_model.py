from typing import Any, Dict, Generator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import KVCache
from mlx_lm.sample_utils import top_p_sampling

from ..shard import Shard

class StatefulShardedModel:
    def __init__(self, shard: Shard, model: nn.Module):
        self.shard = shard
        self.model = model
        self.request_cache: Dict[str, Tuple[str, KVCache]] = {}

    def step(
        self,
        request_id: str,
        x,
        pixel_values=None,
        temp: float = 0.0,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            if logit_bias:
                indices = mx.array(list(logit_bias.keys()))
                values = mx.array(list(logit_bias.values()))
                logits[:, indices] += values

            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    token = top_p_sampling(logits, top_p, temp)
                else:
                    token = mx.random.categorical(logits * (1 / temp))

            return token

        y = x

        if request_id not in self.request_cache:
            self.init_cache(request_id)

        if pixel_values is None:
            output = self.model(y[None] if self.shard.is_first_layer() else y, cache=self.request_cache[request_id])
        else:
            output = self.model(y, pixel_values=pixel_values, cache=self.request_cache[request_id])

        if self.shard.is_last_layer():
            logits = output[:, -1, :]
            y = sample(logits)
            return y
        else:
            return output

    def __call__(
            self,
            x,
            temp: float = 0.0,
            top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        return self.step(x, temp, top_p, logit_bias)

    def init_cache(self, request_id: str):
        kv_heads = (
            [self.model.n_kv_heads] * len(self.model.layers)
            if isinstance(self.model.n_kv_heads, int)
            else self.model.n_kv_heads
        )
        self.request_cache[request_id] = [KVCache(self.model.head_dim, n) for n in kv_heads]