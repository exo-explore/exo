from typing import Tuple, Union, Optional, Dict, Any, List
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv
from collections import OrderedDict


# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.half, rope_scaling: Optional[Dict[str, float]] = None) -> Tensor:
  freqs = 1.0/(theta**(Tensor.arange(0, dim, 2)[:(dim // 2)]/dim))

  if rope_scaling:
    factor = rope_scaling.get('factor', 1.0)
    low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
    high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
    original_max_pos_emb = rope_scaling.get('original_max_position_embeddings', end)

    freqs[:dim // 4] *= low_freq_factor
    freqs[dim // 4:] = freqs[dim // 4:].contiguous()*high_freq_factor
    freqs *= (original_max_pos_emb/end)**(1.0/factor)

  freqs = Tensor.arange(end).unsqueeze(dim=1)*freqs.unsqueeze(dim=0)
  # TODO: move dtype outside this
  return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim // 2, 2)


# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a, b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads*n_rep, head_dim)

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, linear=nn.Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context

    self.wq = linear(dim, self.n_heads*self.head_dim, bias=False)
    self.wk = linear(dim, self.n_kv_heads*self.head_dim, bias=False)
    self.wv = linear(dim, self.n_kv_heads*self.head_dim, bias=False)
    self.wo = linear(self.n_heads*self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor], cache: Optional[Tensor]=None) -> Tensor:
    if getenv("WQKV"):
      if not hasattr(self, 'wqkv'): self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
      xqkv = x @ self.wqkv.T
      xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    if cache is not None:
      # update the cache
      assert xk.dtype == xv.dtype == cache.dtype, f"{xk.dtype=}, {xv.dtype=}, {cache.dtype=}"
      cache.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(Tensor.stack(xk, xv)).realize()

      keys = cache[0].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xk
      values = cache[1].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xv
    else:
      keys = xk
      values = xv

    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    return self.wo(attn)


class FeedForward:
  def __init__(self, dim: int, hidden_dim: int, linear=nn.Linear):
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False)  # the gate in Gated Linear Unit

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu()*self.w3(x))  # SwiGLU [arxiv/2002.05202, eq (5)]


class TransformerBlock:
  def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int, linear=nn.Linear, feed_forward=FeedForward):
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)
    self.feed_forward = feed_forward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor], cache: Optional[Tensor]=None):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, cache=cache)
    return (h + self.feed_forward(self.ffn_norm(h))).contiguous()


# standard openai sampling
def sample_logits(logits: Tensor, temp: float, k: int, p: float, af: float, ap: float):
  assert logits.ndim == 1, "only works on 1d tensors"
  assert 0 <= p <= 1, "p must be between 0 and 1"
  assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

  # if temperature is very low just use argmax
  if temp < 1e-6: return logits.argmax().reshape(1)

  # alpha sampling
  if af or ap:
    if not hasattr(sample, "alpha_counter"):
      setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
    logits = logits - (sample.alpha_counter*af + (sample.alpha_counter > 0)*ap)

  # replace NaNs with -inf
  logits = (logits != logits).where(-float("inf"), logits)

  # softmax
  t = (logits/temp).softmax()

  counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
  # top k
  if k:
    output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k, device=logits.device, dtype=dtypes.int32).contiguous()
    for i in range(k):
      t_argmax = (t.numel() - ((t == (t_max := t.max()))*counter2).max() - 1).cast(dtypes.default_int)
      output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
      output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
      t = (counter == t_argmax).where(0, t)

    # approximate top p
    # because we are already limited to top k elements we can do top p "without sorting"
    output_cumsum = output[::-1]._cumsum()[::-1] + t.sum()
    output = (output_cumsum >= (1 - p))*output
    output_indices = (output_cumsum >= (1 - p))*output_indices

    # sample
    output_idx = output.multinomial()
    output_token = output_indices[output_idx]
  else:
    output_token = t.multinomial()

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  return output_token


from exo.inference.shard import Shard


class Transformer:
  def __init__(
    self,
    dim: int,
    hidden_dim: int,
    n_heads: int,
    n_layers: int,
    norm_eps: float,
    vocab_size,
    shard: Shard = None,
    linear=nn.Linear,
    n_kv_heads=None,
    rope_theta=10000,
    max_context=1024,
    jit=True,
    feed_forward=FeedForward,
    rope_scaling: Optional[Dict[str, float]] = None,
    tie_word_embeddings=False,
  ):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear, feed_forward=feed_forward) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    if tie_word_embeddings:
      self.output.weight = self.tok_embeddings.weight
    self.max_context = max_context
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, self.max_context*2, rope_theta, rope_scaling=rope_scaling).contiguous()
    self.forward_jit = TinyJit(self.forward_base) if jit else None
    self.shard = shard

  def forward_base(self, x: Tensor, start_pos: Union[Variable, int], cache: Optional[List[Tensor]] = None):
    seqlen = x.shape[1]
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-100000000"), dtype=x.dtype, device=x.device).triu(start_pos + 1).realize() if seqlen > 1 else None

    h = x

    if cache is None:
      cache = [None for _ in range(self.shard.start_layer, self.shard.end_layer + 1)]  
    for i, c in zip(range(self.shard.start_layer, self.shard.end_layer + 1), cache):
      layer = self.layers[i]
      h = layer(h, start_pos, freqs_cis, mask, cache=c)

    if self.shard.is_last_layer():
      logits = self.output(self.norm(h)).float().realize()
      return logits
    else:
      return h

  def embed(self, inputs: Tensor):
    if self.shard.is_first_layer():
      h = self.tok_embeddings(inputs)
    else:
      h = inputs
    return h

  def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
    if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
      return self.forward_jit(x, Variable("start_pos", 1, self.max_context).bind(start_pos), cache=cache)
    return self.forward_base(x, start_pos, cache=cache)

  def __call__(self, x: Tensor, start_pos: Variable, cache: Optional[List[Tensor]] = None):
    # TODO: better way to handle the first call v.s. the rest?
    h = self.embed(x)
    return self.forward(h, start_pos, cache=cache)

class TransformerShard:
  def __init__(
    self,
    shard: Shard,
    base,
    jit: bool = True,
  ):
    shardrange = range(shard.start_layer, shard.end_layer + 1)
    self.layers = [layer for layer, n in zip(base.layers, range(shard.n_layers)) if n in shardrange]
    self.norm = base.norm 
    self.tok_embeddings = base.tok_embeddings
    self.embed = (lambda x: self.tok_embeddings(x)) if shard.is_first_layer() else (lambda x: x)
    self.output = base.output
    self.post = (lambda x: self.output(x)) if shard.is_last_layer() else (lambda x: x)
    self.max_context = base.max_context
    self.null_cache = [None for _ in shardrange] 
    self.freqs_cis = base.freqs_cis
    self.forward_jit = TinyJit(self.forward_base) if jit else None

  def forward_base(self, x: Tensor, start_pos: Union[Variable, int], cache):
    seqlen = x.shape[1]
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-100000000"), dtype=x.dtype, device=x.device).triu(start_pos + 1).realize() if seqlen > 1 else None

    for layer, c in zip(self.layers, cache):
      x = layer(x, start_pos, freqs_cis, mask, cache=c)

    out = self.post(x)
    return out

  def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
    if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
      return self.forward_jit(x, Variable("start_pos", 1, self.max_context).bind(start_pos), cache=cache)
    return self.forward_base(x, start_pos, cache=cache)

  def __call__(self, x: Tensor, start_pos: Variable, cache: Optional[List[Tensor]] = None):
    # TODO: better way to handle the first call v.s. the rest?
    h = self.embed(x)
    return self.forward(h, start_pos, cache=self.null_cache if cache is None else cache)
      
# *** helpers ***


def convert_from_huggingface(weights: Dict[str, Tensor], model: Transformer, n_heads: int, n_kv_heads: int):
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight"
       for l in range(len(model.layers))},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight"
       for x in ["q", "k", "v", "o"]
       for l in range(len(model.layers))},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight"
       for l in range(len(model.layers))},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight"
       for x, y in {"gate": "1", "down": "2", "up": "3"}.items()
       for l in range(len(model.layers))},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k:
        v = permute(v, n_heads)
      elif "k_proj" in k:
        v = permute(v, n_kv_heads)
    if k in keymap:
      sd[keymap[k]] = v
    else:
      sd[k] = v
  return sd


def fix_bf16(weights: Dict[Any, Tensor]):
  if Device.DEFAULT == "CLANG":
    # TODO: without casting to float16, 70B llama OOM on tinybox.
    return {
      k: (v.llvm_bf16_cast(dtypes.float32).to(v.device) if v.dtype == dtypes.bfloat16 else v) 
      for k, v in weights.items()
    }
  if getenv("SUPPORT_BF16", 1):
    # TODO: without casting to float16, 70B llama OOM on tinybox.
    return {k: v.cast(dtypes.float32).cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}
  # TODO: check if device supports bf16
  return {k: v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}
