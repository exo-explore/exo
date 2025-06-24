from typing import Tuple, Union, Optional, Dict, Any, List
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv
from collections import OrderedDict
import math

from .llama import precompute_freqs_cis, apply_rotary_emb, repeat_kv, complex_mult, sample_logits
from exo.inference.shard import Shard


def precompute_freqs_cis_deepseek(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.half, rope_scaling: Optional[Dict[str, Any]] = None) -> Tensor:
    """Deepseek V3 uses different RoPE scaling than Llama"""
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    
    if rope_scaling and rope_scaling.get("type") == "deepseek":
        mscale = rope_scaling.get("mscale", 1.0)
        mscale_all = rope_scaling.get("mscale_all", 1.0)
        
        # Apply Deepseek-specific scaling
        freqs = freqs * mscale
    
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim // 2, 2)


class DeepseekV3Attention:
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_context: int,
                 qk_nope_head_dim: int = 128, qk_rope_head_dim: int = 64,
                 v_head_dim: int = 128, q_lora_rank: int = 1536, linear=nn.Linear):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.max_context = max_context
        
        # Deepseek V3 specific dimensions
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        
        # Projections with Deepseek V3 dimensions
        self.wq_a = linear(dim, q_lora_rank, bias=False)
        self.wq_b = linear(q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.wk = linear(dim, self.n_kv_heads * self.qk_head_dim, bias=False)
        self.wv = linear(dim, self.n_kv_heads * self.v_head_dim, bias=False)
        self.wo = linear(self.n_heads * self.v_head_dim, dim, bias=False)
        
        # RoPE frequency initialization
        self.freqs_cis = precompute_freqs_cis_deepseek(self.qk_rope_head_dim, max_context * 2)
    
    def __call__(self, x: Tensor, start_pos: Union[Variable, int], mask: Optional[Tensor], cache: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape
        
        # Query with LoRA compression
        xq = self.wq_b(self.wq_a(x))
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.qk_head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.qk_head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.v_head_dim)
        
        # Split query and key into nope and rope parts
        xq_nope, xq_rope = xq[..., :self.qk_nope_head_dim], xq[..., self.qk_nope_head_dim:]
        xk_nope, xk_rope = xk[..., :self.qk_nope_head_dim], xk[..., self.qk_nope_head_dim:]
        
        # Apply RoPE only to rope parts
        freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
        xq_rope, xk_rope = apply_rotary_emb(xq_rope, xk_rope, freqs_cis)
        
        # Concatenate back
        xq = Tensor.cat(xq_nope, xq_rope, dim=-1)
        xk = Tensor.cat(xk_nope, xk_rope, dim=-1)
        
        # Handle KV cache
        if cache is not None:
            assert xk.dtype == xv.dtype == cache.dtype
            cache.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(
                Tensor.stack(xk, xv)
            ).realize()
            
            keys = cache[0].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xk
            values = cache[1].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xv
        else:
            keys = xk
            values = xv
        
        # Repeat KV heads for MQA
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # Transpose for attention computation
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled dot-product attention
        attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
        attn = attn.reshape(bsz, seqlen, -1)
        
        return self.wo(attn)


class MoEGate:
    """Mixture of Experts routing gate"""
    def __init__(self, hidden_size: int, n_routed_experts: int, topk: int = 4, linear=nn.Linear):
        self.gate = linear(hidden_size, n_routed_experts, bias=False)
        self.topk = topk
        self.n_routed_experts = n_routed_experts
    
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Compute scores for each expert
        scores = self.gate(x).softmax(axis=-1)
        
        # Select top-k experts
        topk_scores, topk_indices = scores.topk(self.topk, dim=-1)
        
        # Normalize selected scores
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        return topk_scores, topk_indices


class DeepseekV3MoE:
    """Mixture of Experts layer for Deepseek V3"""
    def __init__(self, hidden_size: int, intermediate_size: int, 
                 n_routed_experts: int, n_shared_experts: int = 2,
                 topk: int = 4, linear=nn.Linear):
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.topk = topk
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            DeepseekV3Expert(hidden_size, intermediate_size, linear)
            for _ in range(n_shared_experts)
        ])
        
        # Routed experts
        self.experts = nn.ModuleList([
            DeepseekV3Expert(hidden_size, intermediate_size, linear)
            for _ in range(n_routed_experts)
        ])
        
        # Routing gate
        self.gate = MoEGate(hidden_size, n_routed_experts, topk, linear)
    
    def __call__(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Process through shared experts
        shared_output = Tensor.zeros_like(x)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(x)
        shared_output = shared_output / len(self.shared_experts)
        
        # Get routing decisions
        topk_scores, topk_indices = self.gate(x)
        
        # Process through routed experts
        routed_output = Tensor.zeros_like(x)
        
        # Flatten for expert processing
        x_flat = x.reshape(-1, hidden_size)
        topk_scores_flat = topk_scores.reshape(-1, self.topk)
        topk_indices_flat = topk_indices.reshape(-1, self.topk)
        
        # Process each expert
        for expert_idx in range(self.n_routed_experts):
            # Find which tokens are routed to this expert
            expert_mask = (topk_indices_flat == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]
                
                # Get scores for this expert
                expert_scores = topk_scores_flat[expert_mask]
                score_mask = topk_indices_flat[expert_mask] == expert_idx
                expert_scores = (expert_scores * score_mask).sum(dim=-1, keepdim=True)
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by routing score
                expert_output = expert_output * expert_scores
                
                # Add back to output
                routed_output_flat = routed_output.reshape(-1, hidden_size)
                routed_output_flat[expert_mask] += expert_output
        
        routed_output = routed_output_flat.reshape(batch_size, seq_len, hidden_size)
        
        # Combine shared and routed expert outputs
        return shared_output + routed_output


class DeepseekV3Expert:
    """Single expert in MoE layer"""
    def __init__(self, hidden_size: int, intermediate_size: int, linear=nn.Linear):
        self.w1 = linear(hidden_size, intermediate_size, bias=False)
        self.w2 = linear(intermediate_size, hidden_size, bias=False)
        self.w3 = linear(hidden_size, intermediate_size, bias=False)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))


class DeepseekV3Block:
    def __init__(self, dim: int, intermediate_size: int, n_heads: int, n_kv_heads: int,
                 norm_eps: float, max_context: int, n_routed_experts: int = 0,
                 n_shared_experts: int = 0, qk_nope_head_dim: int = 128,
                 qk_rope_head_dim: int = 64, v_head_dim: int = 128,
                 q_lora_rank: int = 1536, linear=nn.Linear):
        self.attention = DeepseekV3Attention(
            dim, n_heads, n_kv_heads, max_context,
            qk_nope_head_dim, qk_rope_head_dim, v_head_dim, q_lora_rank, linear
        )
        
        # Use MoE or standard FFN based on layer configuration
        if n_routed_experts > 0:
            self.feed_forward = DeepseekV3MoE(
                dim, intermediate_size, n_routed_experts, n_shared_experts, linear=linear
            )
        else:
            # Standard FFN for non-MoE layers
            self.feed_forward = DeepseekV3Expert(dim, intermediate_size, linear)
        
        self.attention_norm = nn.RMSNorm(dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    
    def __call__(self, x: Tensor, start_pos: Union[Variable, int], mask: Optional[Tensor], cache: Optional[Tensor] = None):
        h = x + self.attention(self.attention_norm(x), start_pos, mask, cache=cache)
        return (h + self.feed_forward(self.ffn_norm(h))).contiguous()


class DeepseekV3Transformer:
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        norm_eps: float,
        vocab_size: int,
        max_seq_len: int = 8192,
        shard: Shard = None,
        linear=nn.Linear,
        n_kv_heads: Optional[int] = None,
        intermediate_size: int = 14336,
        n_routed_experts: int = 0,
        n_shared_experts: int = 0,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: int = 1536,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        moe_layer_indices: Optional[List[int]] = None,
        jit: bool = True,
    ):
        self.shard = shard if shard is not None else Shard("", 0, n_layers - 1, n_layers)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        if self.shard.is_first_layer():
            self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = []
        for i in range(n_layers):
            if self.shard.start_layer <= i <= self.shard.end_layer:
                # Check if this layer should use MoE
                use_moe = moe_layer_indices and i in moe_layer_indices
                layer = DeepseekV3Block(
                    dim=dim,
                    intermediate_size=intermediate_size,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    norm_eps=norm_eps,
                    max_context=max_seq_len,
                    n_routed_experts=n_routed_experts if use_moe else 0,
                    n_shared_experts=n_shared_experts if use_moe else 0,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    q_lora_rank=q_lora_rank,
                    linear=linear,
                )
                self.layers.append(layer)
            else:
                # Placeholder for layers outside shard
                self.layers.append(None)
        
        # Output norm and projection
        if self.shard.is_last_layer():
            self.norm = nn.RMSNorm(dim, norm_eps)
            self.output = linear(dim, vocab_size, bias=False)
        
        # JIT compilation
        self.forward_jit = TinyJit(self.forward_base) if jit else None
    
    def forward_base(self, x: Tensor, start_pos: Union[Variable, int], cache: Optional[List[Tensor]] = None):
        seqlen = x.shape[1]
        mask = None
        if seqlen > 1:
            mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=x.dtype, device=x.device)
            mask = mask.triu(start_pos + 1).realize()
        
        # Token embeddings
        if self.shard.is_first_layer() and x.dtype == dtypes.int:
            h = self.tok_embeddings(x)
        else:
            h = x
        
        # Process through layers in shard
        if cache is None:
            cache = [None for _ in range(self.shard.start_layer, self.shard.end_layer + 1)]
        
        for i, (layer, c) in enumerate(zip(self.layers[self.shard.start_layer:self.shard.end_layer + 1], cache)):
            if layer is not None:
                h = layer(h, start_pos, mask, cache=c)
        
        # Output projection
        if self.shard.is_last_layer():
            h = self.norm(h)
            logits = self.output(h).float().realize()
            return logits
        else:
            return h
    
    def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
        if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
            return self.forward_jit(x, Variable("start_pos", 1, self.max_seq_len).bind(start_pos), cache=cache)
        return self.forward_base(x, start_pos, cache=cache)
    
    def __call__(self, x: Tensor, start_pos: Union[Variable, int], cache: Optional[List[Tensor]] = None):
        return self.forward(x, start_pos, cache=cache)


def convert_deepseek_v3_from_huggingface(weights: Dict[str, Tensor], model: DeepseekV3Transformer, 
                                        n_heads: int, n_kv_heads: int) -> Dict[str, Tensor]:
    """Convert Hugging Face Deepseek V3 weights to tinygrad format"""
    def permute(v: Tensor, n_heads: int):
        # Rearrange attention weights for tinygrad format
        return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])
    
    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    
    # Add layer-specific mappings
    for l in range(model.n_layers):
        # Attention weights
        keymap.update({
            f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight",
            f"model.layers.{l}.self_attn.q_a_proj.weight": f"layers.{l}.attention.wq_a.weight",
            f"model.layers.{l}.self_attn.q_b_proj.weight": f"layers.{l}.attention.wq_b.weight",
            f"model.layers.{l}.self_attn.k_proj.weight": f"layers.{l}.attention.wk.weight",
            f"model.layers.{l}.self_attn.v_proj.weight": f"layers.{l}.attention.wv.weight",
            f"model.layers.{l}.self_attn.o_proj.weight": f"layers.{l}.attention.wo.weight",
            f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight",
        })
        
        # MoE weights (if applicable)
        keymap.update({
            f"model.layers.{l}.mlp.gate.weight": f"layers.{l}.feed_forward.gate.gate.weight",
        })
        
        # Expert weights
        for e in range(64):  # Max experts
            keymap.update({
                f"model.layers.{l}.mlp.experts.{e}.w1.weight": f"layers.{l}.feed_forward.experts.{e}.w1.weight",
                f"model.layers.{l}.mlp.experts.{e}.w2.weight": f"layers.{l}.feed_forward.experts.{e}.w2.weight",
                f"model.layers.{l}.mlp.experts.{e}.w3.weight": f"layers.{l}.feed_forward.experts.{e}.w3.weight",
            })
        
        # Shared expert weights
        for e in range(2):  # Typically 2 shared experts
            keymap.update({
                f"model.layers.{l}.mlp.shared_experts.{e}.w1.weight": f"layers.{l}.feed_forward.shared_experts.{e}.w1.weight",
                f"model.layers.{l}.mlp.shared_experts.{e}.w2.weight": f"layers.{l}.feed_forward.shared_experts.{e}.w2.weight",
                f"model.layers.{l}.mlp.shared_experts.{e}.w3.weight": f"layers.{l}.feed_forward.shared_experts.{e}.w3.weight",
            })
    
    sd = {}
    for k, v in weights.items():
        if ".rotary_emb." in k:
            continue
        
        v = v.to(Device.DEFAULT)
        
        # Apply attention weight permutation if needed
        if "model.layers" in k and "q_b_proj" in k:
            v = permute(v, n_heads)
        elif "model.layers" in k and "k_proj" in k:
            v = permute(v, n_kv_heads)
        
        # Map to tinygrad naming
        if k in keymap:
            sd[keymap[k]] = v
        else:
            # Keep unmapped weights (might be needed)
            sd[k] = v
    
    return sd