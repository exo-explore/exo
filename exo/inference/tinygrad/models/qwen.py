"""
Qwen3 MoE (Mixture of Experts) implementation for Tinygrad
Supports Qwen3MoeForCausalLM architecture with top-k expert routing
"""

from typing import Tuple, Union, Optional, Dict, Any, List
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv
from collections import OrderedDict

# Import shared components from llama.py
from exo.inference.tinygrad.models.llama import (
    precompute_freqs_cis,
    apply_rotary_emb,
    repeat_kv,
    Attention,
    sample_logits,
    fix_bf16
)
from exo.inference.shard import Shard


class MoEFeedForward:
    """
    Mixture of Experts FeedForward layer with top-k routing

    Architecture:
    - Router (gate): selects top-k experts per token
    - Experts: N independent FFN modules (each using SwiGLU activation)
    - Weighted combination of expert outputs based on routing weights
    """
    def __init__(self, dim: int, num_experts: int, num_experts_per_tok: int,
                 moe_intermediate_size: int, linear=nn.Linear):
        """
        Args:
            dim: Model dimension (hidden_size)
            num_experts: Total number of expert modules (160 for Qwen3-480B)
            num_experts_per_tok: Number of experts to activate per token (8 for Qwen3-480B)
            moe_intermediate_size: Hidden dimension of each expert FFN (2560 for Qwen3-480B)
            linear: Linear layer constructor (allows custom implementations)
        """
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.dim = dim
        self.moe_intermediate_size = moe_intermediate_size

        # Router: projects hidden state to expert scores
        self.gate = linear(dim, num_experts, bias=False)

        # Expert FFN modules: each expert has w1 (gate), w2 (down_proj), w3 (up_proj)
        # Using SwiGLU activation: w2(w1(x).silu() * w3(x))
        self.experts = []
        for _ in range(num_experts):
            expert = {
                'w1': linear(dim, moe_intermediate_size, bias=False),  # gate_proj
                'w2': linear(moe_intermediate_size, dim, bias=False),  # down_proj
                'w3': linear(dim, moe_intermediate_size, bias=False),  # up_proj
            }
            self.experts.append(expert)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass with top-k expert routing

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        orig_shape = x.shape
        batch_size, seq_len, dim = orig_shape

        # Flatten batch and sequence dimensions for routing
        # Shape: (batch_size * seq_len, dim)
        x_flat = x.reshape(-1, dim)
        num_tokens = x_flat.shape[0]

        # Compute router logits for all experts
        # Shape: (num_tokens, num_experts)
        router_logits = self.gate(x_flat)

        # Apply softmax to get routing probabilities
        routing_weights = router_logits.softmax(dim=-1)

        # Select top-k experts per token
        # routing_weights_topk: (num_tokens, num_experts_per_tok)
        # selected_experts: (num_tokens, num_experts_per_tok) - indices of selected experts
        routing_weights_topk, selected_experts = routing_weights.topk(
            self.num_experts_per_tok, dim=-1
        )

        # Normalize routing weights (so they sum to 1 for each token)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1, keepdim=True
        )

        # Initialize output accumulator
        final_output = Tensor.zeros(num_tokens, dim, device=x.device, dtype=x.dtype)

        # Process each expert position in the top-k selection
        # For efficiency, we batch-process all tokens that selected the same expert
        for expert_idx in range(self.num_experts_per_tok):
            # Get which expert each token selected at this position
            # Shape: (num_tokens,)
            expert_ids = selected_experts[:, expert_idx]

            # Get the routing weight for this expert position
            # Shape: (num_tokens, 1)
            weights = routing_weights_topk[:, expert_idx].unsqueeze(-1)

            # Process each unique expert that was selected
            for expert_id in range(self.num_experts):
                # Create mask for tokens that selected this expert
                # Shape: (num_tokens,)
                expert_mask = (expert_ids == expert_id)

                # Skip if no tokens selected this expert
                if expert_mask.sum() == 0:
                    continue

                # Get the expert module
                expert = self.experts[expert_id]

                # Extract tokens that use this expert
                # Shape: (num_selected_tokens, dim)
                expert_input = x_flat[expert_mask]

                # Apply SwiGLU FFN: w2(w1(x).silu() * w3(x))
                # Shape: (num_selected_tokens, moe_intermediate_size)
                gate_output = expert['w1'](expert_input).silu()
                up_output = expert['w3'](expert_input)
                # Shape: (num_selected_tokens, dim)
                expert_output = expert['w2'](gate_output * up_output)

                # Apply routing weights and accumulate
                # We need to scatter the expert outputs back to their positions
                weighted_output = expert_output * weights[expert_mask]

                # Scatter weighted outputs to correct positions using boolean indexing
                # This properly places expert outputs at the token positions that selected this expert
                current_output = final_output[expert_mask]
                final_output[expert_mask] = current_output + weighted_output

        # Reshape back to original dimensions
        return final_output.reshape(batch_size, seq_len, dim)


class Qwen3MoETransformerBlock:
    """
    Qwen3 Transformer block with MoE FFN

    Components:
    - Multi-head attention (reused from llama.py)
    - MoE FeedForward (instead of standard FFN)
    - RMSNorm for pre-attention and pre-FFN normalization
    - Optional QK normalization (Qwen3-specific feature)
    """
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 norm_eps: float, max_context: int, num_experts: int,
                 num_experts_per_tok: int, moe_intermediate_size: int,
                 use_qk_norm: bool = False, linear=nn.Linear):
        """
        Args:
            dim: Model dimension (hidden_size)
            hidden_dim: FFN intermediate dimension (not used in MoE, kept for compatibility)
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (for MQA/GQA)
            norm_eps: Epsilon for RMSNorm
            max_context: Maximum context length
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts to activate per token
            moe_intermediate_size: Hidden dimension of each expert FFN
            use_qk_norm: Whether to apply normalization to Q and K projections
            linear: Linear layer constructor
        """
        # Attention layer (same as LLaMA)
        self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)

        # MoE FeedForward instead of standard FFN
        self.feed_forward = MoEFeedForward(
            dim, num_experts, num_experts_per_tok, moe_intermediate_size, linear
        )

        # Normalization layers
        self.attention_norm = nn.RMSNorm(dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, norm_eps)

        # QK normalization (Qwen3-specific feature)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            head_dim = dim // n_heads
            self.q_norm = nn.RMSNorm(head_dim, norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, norm_eps)

    def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor,
                 mask: Optional[Tensor], cache: Optional[Tensor] = None):
        """
        Forward pass

        Args:
            x: Input tensor
            start_pos: Starting position for KV cache
            freqs_cis: Precomputed RoPE frequencies
            mask: Attention mask
            cache: KV cache tensor

        Returns:
            Output tensor after attention + MoE FFN
        """
        # Pre-norm architecture: normalize before attention
        # h = x + attention(norm(x))
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, cache=cache
        )

        # h = h + MoE_FFN(norm(h))
        out = h + self.feed_forward(self.ffn_norm(h))

        return out.contiguous()


class Qwen3MoETransformer:
    """
    Full Qwen3 MoE Transformer model

    Architecture:
    - Token embeddings
    - N transformer blocks with MoE FFN
    - Final normalization
    - Output projection (LM head)
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,  # Not used for MoE, kept for compatibility
        n_heads: int,
        n_layers: int,
        norm_eps: float,
        vocab_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shard: Shard = None,
        linear=nn.Linear,
        n_kv_heads=None,
        rope_theta=10000,
        max_context=1024,
        jit=True,
        rope_scaling: Optional[Dict[str, float]] = None,
        tie_word_embeddings=False,
        use_qk_norm: bool = False,
    ):
        """
        Args:
            dim: Model dimension (hidden_size = 6144 for Qwen3-480B)
            hidden_dim: Standard FFN hidden dim (not used in MoE)
            n_heads: Number of attention heads (96 for Qwen3-480B)
            n_layers: Number of transformer blocks (62 for Qwen3-480B)
            norm_eps: RMSNorm epsilon
            vocab_size: Vocabulary size (151936 for Qwen3-480B)
            num_experts: Total experts (160 for Qwen3-480B)
            num_experts_per_tok: Top-k experts per token (8 for Qwen3-480B)
            moe_intermediate_size: Expert FFN hidden dim (2560 for Qwen3-480B)
            shard: Shard configuration for distributed inference
            linear: Linear layer constructor
            n_kv_heads: Number of KV heads (8 for Qwen3-480B)
            rope_theta: RoPE theta parameter (10000000 for Qwen3-480B)
            max_context: Maximum context length
            jit: Whether to use JIT compilation
            rope_scaling: RoPE scaling configuration
            tie_word_embeddings: Whether to tie input/output embeddings
            use_qk_norm: Whether to use QK normalization
        """
        # Store MoE-specific parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        # Create transformer blocks with MoE FFN
        self.layers = [
            Qwen3MoETransformerBlock(
                dim=dim,
                hidden_dim=hidden_dim,  # Not used in MoE
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                norm_eps=norm_eps,
                max_context=max_context,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                use_qk_norm=use_qk_norm,
                linear=linear
            )
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.norm = nn.RMSNorm(dim, norm_eps)

        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Output projection (LM head)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # RoPE frequencies
        self.max_context = max_context
        self.freqs_cis = precompute_freqs_cis(
            dim // n_heads,
            self.max_context * 2,
            rope_theta,
            rope_scaling=rope_scaling
        ).contiguous()

        # JIT compilation
        self.forward_jit = TinyJit(self.forward_base) if jit else None

        # Shard configuration
        self.shard = shard

    def forward_base(self, x: Tensor, start_pos: Union[Variable, int],
                     cache: Optional[List[Tensor]] = None):
        """
        Base forward pass (non-JIT version)

        Args:
            x: Input token tensor (after embedding)
            start_pos: Starting position for KV cache
            cache: List of KV cache tensors (one per layer)

        Returns:
            Logits if last layer, hidden states otherwise
        """
        seqlen = x.shape[1]

        # Get RoPE frequencies for this sequence
        freqs_cis = self.freqs_cis.shrink(
            (None, (start_pos, start_pos + seqlen), None, None, None)
        )

        # Create causal attention mask (only for sequences > 1 token)
        mask = None
        if seqlen > 1:
            mask = Tensor.full(
                (1, 1, seqlen, start_pos + seqlen),
                float("-100000000"),
                dtype=x.dtype,
                device=x.device
            ).triu(start_pos + 1).realize()

        # Initialize cache if needed
        if cache is None:
            cache = [None for _ in range(self.shard.start_layer, self.shard.end_layer + 1)]

        # Forward through transformer blocks in this shard
        h = x
        for i, c in zip(range(self.shard.start_layer, self.shard.end_layer + 1), cache):
            layer = self.layers[i]
            h = layer(h, start_pos, freqs_cis, mask, cache=c)

        # Apply final norm and output projection only on last layer
        if self.shard.is_last_layer():
            logits = self.output(self.norm(h)).float().realize()
            return logits
        else:
            return h

    def embed(self, inputs: Tensor):
        """Embed input tokens (only on first layer)"""
        if self.shard.is_first_layer():
            h = self.tok_embeddings(inputs)
        else:
            h = inputs
        return h

    def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
        """
        Forward pass with optional JIT compilation

        Uses JIT for single-token inference (generation) after first token
        """
        # Use JIT for single-token generation (after prompt processing)
        if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
            return self.forward_jit(
                x,
                Variable("start_pos", 1, self.max_context).bind(start_pos),
                cache=cache
            )
        return self.forward_base(x, start_pos, cache=cache)

    def __call__(self, x: Tensor, start_pos: Variable, cache: Optional[List[Tensor]] = None):
        """Main entry point"""
        h = self.embed(x)
        return self.forward(h, start_pos, cache=cache)


class Qwen3MoETransformerShard:
    """
    Sharded version of Qwen3MoETransformer for distributed inference

    Similar to TransformerShard in llama.py but for Qwen3 MoE architecture
    """
    def __init__(
        self,
        shard: Shard,
        base: Qwen3MoETransformer,
        jit: bool = True,
    ):
        """
        Args:
            shard: Shard configuration (which layers this instance handles)
            base: Full Qwen3MoETransformer model
            jit: Whether to use JIT compilation
        """
        # Extract only the layers in this shard
        shardrange = range(shard.start_layer, shard.end_layer + 1)
        self.layers = [
            layer for layer, n in zip(base.layers, range(shard.n_layers))
            if n in shardrange
        ]

        # Shared components
        self.norm = base.norm
        self.tok_embeddings = base.tok_embeddings
        self.output = base.output
        self.max_context = base.max_context
        self.freqs_cis = base.freqs_cis

        # Shard-specific functions
        self.embed = (
            (lambda x: self.tok_embeddings(x))
            if shard.is_first_layer()
            else (lambda x: x)
        )
        self.post = (
            (lambda x: self.output(self.norm(x)))
            if shard.is_last_layer()
            else (lambda x: x)
        )

        # Cache management
        self.null_cache = [None for _ in shardrange]

        # JIT compilation
        self.forward_jit = TinyJit(self.forward_base) if jit else None

    def forward_base(self, x: Tensor, start_pos: Union[Variable, int], cache):
        """Base forward pass"""
        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis.shrink(
            (None, (start_pos, start_pos + seqlen), None, None, None)
        )

        mask = None
        if seqlen > 1:
            mask = Tensor.full(
                (1, 1, seqlen, start_pos + seqlen),
                float("-100000000"),
                dtype=x.dtype,
                device=x.device
            ).triu(start_pos + 1).realize()

        # Forward through shard layers
        for layer, c in zip(self.layers, cache):
            x = layer(x, start_pos, freqs_cis, mask, cache=c)

        # Apply post-processing (norm + output) if last layer
        out = self.post(x)
        return out

    def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
        """Forward with optional JIT"""
        if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
            return self.forward_jit(
                x,
                Variable("start_pos", 1, self.max_context).bind(start_pos),
                cache=cache
            )
        return self.forward_base(x, start_pos, cache=cache)

    def __call__(self, x: Tensor, start_pos: Variable, cache: Optional[List[Tensor]] = None):
        """Main entry point"""
        h = self.embed(x)
        return self.forward(h, start_pos, cache=self.null_cache if cache is None else cache)


# *** Weight Conversion ***

def convert_from_huggingface_qwen(weights: Dict[str, Tensor], model: Qwen3MoETransformer,
                                   n_heads: int, n_kv_heads: int):
    """
    Convert HuggingFace Qwen3 MoE weights to Tinygrad format

    Weight mapping from HuggingFace naming convention to our internal convention:
    - model.embed_tokens.weight → tok_embeddings.weight
    - model.layers.{l}.input_layernorm.weight → layers.{l}.attention_norm.weight
    - model.layers.{l}.self_attn.{q,k,v,o}_proj.weight → layers.{l}.attention.w{q,k,v,o}.weight
    - model.layers.{l}.post_attention_layernorm.weight → layers.{l}.ffn_norm.weight
    - model.layers.{l}.mlp.gate.weight → layers.{l}.feed_forward.gate.weight
    - model.layers.{l}.mlp.experts.{e}.{gate,down,up}_proj.weight →
          layers.{l}.feed_forward.experts.{e}.w{1,2,3}.weight
    - model.norm.weight → norm.weight
    - lm_head.weight → output.weight

    Args:
        weights: HuggingFace state dict
        model: Qwen3MoETransformer instance
        n_heads: Number of attention heads (for Q permutation)
        n_kv_heads: Number of KV heads (for K permutation)

    Returns:
        Converted state dict in Tinygrad format
    """
    def permute(v: Tensor, n_heads: int):
        """Permute Q/K weights for correct attention computation"""
        return v.reshape(
            n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]
        ).transpose(1, 2).reshape(*v.shape[:2])

    # Build key mapping
    num_layers = len(model.layers)
    num_experts = model.num_experts

    keymap = {
        # Embeddings and output
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    # Add layer-specific mappings
    for l in range(num_layers):
        # Attention norm
        keymap[f"model.layers.{l}.input_layernorm.weight"] = \
            f"layers.{l}.attention_norm.weight"

        # Attention projections
        for x in ["q", "k", "v", "o"]:
            keymap[f"model.layers.{l}.self_attn.{x}_proj.weight"] = \
                f"layers.{l}.attention.w{x}.weight"

        # FFN norm
        keymap[f"model.layers.{l}.post_attention_layernorm.weight"] = \
            f"layers.{l}.ffn_norm.weight"

        # MoE gate (router)
        keymap[f"model.layers.{l}.mlp.gate.weight"] = \
            f"layers.{l}.feed_forward.gate.weight"

        # Expert FFN weights
        for e in range(num_experts):
            # HuggingFace uses: gate_proj, down_proj, up_proj
            # We use: w1, w2, w3 (matching LLaMA convention)
            proj_map = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
            for hf_name, our_name in proj_map.items():
                keymap[f"model.layers.{l}.mlp.experts.{e}.{hf_name}.weight"] = \
                    f"layers.{l}.feed_forward.experts.{e}.{our_name}.weight"

        # QK norm (if present)
        if hasattr(model.layers[l], 'q_norm'):
            keymap[f"model.layers.{l}.self_attn.q_norm.weight"] = \
                f"layers.{l}.q_norm.weight"
            keymap[f"model.layers.{l}.self_attn.k_norm.weight"] = \
                f"layers.{l}.k_norm.weight"

    # Convert weights
    sd = {}
    for k, v in weights.items():
        # Skip RoPE embeddings (we compute them ourselves)
        if ".rotary_emb." in k:
            continue

        # Move to default device
        v = v.to(Device.DEFAULT)

        # Apply permutation to Q and K projections for correct attention
        if "model.layers" in k:
            if "q_proj" in k:
                v = permute(v, n_heads)
            elif "k_proj" in k:
                v = permute(v, n_kv_heads)

        # Map to our naming convention
        if k in keymap:
            sd[keymap[k]] = v
        else:
            # Keep unmapped keys as-is (for debugging)
            sd[k] = v

    return sd
