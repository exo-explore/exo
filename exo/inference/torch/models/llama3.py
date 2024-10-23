"""
llama3 model

Written with pytorch using torchtune and other methods
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings, KVCache

class LlamaBlock(nn.Module):
  """
  Encoder block class for the LLaMA model without residual connections.
  """
  def __init__(
    self,
    dim,
    heads,
    num_kv_heads,
    head_dim,
    ff_dim,
    rms_norm_eps,
    attention_dropout=0.0,
    use_bias=False,
    max_seq_len=4096,
    pos_embeddings=None
  ):
    super(LlamaBlock, self).__init__()

    # Define linear projections for Q, K, V, and Output
    self.q_proj = nn.Linear(dim, heads * head_dim, bias=use_bias)
    self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.output_proj = nn.Linear(heads * head_dim, dim, bias=use_bias)

    # Define optional query normalization
    self.q_norm = nn.LayerNorm(head_dim, eps=rms_norm_eps)

    # MultiHeadAttention from torchtune
    self.attn = MultiHeadAttention(
      embed_dim=dim,
      num_heads=heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      q_proj=self.q_proj,
      k_proj=self.k_proj,
      v_proj=self.v_proj,
      output_proj=self.output_proj,
      pos_embeddings=pos_embeddings,
      q_norm=self.q_norm,
      k_norm=self.q_norm,
      kv_cache=None,
      max_seq_len=max_seq_len,
      is_causal=True,
      attn_dropout=attention_dropout
    )

    # RMSNorm layers before and after attention and feed-forward layers
    self.norm1 = nn.LayerNorm(dim, eps=rms_norm_eps)
    self.norm2 = nn.LayerNorm(dim, eps=rms_norm_eps)

    # Feed-forward layer with SwiGLU activation
    self.feed_forward = nn.Sequential(
      nn.Linear(dim, ff_dim),
      nn.GLU(),  # SwiGLU approximation
      nn.Linear(ff_dim // 2, dim)
    )

  def forward(
      self,
      x,
      kv_cache: Optional[KVCache] = None,
      attention_mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, KVCache]:
    """
    Forward pass with integrated attention and key-value caching.

    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
      kv_cache (Optional[KVCache]): KVCache object for managing past key-value states.
      attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len).
      input_pos (Optional[torch.Tensor]): Position IDs tensor of shape (batch_size, seq_len).

    Returns:
      Tuple[torch.Tensor, KVCache]: 
        - x (torch.Tensor): Output tensor of shape (batch_size, seq_len, dim).
        - kv_cache (KVCache): Updated KVCache object.
    """
    # Apply normalization before attention
    residual = x
    x = self.norm1(x)

    # Compute Q, K, V projections
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Initialize or update KVCache
    if kv_cache is None:
      kv_cache = KVCache(
        batch_size=x.size(0),
        max_seq_len=x.size(1),
        num_heads=self.attn.num_heads,
        head_dim=self.attn.head_dim,
        dtype=x.dtype
      )

    # Update KVCache with new key-value pairs
    k_val, v_val = kv_cache.update(k, v)

    # Apply MultiHeadAttention with key-value caching
    x = self.attn(q, k_val, v_val, mask=attention_mask, input_pos=input_pos)

    # Residual connection
    x = x + residual

    # Apply feed-forward network with residual connection
    residual = x
    x = self.norm2(x)
    x = self.feed_forward(x)
    x = x + residual

    return x, kv_cache

class LlamaModel(nn.Module):
  """
  LlamaModel is a pure PyTorch implementation of the LLaMA architecture
  """

  def __init__(self, config, tokenizer):
    """
    Initialize the LlamaModel.

    Args:
      config (dict): Configuration dictionary containing model parameters.
        - hidden_size (int): Size of the hidden layers.
        - num_hidden_layers (int): Number of transformer layers.
        - num_attention_heads (int): Number of attention heads.
        - intermediate_size (int): Size of the intermediate (feed-forward) layers.
        - vocab_size (int): Vocabulary size for the embedding layer.
        - max_position_embeddings (int): Maximum number of positional embeddings.
        - rms_norm_eps (float): Epsilon for RMS normalization.
        - head_dim (int): Dimension of each attention head.
        - attention_dropout (float): Dropout rate for attention layers.
      tokenizer: Tokenizer used for input preprocessing.
    """
    super(LlamaModel, self).__init__()

    # Load configurations from config
    self.config = config
    self.hidden_size = config['hidden_size']
    self.num_layers = config['num_hidden_layers']
    self.num_heads = config['num_attention_heads']
    self.num_kv_heads = config['num_key_value_heads']
    self.intermediate_size = config['intermediate_size']
    self.vocab_size = config['vocab_size']
    self.max_position_embeddings = config['max_position_embeddings']
    self.rms_norm_eps = config['rms_norm_eps']
    self.head_dim = config['head_dim']
    self.attention_dropout = config.get('attention_dropout', 0.0)

    # Model layers
    self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
    self.rotary_pos_emb = RotaryPositionalEmbeddings(
      self.hidden_size // self.num_heads,
      config['rope_scaling']['original_max_position_embeddings'],
      config['rope_theta']
    )
    self.layers = nn.ModuleList([
      LlamaBlock(
        dim=self.hidden_size,
        heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        ff_dim=self.intermediate_size,
        rms_norm_eps=self.rms_norm_eps,
        attention_dropout=self.attention_dropout,
        use_bias=config.get('attention_bias', False)
      ) for _ in range(self.num_layers)
    ])
    self.norm = nn.LayerNorm(self.hidden_size, eps=self.rms_norm_eps)
    self.to_logits = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pos_ids: Optional[torch.Tensor] = None,
    past_kv_cache: Optional[KVCache] = None,
    return_hidden_states: bool = False
  ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], KVCache]:
    """
    Forward pass with integrated position ID handling, attention mask, and optional KVCache.

    Args:
      input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
      attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).
      pos_ids (Optional[torch.Tensor]): Position IDs. If None, they are calculated automatically.
      past_kv_cache (Optional[KVCache]): Optional KVCache for efficient generation.
        If provided, it stores past key-value states for faster autoregressive inference.
      return_hidden_states (bool): Whether to return hidden states from each layer.

    Returns:
      Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], KVCache]:
        - logits (torch.Tensor): Output logits of shape (batch_size, seq_len, vocab_size).
        - hidden_states (Optional[Tuple[torch.Tensor]]): Hidden states from each layer, if return_hidden_states is True.
        - past_kv_cache (KVCache): Updated KVCache object.
    """
    batch_size, seq_len = input_ids.shape

    # Create initial embeddings
    x = self.embed(input_ids)

    # Initialize position IDs if not provided
    if pos_ids is None:
      past_seen_tokens = past_kv_cache.size if past_kv_cache is not None else 0
      pos_ids = torch.arange(
        past_seen_tokens,
        past_seen_tokens + seq_len,
        device=input_ids.device
      ).unsqueeze(0).expand(batch_size, -1)

    # Reshape x to prepare for rotary embeddings: (batch_size, seq_len, num_heads, head_dim)
    x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    # Apply rotary positional embeddings
    x = self.rotary_pos_emb(
      x=x,
      input_pos=pos_ids
    )

    # Reshape x back to original shape: (batch_size, seq_len, hidden_size)
    x = x.view(batch_size, seq_len, self.hidden_size)

    # Initialize or use the provided KVCache
    if past_kv_cache is None:
      past_kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=self.max_position_embeddings,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dtype=x.dtype
      )

    # Apply attention mask if provided (convert to appropriate format)
    if attention_mask is not None:
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
      attention_mask = (1.0 - attention_mask) * -1e4  # Convert to large negative values

    # Track hidden states if required
    hidden_states = []

    # Forward pass through layers with KVCache
    for layer_idx, layer in enumerate(self.layers):
      x, k_val, v_val = layer(x, past_kv_cache, layer_idx, attention_mask)

      # Update KVCache
      past_kv_cache.update(k_val, v_val)

      if return_hidden_states:
        hidden_states.append(x)

    # Apply final layer normalization
    x = self.norm(x)

    # Compute logits
    logits = self.to_logits(x)

    # Prepare the return values
    if return_hidden_states:
      return logits, tuple(hidden_states), past_kv_cache
    else:
      return logits, None, past_kv_cache
