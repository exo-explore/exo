"""
Utility methods used by LLMs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from typing import Optional

def rope_embed(
  head_dim: int,
  input_embeddings: torch.Tensor,
  position_ids: Optional[torch.Tensor],
) -> torch.Tensor:
  """
  Wrapper of rotary embeddings using pytorch module

  Args:
    input_embeddings (torch.Tensor): token embeddings from input
    position_ids (torch.Tensor): position ids of tokens

  Returns:
    torch.Tensor: output with RoPE applied
  """
  try:
    rotary_emb = RotaryPositionalEmbeddings(head_dim)
    output = rotary_emb.forward(
      input_embeddings,
      input_pos=position_ids
    )
  except Exception:
    raise

  return output


