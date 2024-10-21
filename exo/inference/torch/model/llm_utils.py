"""
Utility methods used by LLMs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

def rope_embed(
  self,
  input_embeddings: torch.Tensor,
  position_ids: torch.Tensor,
):
  """
  Wrapper of rotary embeddings using pytorch module

  Args:
    input_embeddings (torch.Tensor): token embeddings from input  
    position_ids (torch.Tensor): position ids of tokens
  """
  rotary_emb = RotaryPositionalEmbeddings()


