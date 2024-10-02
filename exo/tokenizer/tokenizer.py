import json
import os
from typing import Dict, List, Set

import tiktoken

class Tokenizer:
  def __init__(self, model_path: str):
    # Load tokenizer configuration files
    tokenizer_config = self._load_json(os.path.join(model_path, 'tokenizer_config.json'))
    special_tokens_map = self._load_json(os.path.join(model_path, 'special_tokens_map.json'))
    tokenizer_file = os.path.join(model_path, 'tokenizer.json')

    if os.path.exists(tokenizer_file):
      tokenizer_data = self._load_json(tokenizer_file)
      merges = tokenizer_data.get('model', {}).get('merges', [])
      vocab = tokenizer_data.get('model', {}).get('vocab', {})
      vocab = {token: int(idx) for token, idx in vocab.items()}
    else:
      # Fallback to merges.txt and vocab.json if tokenizer.json is not available
      merges_file = os.path.join(model_path, 'merges.txt')
      vocab_file = os.path.join(model_path, 'vocab.json')
      merges = self._load_merges(merges_file)
      vocab = self._load_json(vocab_file)
      vocab = {token: int(idx) for token, idx in vocab.items()}

    # Create mergeable ranks with string keys
    self.mergeable_ranks = {merge: idx for idx, merge in enumerate(merges)}
    self.num_base_tokens = len(vocab)

    # Define special tokens
    self.special_tokens: Dict[str, int] = {}
    if special_tokens_map:
      for token_name, token_value in special_tokens_map.items():
        if isinstance(token_value, dict):
          # Extract the actual token string
          token_str = token_value.get('content', '')
          token_id = token_value.get('id', len(vocab) + len(self.special_tokens))
        else:
          token_str = token_value
          token_id = vocab.get(token_str, len(vocab) + len(self.special_tokens))
        self.special_tokens[token_str] = token_id
    else:
      # Default special tokens if not defined
      self.special_tokens = {
        '<|bos|>': len(vocab),
        '<|eos|>': len(vocab) + 1
      }

    # Initialize tiktoken encoding
    self.model = tiktoken.Encoding(
      name=os.path.basename(model_path),
      pat_str=tokenizer_config.get('pattern', r'\S+|\n'),
      mergeable_ranks=self.mergeable_ranks,
      special_tokens=self.special_tokens
    )

  def _load_json(self, path: str) -> Dict:
    if os.path.exists(path):
      with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    return {}

  def _load_merges(self, path: str) -> List[str]:
    if os.path.exists(path):
      with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        # Skip the first line if it's a header
        if lines and lines[0].startswith('#'):
          lines = lines[1:]
        return lines
    return []

  @property
  def bos_id(self) -> int:
    return self.special_tokens.get('<|bos|>', None)

  @property
  def eos_id(self) -> int:
    return self.special_tokens.get('<|eos|>', None)

  @property
  def stop_tokens(self) -> Set[int]:
    return {self.eos_id} if self.eos_id is not None else set()

  def decode(self, tokens: List[int]) -> str:
    return self.model.decode(tokens)

  def encode(self, text: str, allow_special: bool = False) -> List[int]:
    allowed_special = set(self.special_tokens.keys()) if allow_special else set()
    return self.model.encode(
      text,
      allowed_special=allowed_special,
      disallowed_special=set()
    )
