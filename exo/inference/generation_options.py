from typing import Optional


class GenerationOptions:
  max_completion_tokens: Optional[int] = None

  def __init__(self, max_completion_tokens: Optional[int] = None):
    self.max_completion_tokens = max_completion_tokens
