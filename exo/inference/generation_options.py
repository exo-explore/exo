from typing import Optional, List


class GenerationOptions:
  max_completion_tokens: Optional[int] = None

  # Textual stop sequences that will halt generation when encountered
  stop: Optional[List[str]] = None

  def __init__(self, max_completion_tokens: Optional[int] = None, stop: Optional[List[str]] = None):
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop
