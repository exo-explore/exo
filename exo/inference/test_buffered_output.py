import pytest
from transformers import AutoProcessor

from .buffered_output import BufferedOutput
from ..download.new_shard_download import exo_home


@pytest.fixture
def tokenizer():
  return AutoProcessor.from_pretrained(
    exo_home() / "downloads" / "mlx-community/Llama-3.2-1B-Instruct-4bit".replace("/", "--")
  )


def test_stop_sequence(tokenizer):
  buffered_output = BufferedOutput(
    tokenizer=tokenizer,
    stop_sequences=['stop'],
    max_tokens=100,
    eos_token_id=tokenizer.eos_token_id
  )

  tokens = tokenizer.encode('stop', add_special_tokens=False)

  for token in tokens:
    buffered_output.append(token)

  assert buffered_output.is_finished
  assert buffered_output.finish_reason == 'stop'
  assert buffered_output.next_tokens() == []
