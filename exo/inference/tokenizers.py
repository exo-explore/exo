import traceback
from transformers import AutoTokenizer, AutoProcessor
from exo.helpers import DEBUG

async def resolve_tokenizer(model_id: str):
  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {model_id}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {model_id}")
    return AutoTokenizer.from_pretrained(model_id)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {model_id}. Falling back to tinygrad tokenizer. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  raise ValueError(f"[TODO] Unsupported model: {model_id}")
