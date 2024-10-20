"""
Simple model test using basic pytorch/huggingface LLM model loading, inference and generation
with logit sampling
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_simple(prompt: str):
  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2-0.5B-Instruct",
      torch_dtype="auto",
      device_map="auto"
  )

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt}
  ]
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  model_inputs = tokenizer([text], return_tensors="pt")

  print(f"model_inputs:\n{model_inputs}")

  print(f"generation_config:\n{model.generation_config}")

  generated_ids = model.generate(
      model_inputs.input_ids,
      attention_mask=model_inputs.attention_mask,
      max_new_tokens=512,
      do_sample=True
  )

  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  print(f"Prompt: {prompt}\n")
  print(f"Response: {response}\n")

if __name__ == "__main__":
  run_simple(
    "In a single word only, what is the last name of the current president of the USA?"
  )
