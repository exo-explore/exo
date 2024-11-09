import asyncio
import time
import uuid
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from exo.inference.tokenizers import resolve_tokenizer
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader, RepoProgressEvent
from exo.helpers import pretty_print_bytes_per_second

async def run_bench(inference_engine: InferenceEngine, tokenizer, shard: Shard, num_tokens: int = 500, verbose=True):
  req_id = str(uuid.uuid4())
  start_time = time.time()
  total_tokens = 0
  tokens_over_time = []
  times = []

  prompt = tokenizer.apply_chat_template([{"role": "user", "content": "write an essay about the importance of the internet"}], tokenize=False, add_generation_prompt=True)
  if verbose: print(f"Prompt: {prompt}\n", flush=True)
  resp, inference_state, is_finished = await inference_engine.infer_prompt(req_id, shard, prompt)
  total_tokens += 1
  tokens_over_time.append(total_tokens)
  times.append(time.time() - start_time)

  while not is_finished and total_tokens < num_tokens:
    resp, inference_state, is_finished = await inference_engine.infer_tensor(req_id, shard, resp, inference_state)
    total_tokens += 1
    tokens_over_time.append(total_tokens)
    times.append(time.time() - start_time)
    if verbose: print(tokenizer.decode(resp), end='', flush=True)

  return tokens_over_time, times

async def main():
  shard_downloader = HFShardDownloader()
  def on_progress(shard: Shard, event: RepoProgressEvent):
    print(f"Downloading shard {shard} {pretty_print_bytes_per_second(event.overall_speed)} | {event.overall_eta}")
  shard_downloader.on_progress.register("print").on_next(on_progress)

  engines = [
    # (TinygradDynamicShardInferenceEngine(shard_downloader), "Tinygrad", "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"),
    # (TorchDynamicShardInferenceEngine(shard_downloader), "Torch", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    # (MLXDynamicShardInferenceEngine(shard_downloader), "MLX", "mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated"),
    (MLXDynamicShardInferenceEngine(shard_downloader), "MLX", "mlx-community/gemma-2-9b-it-4bit")
  ]

  plt.figure(figsize=(12, 6))
  summary = {}

  for engine, name, model_id in engines:
    shard = Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32)
    tokenizer = await resolve_tokenizer(model_id)
    await run_bench(engine, tokenizer, shard, 10)
    tokens, times = await run_bench(engine, tokenizer, shard)

    plt.plot(times, tokens, label=name)

    first_token_time = times[0]

    # Calculate sustained TPS using the latter half of the data
    mid_point = len(tokens) // 2
    sustained_tps = (tokens[-1] - tokens[mid_point]) / (times[-1] - times[mid_point])

    peak_tps = max([tokens[i] / times[i] for i in range(1, len(tokens))])

    summary[name] = {
      "first_token_time": first_token_time,
      "sustained_tps": sustained_tps,
      "peak_tps": peak_tps
    }

  plt.xlabel("Time (seconds)")
  plt.ylabel("Tokens Generated")
  plt.title("Token Generation Over Time")
  plt.legend()
  plt.grid(True)
  plt.savefig("token_generation_comparison.png")
  plt.close()

  print("\nPerformance Summary:")
  for name, metrics in summary.items():
    print(f"\n{name}:")
    print(f"  Time to First Token: {metrics['first_token_time']:.4f} seconds")
    print(f"  Sustained TPS: {metrics['sustained_tps']:.2f} tokens/second")
    print(f"  Peak TPS: {metrics['peak_tps']:.2f} tokens/second")

  fastest_first_token = min(summary.items(), key=lambda x: x[1]['first_token_time'])
  fastest_sustained = max(summary.items(), key=lambda x: x[1]['sustained_tps'])
  fastest_peak = max(summary.items(), key=lambda x: x[1]['peak_tps'])

  print("\nFastest Engines:")
  print(f"Fastest to First Token: {fastest_first_token[0]} ({fastest_first_token[1]['first_token_time']:.4f} seconds)")
  print(f"Fastest Sustained TPS: {fastest_sustained[0]} ({fastest_sustained[1]['sustained_tps']:.2f} tokens/second)")
  print(f"Fastest Peak TPS: {fastest_peak[0]} ({fastest_peak[1]['peak_tps']:.2f} tokens/second)")

if __name__ == "__main__":
  asyncio.run(main())
