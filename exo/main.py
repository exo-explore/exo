import argparse
import asyncio
import atexit
import signal
import json
import logging
import platform
import os
import sys
import time
import traceback
import uuid
import numpy as np
from functools import partial
from tqdm import tqdm
from exo.train.dataset import load_dataset, iterate_batches, compose
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.manual.network_topology_config import NetworkTopology
from exo.orchestration.node import Node
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.shard_download import ShardDownloader, RepoProgressEvent, NoopShardDownloader
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import build_base_shard, get_repo
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import has_hf_home_read_access, has_hf_home_write_access, get_hf_home, move_models_to_hf
import uvloop
from contextlib import asynccontextmanager
import concurrent.futures
import socket
import resource
import psutil
import grpc

# Configure uvloop for maximum performance
def configure_uvloop():
    # Install uvloop as event loop policy
    uvloop.install()

    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    if not psutil.WINDOWS:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try:
          resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try:
          resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError:
          pass

    # Configure thread pool for blocking operations
    loop.set_default_executor(
      concurrent.futures.ThreadPoolExecutor(
        max_workers=min(32, (os.cpu_count() or 1) * 4)
      )
    )

    return loop

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run", "eval", "train"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--default-model", type=str, default=None, help="Default model")
parser.add_argument("--iters", type=int, default=100, help="Training iterations")
parser.add_argument("--save-every", type=int, default=5, help="Save the model every N iterations.")
parser.add_argument("--data", type=str, default="exo/train/data/lora", help="Directory where training data lives")
parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a custom checkpoint to load")
parser.add_argument("--save-checkpoint-dir", type=str, default="checkpoints", help="Path to a folder where checkpoints are stored")
parser.add_argument("--node-id", type=str, default=None, help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=4, help="Max parallel downloads for model shards download")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="Discovery module to use")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=90, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use (mlx, tinygrad, or dummy)")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
parser.add_argument("--default-temp", type=float, help="Default token sampling temperature", default=0.0)
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
parser.add_argument("--node-id-filter", type=str, default=None, help="Comma separated list of allowed node IDs (only for UDP and Tailscale discovery)")
parser.add_argument("--interface-type-filter", type=str, default=None, help="Comma separated list of allowed interface types (only for UDP discovery)")
args = parser.parse_args()
print(f"Selected inference engine: {args.inference_engine}")

print_yellow_exo()

system_info = get_system_info()
print(f"Detected system: {system_info}")

shard_downloader: ShardDownloader = HFShardDownloader(quick_check=args.download_quick_check,
                                                      max_parallel_downloads=args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
print(f"Inference engine name after selection: {inference_engine_name}")

inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

# Convert node-id-filter and interface-type-filter to lists if provided
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None
allowed_interface_types = args.interface_type_filter.split(',') if args.interface_type_filter else None

if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids,
    allowed_interface_types=allowed_interface_types
  )
elif args.discovery_module == "tailscale":
  discovery = TailscaleDiscovery(
    args.node_id,
    args.node_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    tailscale_api_key=args.tailscale_api_key,
    tailnet=args.tailnet_name,
    allowed_node_ids=allowed_node_ids
  )
elif args.discovery_module == "manual":
  if not args.discovery_config_path:
    raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None
node = Node(
  args.node_id,
  None,
  inference_engine,
  discovery,
  partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
  max_generate_tokens=args.max_generate_tokens,
  topology_viz=topology_viz,
  shard_downloader=shard_downloader,
  default_sample_temperature=args.default_temp
)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
api = ChatGPTAPI(
  node,
  inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model
)
# node.on_token.register("update_topology_viz").on_next(
#   lambda req_id, token, __: topology_viz.update_prompt_output(req_id, inference_engine.tokenizer.decode([token])) if topology_viz and hasattr(inference_engine, "tokenizer") else None
# )

def preemptively_start_download(request_id: str, opaque_status: str):
  try:
    status = json.loads(opaque_status)
    if status.get("type") == "node_status" and status.get("status") == "start_process_prompt":
      current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
      if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
      asyncio.create_task(shard_downloader.ensure_shard(current_shard, inference_engine.__class__.__name__))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to preemptively start download: {e}")
      traceback.print_exc()


node.on_opaque_status.register("start_download").on_next(preemptively_start_download)

last_broadcast_time = 0


def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_broadcast_time
  current_time = time.time()
  if event.status == "complete" or current_time - last_broadcast_time >= 0.1:
    last_broadcast_time = current_time
    asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))


shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

async def run_model_cli(node: Node, inference_engine: InferenceEngine, model_name: str, prompt: str):
  inference_class = inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_engine.__class__.__name__}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, request_id=request_id)

    first_token_time = time.time()
    tokens = []
    i = 0
    def on_token(_request_id, _token, _is_finished):
      nonlocal i
      i += 1
      if i % 20 == 0:
        print(f"TPS: {i / (time.time() - first_token_time)}")

      tokens.append(_token)
      return _request_id == request_id and _is_finished
    await callback.wait(on_token, timeout=300)

    print("=== Stats ===")
    print(f"Total time: {time.time() - first_token_time}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Total tokens per second: {len(tokens) / (time.time() - first_token_time)}")

    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    node.on_token.deregister(callback_id)

def clean_path(path):
    """Clean and resolve path"""
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')
    return os.path.expanduser(path)

async def hold_outstanding(node: Node):
  while node.outstanding_requests:
    await asyncio.sleep(.5)
  return

async def run_iter(node: Node, shard: Shard, train: bool, data, batch_size=1):
  losses = []
  tokens = []
  for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
    _, _, lengths = batch
    losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
    tokens.append(np.sum(lengths))
  total_tokens = np.sum(tokens)
  total_loss = np.sum(losses) / total_tokens

  return total_loss, total_tokens

async def eval_model_cli(node: Node, inference_engine: InferenceEngine, model_name, dataloader, batch_size, num_batches=-1):
  inference_class = inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_engine.__class__.__name__}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
  loss, tokens = await run_iter(node, shard, False, test, batch_size)
  print(f"total | {loss=}, {tokens=}")
  print("Waiting for outstanding tasks")
  await hold_outstanding(node)

async def train_model_cli(node: Node, inference_engine: InferenceEngine, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
  inference_class = inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_engine.__class__.__name__}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
  for i in tqdm(range(3)):
    await asyncio.sleep(1)
  for epoch in range(iters):
    loss, tokens = await run_iter(node, shard, True, train, batch_size)
    print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
    if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
      await node.coordinate_save(shard, epoch, checkpoint_dir)
      await hold_outstanding(node)
  await hold_outstanding(node)


async def main():
  loop = asyncio.get_running_loop()

  # Set up OpenTelemetry
  from opentelemetry import trace
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
  from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
  from opentelemetry.sdk.resources import Resource
  
  # Check if Jaeger is available
  def check_jaeger_connection():
    try:
      # Try to connect to the OTLP gRPC port
      sock = socket.create_connection(("localhost", 4317), timeout=1)
      sock.close()
      return True
    except (socket.timeout, socket.error):
      return False
  
  # Create and configure the tracer
  resource = Resource.create({
    "service.name": "exo-distributed",
    "service.instance.id": args.node_id
  })
  
  tracer_provider = TracerProvider(resource=resource)
  
  if check_jaeger_connection():
    print("Jaeger connection successful, setting up tracing...")
    # Configure the OTLP exporter with better defaults for high throughput
    otlp_exporter = OTLPSpanExporter(
      endpoint="http://localhost:4317",
      # Increase timeout to handle larger batches
      timeout=30.0,
    )
    
    # Configure the BatchSpanProcessor with appropriate batch settings
    span_processor = BatchSpanProcessor(
      otlp_exporter,
      # Reduce export frequency
      schedule_delay_millis=5000,
      # Increase max batch size
      max_export_batch_size=512,
      # Limit queue size to prevent memory issues
      max_queue_size=2048,
    )
    
    tracer_provider.add_span_processor(span_processor)
  else:
    print("Warning: Could not connect to Jaeger, tracing will be disabled")
    # Use a no-op tracer provider instead
    tracer_provider = trace.NoOpTracerProvider()
  
  # Set the tracer provider
  trace.set_tracer_provider(tracer_provider)

  # Check HuggingFace directory permissions
  hf_home, has_read, has_write = get_hf_home(), await has_hf_home_read_access(), await has_hf_home_write_access()
  if DEBUG >= 1: print(f"Model storage directory: {hf_home}")
  print(f"{has_read=}, {has_write=}")
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for model storage directory: {hf_home}.
          This may prevent model downloads from working correctly.
          {"❌ No read access" if not has_read else ""}
          {"❌ No write access" if not has_write else ""}
          """)

  if not args.models_seed_dir is None:
    try:
      models_seed_dir = clean_path(args.models_seed_dir)
      await move_models_to_hf(models_seed_dir)
    except Exception as e:
      print(f"Error moving models to .cache/huggingface: {e}")

  def restore_cursor():
    if platform.system() != "Windows":
        os.system("tput cnorm")  # Show cursor

  # Restore the cursor when the program exits
  atexit.register(restore_cursor)

  # Use a more direct approach to handle signals
  def handle_exit():
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))

  if platform.system() != "Windows":
    for s in [signal.SIGINT, signal.SIGTERM]:
      loop.add_signal_handler(s, handle_exit)

  await node.start(wait_for_peers=args.wait_for_peers)

  if args.command == "run" or args.run_model:
    model_name = args.model_name or args.run_model
    if not model_name:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    await run_model_cli(node, inference_engine, model_name, args.prompt)
  elif args.command == "eval" or args.command == 'train':
    model_name = args.model_name
    dataloader = lambda tok: load_dataset(args.data, preprocess=lambda item: tok(item)
                                                   , loadline=lambda line: json.loads(line).get("text",""))
    if args.command == 'eval':
      if not model_name:
        print("Error: Much like a human, I can't evaluate anything without a model")
        return
      await eval_model_cli(node, inference_engine, model_name, dataloader, args.batch_size)
    else:
      if not model_name:
        print("Error: This train ain't leaving the station without a model")
        return
      await train_model_cli(node, inference_engine, model_name, dataloader, args.batch_size, args.iters, save_interval=args.save_every, checkpoint_dir=args.save_checkpoint_dir)

  else:
    asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
    await asyncio.Event().wait()

  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)

@asynccontextmanager
async def setup_node(args):
    # Rest of setup_node implementation...
    pass

def run():
    loop = None
    try:
        loop = configure_uvloop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    finally:
        if loop:
            loop.close()

if __name__ == "__main__":
  run()
