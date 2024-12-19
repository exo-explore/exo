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
import json
from pathlib import Path
import numpy as np
from functools import partial
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
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
from exo.models import model_cards, build_local_model_card, build_base_shard, get_repo
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import has_hf_home_read_access, has_hf_home_write_access, get_hf_home, move_models_to_hf

from exo.localmodel.http.http_server import LocalModelAPI 
from exo.localmodel.lh_helpers import has_exo_home_read_access, has_exo_home_write_access, get_exo_home, downlaod_tokenizer_config
from exo.helpers import check_agent, get_stored_models, get_stored_model_config

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
parser.add_argument("--prometheus-client-port", type=int, default=None, help="Prometheus client port")
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
parser.add_argument("--stored-model-ip", type=str, default="0.0.0.0", help="Stored Local Model IP")
parser.add_argument("--stored-model-port", type=int, default=52525, help="Local Model API port")
args = parser.parse_args()
print(f"Selected inference engine: {args.inference_engine}")

print_yellow_exo()

system_info = get_system_info()
print(f"Detected system: {system_info}")

node_agent = check_agent(args.stored_model_ip)
print(f"Node Agent: {node_agent}")

inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
print(f"Inference engine name after selection: {inference_engine_name}")

shard_downloader: ShardDownloader = HFShardDownloader(quick_check=args.download_quick_check,
                                                      max_parallel_downloads=args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()

inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

# (Host) Local Model Card
print(f"Add Local Model To Model Card：")
exo_home = get_exo_home()
for folder_path in Path(exo_home).iterdir():
  if folder_path.is_dir():
    model_name = folder_path.name
    config_file_path = Path(folder_path/'config.json')
    if config_file_path.is_file():
      with open(config_file_path, 'r') as file:
        config = json.load(file)
        config_n_layers = config['num_hidden_layers']
        build_local_model_card(model_name, None, inference_engine.__class__.__name__, int(config_n_layers))
        print(f"- Local/{model_name}")

if args.node_port is None:
  args.node_port = find_available_port(args.node_host) 
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()


if args.node_host == "0.0.0.0":
  chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
  web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
else:
  chatgpt_api_endpoints = [f"http://{args.node_host}:{args.chatgpt_api_port}/v1/chat/completions"]
  web_chat_urls = [f"http://{args.node_host}:{args.chatgpt_api_port}"]

if args.stored_model_ip == "0.0.0.0":
  local_model_api_endpoints = [f"http://{ip}:{args.stored_model_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
else:
  local_model_api_endpoints = [f"http://{args.stored_model_ip}:{args.stored_model_port}"]

if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

  print("Local Model API endpoint served at:")
  for local_model_api_endpoint in local_model_api_endpoints:
    print(f" - {terminal_link(local_model_api_endpoint)}")

# Convert node-id-filter to list if provided
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None

if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids
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
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls, local_model_api_endpoints=local_model_api_endpoints) if not args.disable_tui else None
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
chatgpt_api = ChatGPTAPI(
  node,
  inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model
)
local_model_api = LocalModelAPI()
node.on_token.register("update_topology_viz").on_next(
  lambda req_id, tokens, __: topology_viz.update_prompt_output(req_id, inference_engine.tokenizer.decode(tokens)) if topology_viz and hasattr(inference_engine, "tokenizer") else None
)

def preemptively_start_download(request_id: str, opaque_status: str):
  try:
    status = json.loads(opaque_status)
    if status.get("type") == "node_status" and status.get("status") == "start_process_prompt":
      current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
      if os.path.isdir(current_shard.model_id):
        if DEBUG >= 2: print("Local model detected, skip download")
        pass
      elif "Local" in current_shard.model_id:
        if DEBUG >= 2: print("Local model detected, http download to other node")
        # borcast download model
        if node_agent == "client":
          asyncio.create_task(shard_downloader.ensure_shard(current_shard, inference_engine.__class__.__name__))
      else:
        if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
        asyncio.create_task(shard_downloader.ensure_shard(current_shard, inference_engine.__class__.__name__))
    
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to preemptively start download: {e}")
      traceback.print_exc()

node.on_opaque_status.register("start_download").on_next(preemptively_start_download)

if args.prometheus_client_port:
  from exo.stats.metrics import start_metrics_server
  start_metrics_server(node, args.prometheus_client_port)

last_broadcast_time = 0


def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_broadcast_time
  current_time = time.time()
  if event.status == "complete" or current_time - last_broadcast_time >= 0.1:
    last_broadcast_time = current_time
    asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))

shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

async def run_model_cli(node: Node, inference_engine: InferenceEngine, model_name_or_path: str, prompt: str):
  inference_class = inference_engine.__class__.__name__
  if model_cards.get(model_name_or_path):
    shard = build_base_shard(model_name_or_path, inference_class)
    if not shard:
      print(f"Error: exo Unsupported model '{model_name_or_path}' for inference engine {inference_engine.__class__.__name__}")
      return
    if "Local" in model_name_or_path:
      await downlaod_tokenizer_config(model_name_or_path)
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  elif os.path.isdir(model_name_or_path):
    model_path = model_name_or_path.rstrip('/')
    model_name = str(model_path.split('/')[-1])
    model_config_path = Path(model_path)/'config.json'

    if os.path.isfile(model_config_path):
      with open(model_config_path, 'r') as file:
        config = json.load(file)
        config_n_layers = config['num_hidden_layers']
      build_local_model_card(model_name, model_path, inference_class, config_n_layers)
      shard = Shard(model_id=model_path, start_layer=0, end_layer=0, n_layers=config_n_layers)
      tokenizer = await resolve_tokenizer(model_path) # transformer tokenizer
    else:
      print(f"Error: Not Find Local model '{model_name}' for inference engine {inference_class}")
      return
  else:
    print(f"Error: Unsupported model '{model_name_or_path}' for inference engine {inference_engine.__class__.__name__}")
    return
  
  print(f'model_card = {get_repo(model_name_or_path, inference_class)}')
  print(f'model_id = {get_repo(shard.model_id, inference_class)}')
  
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
  
  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, request_id=request_id) # download model entry
    
    _, tokens, _ = await callback.wait(lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished, timeout=300)

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

  # Check HuggingFace directory permissions
  hf_home, has_read, has_write = get_hf_home(), await has_hf_home_read_access(), await has_hf_home_write_access()
  if DEBUG >= 1: print(f"HF Model storage directory: {hf_home}")
  print(f"{has_read=}, {has_write=}")
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for model storage directory: {hf_home}.
          This may prevent model downloads from working correctly.
          {"❌ No read access" if not has_read else ""}
          {"❌ No write access" if not has_write else ""}
          """)
  # Check exo directory permissions
  exo_home, has_read, has_write = get_exo_home(), await has_exo_home_read_access(), await has_exo_home_write_access()
  if DEBUG >= 1: print(f"Local Model storage directory: {exo_home}")
  print(f"{has_read=}, {has_write=}")
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for model storage directory: {exo_home}.
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

  if node_agent == "stored": # Local model server
    asyncio.create_task(local_model_api.run(host=args.stored_model_ip, port=args.stored_model_port))
  elif node_agent == "client": # get local models from master
    stored_models_data = await get_stored_models(args.stored_model_ip, args.stored_model_port)
    if stored_models_data != []:
      for model in stored_models_data.get("models"):
        model_name = model.get('name')
        model_config_data = await get_stored_model_config(args.stored_model_ip, args.stored_model_port, model_name)
        if model_config_data is not None:
          model_layer = model_config_data.get('num_hidden_layers')
          build_local_model_card(model_name, None, inference_engine.__class__.__name__, int(model_layer))

  await node.start(wait_for_peers=args.wait_for_peers)

  if args.command == "run" or args.run_model:
    model_name_or_path = args.model_name or args.run_model
    if not model_name_or_path:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    await run_model_cli(node, inference_engine, model_name_or_path, args.prompt)
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
    asyncio.create_task(chatgpt_api.run(host=args.node_host, port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
    await asyncio.Event().wait()
  
  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)


def run():
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    loop.run_until_complete(main())
      
  except KeyboardInterrupt:
    print("Received keyboard interrupt. Shutting down...")
  finally:
    loop.run_until_complete(shutdown(signal.SIGTERM, loop, node.server))
    loop.close()


if __name__ == "__main__":
  run()
