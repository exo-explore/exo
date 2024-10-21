import argparse
import asyncio
import signal
import json
import time
import traceback
import uuid
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.manual.network_topology_config import NetworkTopology
from exo.orchestration.standard_node import StandardNode
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.shard_download import ShardDownloader, RepoProgressEvent
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses, terminal_link
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration.node import Node
from exo.models import model_base_shards
from exo.viz.topology_viz import TopologyViz

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--node-id", type=str, default=None, help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=4, help="Max parallel downloads for model shards download")
parser.add_argument("--prometheus-client-port", type=int, default=None, help="Prometheus client port")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="Discovery module to use")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=8000, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=90, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
args = parser.parse_args()

print_yellow_exo()

system_info = get_system_info()
print(f"Detected system: {system_info}")

shard_downloader: ShardDownloader = HFShardDownloader(quick_check=args.download_quick_check, max_parallel_downloads=args.max_parallel_downloads)
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip in get_all_ip_addresses()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip in get_all_ip_addresses()]
if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

if args.discovery_module == "udp":
  discovery = UDPDiscovery(args.node_id, args.node_port, args.listen_port, args.broadcast_port, lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities), discovery_timeout=args.discovery_timeout)
elif args.discovery_module == "tailscale":
  discovery = TailscaleDiscovery(args.node_id, args.node_port, lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities), discovery_timeout=args.discovery_timeout, tailscale_api_key=args.tailscale_api_key, tailnet=args.tailnet_name)
elif args.discovery_module == "manual":
  if not args.discovery_config_path:
    raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities), discovery_timeout=args.discovery_timeout)
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None
node = StandardNode(
  args.node_id,
  None,
  inference_engine,
  discovery,
  partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
  max_generate_tokens=args.max_generate_tokens,
  topology_viz=topology_viz
)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
api = ChatGPTAPI(
  node,
  inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None
)
node.on_token.register("update_topology_viz").on_next(
  lambda req_id, tokens, __: topology_viz.update_prompt_output(req_id, inference_engine.tokenizer.decode(tokens)) if topology_viz and hasattr(inference_engine, "tokenizer") else None
)
def preemptively_start_download(request_id: str, opaque_status: str):
  try:
    status = json.loads(opaque_status)
    if status.get("type") == "node_status" and status.get("status") == "start_process_prompt":
      current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
      if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
      asyncio.create_task(shard_downloader.ensure_shard(current_shard))
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
        asyncio.create_task(node.broadcast_opaque_status("", json.dumps({
            "type": "download_progress",
            "node_id": node.id,
            "progress": event.to_dict()
        })))

shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)


async def shutdown(signal, loop):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using exo.")
  print_yellow_exo()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()
  loop.stop()


async def run_model_cli(node: Node, inference_engine: InferenceEngine, model_name: str, prompt: str):
  shard = model_base_shards.get(model_name, {}).get(inference_engine.__class__.__name__)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_engine.__class__.__name__}")
    return
  tokenizer = await resolve_tokenizer(shard.model_id)
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, None, request_id=request_id)

    _, tokens, _ = await callback.wait(lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished, timeout=300)

    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    node.on_token.deregister(callback_id)


async def main():
  loop = asyncio.get_running_loop()

  # Use a more direct approach to handle signals
  def handle_exit():
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop))

  for s in [signal.SIGINT, signal.SIGTERM]:
    loop.add_signal_handler(s, handle_exit)

  await node.start(wait_for_peers=args.wait_for_peers)

  if args.command == "run" or args.run_model:
    model_name = args.model_name or args.run_model
    if not model_name:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    await run_model_cli(node, inference_engine, model_name, args.prompt)
  else:
    asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
    await asyncio.Event().wait()


def run():
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    loop.run_until_complete(main())
  except KeyboardInterrupt:
    print("Received keyboard interrupt. Shutting down...")
  finally:
    loop.run_until_complete(shutdown(signal.SIGTERM, loop))
    loop.close()

if __name__ == "__main__":
  run()
