import argparse
import asyncio
import signal
import uuid
from typing import List
from exo.orchestration.standard_node import StandardNode
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.grpc.grpc_discovery import GRPCDiscovery
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_inference_engine, get_system_info

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("--node-id", type=str, default=str(uuid.uuid4()), help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--prometheus-client-port", type=int, default=None, help="Prometheus client port")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=8000, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout-secs", type=int, default=90, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=256, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
args = parser.parse_args()

print_yellow_exo()

system_info = get_system_info()
print(f"Detected system: {system_info}")

inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
inference_engine = get_inference_engine(inference_engine_name)
print(f"Using inference engine: {inference_engine.__class__.__name__}")

if args.node_port is None:
    args.node_port = find_available_port(args.node_host)
    if DEBUG >= 1: print(f"Using available port: {args.node_port}")

discovery = GRPCDiscovery(args.node_id, args.node_port, args.listen_port, args.broadcast_port)
node = StandardNode(args.node_id, None, inference_engine, discovery, partitioning_strategy=RingMemoryWeightedPartitioningStrategy(), chatgpt_api_endpoint=f"http://localhost:{args.chatgpt_api_port}/v1/chat/completions", web_chat_url=f"http://localhost:{args.chatgpt_api_port}", disable_tui=args.disable_tui, max_generate_tokens=args.max_generate_tokens)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
api = ChatGPTAPI(node, inference_engine.__class__.__name__, response_timeout_secs=args.chatgpt_api_response_timeout_secs)
node.on_token.register("main_log").on_next(lambda _, tokens , __: print(inference_engine.tokenizer.decode(tokens) if hasattr(inference_engine, "tokenizer") else tokens))
if args.prometheus_client_port:
    from exo.stats.metrics import start_metrics_server
    start_metrics_server(node, args.prometheus_client_port)

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

async def main():
    loop = asyncio.get_running_loop()

    # Use a more direct approach to handle signals
    def handle_exit():
        asyncio.ensure_future(shutdown(signal.SIGTERM, loop))

    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, handle_exit)

    await node.start(wait_for_peers=args.wait_for_peers)
    asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task

    await asyncio.Event().wait()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Shutting down...")
    finally:
        loop.run_until_complete(shutdown(signal.SIGTERM, loop))
        loop.close()
