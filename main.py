import argparse
import asyncio
import signal
import uuid
import platform
from typing import List
from exo.orchestration.standard_node import StandardNode
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.grpc.grpc_discovery import GRPCDiscovery
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("--node-id", type=str, default=str(uuid.uuid4()), help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=8080, help="Node port")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=8000, help="ChatGPT API port")
args = parser.parse_args()

print(f"Starting {platform.system()=}")
if platform.system() == "Darwin":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
    inference_engine = MLXDynamicShardInferenceEngine()
else:
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    inference_engine = TinygradDynamicShardInferenceEngine()

def on_token(tokens: List[int]):
    if inference_engine.tokenizer:
        print(inference_engine.tokenizer.decode(tokens))
discovery = GRPCDiscovery(args.node_id, args.node_port, args.listen_port, args.broadcast_port)
node = StandardNode(args.node_id, None, inference_engine, discovery, partitioning_strategy=RingMemoryWeightedPartitioningStrategy(), on_token=on_token)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server

api = ChatGPTAPI(node)

async def shutdown(signal, loop):
    """Gracefully shutdown the server and close the asyncio loop."""
    print(f"Received exit signal {signal.name}...")
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
