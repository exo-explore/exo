import argparse
import asyncio
import signal
import mlx.core as mx
import mlx.nn as nn
from orchestration.standard_node import StandardNode
from networking.grpc.grpc_server import GRPCServer
from inference.mlx.sharded_inference_engine import MLXFixedShardInferenceEngine
from inference.shard import Shard
from networking.grpc.grpc_discovery import GRPCDiscovery
from topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("--node-id", type=str, default="node1", help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=8080, help="Node port")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--model-id", type=str, default="mlx-community/Meta-Llama-3-8B-Instruct-4bit", help="Path to the model")
parser.add_argument("--n-layers", type=int, default=32, help="Number of layers in the model")
parser.add_argument("--start-layer", type=int, default=0, help="Start layer index")
parser.add_argument("--end-layer", type=int, default=31, help="End layer index")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
args = parser.parse_args()

inference_engine = MLXFixedShardInferenceEngine(args.model_id, shard=Shard(model_id=args.model_id, n_layers=args.n_layers, start_layer=args.start_layer, end_layer=args.end_layer))
discovery = GRPCDiscovery(args.node_id, args.node_port, args.listen_port, args.broadcast_port)
node = StandardNode(args.node_id, None, inference_engine, discovery, partitioning_strategy=RingMemoryWeightedPartitioningStrategy())
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server


async def shutdown(signal, loop):
    """Gracefully shutdown the server and close the asyncio loop."""
    print(f"Received exit signal {signal.name}...")
    server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in server_tasks]
    print(f"Cancelling {len(server_tasks)} outstanding tasks")
    await asyncio.gather(*server_tasks, return_exceptions=True)
    await server.shutdown()
    loop.stop()

async def main():
    loop = asyncio.get_running_loop()

    # Use a more direct approach to handle signals
    def handle_exit():
        asyncio.ensure_future(shutdown(signal.SIGTERM, loop))

    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, handle_exit)

    await node.start(wait_for_peers=args.wait_for_peers)

    await asyncio.Event().wait()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
