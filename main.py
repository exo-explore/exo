import argparse
import asyncio
import signal
import mlx.core as mx
import mlx.nn as nn
from orchestration.standard_node import StandardNode
from networking.grpc.grpc_server import GRPCServer
from inference.inference_engine import MLXFixedShardInferenceEngine
from inference.shard import Shard
from networking.grpc.grpc_discovery import GRPCDiscovery

class SimpleMLXModel(nn.Module):
    def __init__(self):
        super(SimpleMLXModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Example dimensions

    def forward(self, x):
        return self.linear(x)


# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("--node-id", type=str, default="node1", help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=8080, help="Node port")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
args = parser.parse_args()

mlx_model = SimpleMLXModel()
inference_engine = MLXFixedShardInferenceEngine(mlx_model, shard=Shard(model_id="test", n_layers=32, start_layer=0, end_layer=31))
discovery = GRPCDiscovery(args.node_id, args.node_port, args.listen_port, args.broadcast_port)
node = StandardNode(args.node_id, None, inference_engine, discovery)
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

    await node.start()

    await asyncio.sleep(5)
    print("Sending reset shard request")
    await node.peers[0].reset_shard(f"regards from {node.id}")

    await asyncio.Event().wait()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
