# In this example, a user is running a home cluster with 3 shards.
# They are prompting the cluster to generate a response to a question.
# The cluster is given the question, and the user is given the response.

from exo.inference.mlx.sharded_utils import get_model_path, load_tokenizer
from exo.inference.shard import Shard
from exo.networking.peer_handle import PeerHandle
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.device_capabilities import DeviceCapabilities
from typing import List
import asyncio
import argparse

path_or_hf_repo = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model_path = get_model_path(path_or_hf_repo)
tokenizer_config = {}
tokenizer = load_tokenizer(model_path, tokenizer_config)

peers: List[PeerHandle] = [
    GRPCPeerHandle(
        "node1",
        "localhost:8080",
        DeviceCapabilities(model="test1", chip="test1", memory=10000)
    ),
    GRPCPeerHandle(
        "node2",
        "localhost:8081",
        DeviceCapabilities(model="test2", chip="test2", memory=20000)
    )
]
shards: List[Shard] = [
    Shard(model_id=path_or_hf_repo, start_layer=0, end_layer=15, n_layers=32),
    Shard(model_id=path_or_hf_repo, start_layer=16, end_layer=31, n_layers=32),
    # Shard(model_id=path_or_hf_repo, start_layer=0, end_layer=30, n_layers=32),
    # Shard(model_id=path_or_hf_repo, start_layer=31, end_layer=31, n_layers=32),
]

async def run_prompt(prompt: str):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = tokenizer.default_chat_template
    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    for peer, shard in zip(peers, shards):
        await peer.connect()
        await peer.reset_shard(shard)

    tokens = []
    last_output = prompt

    for _ in range(20):
        for peer, shard in zip(peers, shards):
            if isinstance(last_output, str):
                last_output = await peer.send_prompt(shard, last_output)
                print("prompt output:", last_output)
            else:
                last_output = await peer.send_tensor(shard, last_output)
                print("tensor output:", last_output)

        if not last_output:
            break

        tokens.append(last_output.item())

    print(tokenizer.decode(tokens))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt")
    parser.add_argument("--prompt", type=str, help="The prompt to run")
    args = parser.parse_args()

    asyncio.run(run_prompt(args.prompt))
