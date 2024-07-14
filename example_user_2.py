# In this example, a user is running a home cluster with 3 shards.
# They are prompting the cluster to generate a response to a question.
# The cluster is given the question, and the user is given the response.

from inference.mlx.sharded_utils import get_model_path, load_tokenizer
from inference.shard import Shard
from networking.peer_handle import PeerHandle
from networking.grpc.grpc_peer_handle import GRPCPeerHandle
from topology.device_capabilities import DeviceCapabilities
from typing import List
import asyncio
import argparse

path_or_hf_repo = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model_path = get_model_path(path_or_hf_repo)
tokenizer_config = {}
tokenizer = load_tokenizer(model_path, tokenizer_config)

peer1 = GRPCPeerHandle(
    "node1",
    "localhost:8080",
    DeviceCapabilities(model="test1", chip="test1", memory=10000)
)
peer2 = GRPCPeerHandle(
    "node2",
    "localhost:8081",
    DeviceCapabilities(model="test1", chip="test1", memory=10000)
)
shard = Shard(model_id=path_or_hf_repo, start_layer=0, end_layer=0, n_layers=32)

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

    for peer in [peer1, peer2]:
        await peer.connect()
        await peer.reset_shard(shard)

    try:
        await peer1.send_prompt(shard, prompt, "request-id-1")
    except Exception as e:
        print(e)

    import sys
    # poll 10 times per second for result (even though generation is faster, any more than this it's not nice for the user)
    previous_length = 0
    while True:
        result, is_finished = await peer2.get_inference_result("request-id-1")
        await asyncio.sleep(0.1)

        # Print the updated string in place
        updated_string = tokenizer.decode(result)
        print(updated_string[previous_length:], end='', flush=True)
        previous_length = len(updated_string)

        if is_finished:
            print("\nDone")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt")
    parser.add_argument("--prompt", type=str, help="The prompt to run")
    args = parser.parse_args()

    asyncio.run(run_prompt(args.prompt))
