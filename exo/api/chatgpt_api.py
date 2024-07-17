import uuid
import time
import asyncio
from transformers import AutoTokenizer
from typing import List
from aiohttp import web
from exo import DEBUG
from exo.inference.shard import Shard
from exo.orchestration import Node

shard_mappings = {
    "llama-3-8b": {
        "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=32),
        "TinygradDynamicShardInferenceEngine": Shard(model_id="llama3-8b-sfr", start_layer=0, end_layer=0, n_layers=32),
    },
    "llama-3-70b": {
        "MLXDynamicShardInferenceEngine": Shard(model_id="mlx-community/Meta-Llama-3-70B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),
        "TinygradDynamicShardInferenceEngine": Shard(model_id="llama3-70b-sfr", start_layer=0, end_layer=0, n_layers=80),
    },
}

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class ChatCompletionRequest:
    def __init__(self, model: str, messages: List[Message], temperature: float):
        self.model = model
        self.messages = messages
        self.temperature = temperature

def resolve_tinygrad_tokenizer(model_id: str):
    if model_id == "llama3-8b-sfr":
        return AutoTokenizer.from_pretrained("TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R")
    elif model_id == "llama3-70b-sfr":
        return AutoTokenizer.from_pretrained("TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R")
    else:
        raise ValueError(f"tinygrad doesnt currently support arbitrary model downloading. unsupported model: {model_id}")

def resolve_tokenizer(model_id: str):
    try:
        if DEBUG >= 2: print(f"Trying AutoTokenizer for {model_id}")
        return AutoTokenizer.from_pretrained(model_id)
    except:
        import traceback
        if DEBUG >= 2: print(traceback.format_exc())
        if DEBUG >= 2: print(f"Failed to load tokenizer for {model_id}. Falling back to tinygrad tokenizer")

    try:
        if DEBUG >= 2: print(f"Trying tinygrad tokenizer for {model_id}")
        return resolve_tinygrad_tokenizer(model_id)
    except:
        import traceback
        if DEBUG >= 2: print(traceback.format_exc())
        if DEBUG >= 2: print(f"Failed again to load tokenizer for {model_id}. Falling back to mlx tokenizer")

    if DEBUG >= 2: print(f"Trying mlx tokenizer for {model_id}")
    from exo.inference.mlx.sharded_utils import get_model_path, load_tokenizer
    return load_tokenizer(get_model_path(model_id))

class ChatGPTAPI:
    def __init__(self, node: Node, inference_engine_classname: str):
        self.node = node
        self.app = web.Application()
        self.app.router.add_post('/v1/chat/completions', self.handle_post)
        self.inference_engine_classname = inference_engine_classname

    async def handle_post(self, request):
        data = await request.json()
        messages = [Message(**msg) for msg in data['messages']]
        chat_request = ChatCompletionRequest(data['model'], messages, data['temperature'])
        prompt = " ".join([msg.content for msg in chat_request.messages if msg.role == "user"])
        shard = shard_mappings.get(chat_request.model, {}).get(self.inference_engine_classname)
        if not shard:
            return web.json_response({'detail': f"Invalid model: {chat_request.model}. Supported: {list(shard_mappings.keys())}"}, status=400)
        request_id = str(uuid.uuid4())

        tokenizer = resolve_tokenizer(shard.model_id)
        if DEBUG >= 4: print(f"Resolved tokenizer: {tokenizer}")
        prompt = tokenizer.apply_chat_template(
            chat_request.messages, tokenize=False, add_generation_prompt=True
        )

        if DEBUG >= 2: print(f"Sending prompt from ChatGPT api {request_id=} {shard=} {prompt=}")
        try:
            result = await self.node.process_prompt(shard, prompt, request_id=request_id)
        except Exception as e:
            pass # TODO
            # return web.json_response({'detail': str(e)}, status=500)

        # poll for the response. TODO: implement callback for specific request id
        timeout = 90
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result, is_finished = await self.node.get_inference_result(request_id)
            except Exception as e:
                continue
            await asyncio.sleep(0.1)
            if is_finished:
                eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if isinstance(tokenizer._tokenizer, AutoTokenizer) else tokenizer.eos_token_id
                if DEBUG >= 2: print(f"Checking if end of result {result[-1]=} is {eos_token_id=}")
                if result[-1] == eos_token_id:
                    result = result[:-1]
                return web.json_response({
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": chat_request.model,
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(prompt)),
                        "completion_tokens": len(result),
                        "total_tokens": len(tokenizer.encode(prompt)) + len(result)
                    },
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": tokenizer.decode(result)
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                            "index": 0
                        }
                    ]
                })

        return web.json_response({'detail': "Response generation timed out"}, status=408)

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        if DEBUG >= 1: print(f"Starting ChatGPT API server at {host}:{port}")

# Usage example
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    node = Node()  # Assuming Node is properly defined elsewhere
    api = ChatGPTAPI(node)
    loop.run_until_complete(api.run())
