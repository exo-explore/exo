import uuid
import time
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List
from aiohttp import web
from exo import DEBUG
from exo.inference.shard import Shard
from exo.orchestration import Node
from exo.inference.mlx.sharded_utils import get_model_path, load_tokenizer

shard_mappings = {
    "llama-3-8b": Shard(model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=32),
    "llama-3-70b": Shard(model_id="mlx-community/Meta-Llama-3-70B-Instruct-4bit", start_layer=0, end_layer=0, n_layers=80),
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

class ChatGPTAPI:
    def __init__(self, node: Node):
        self.node = node
        self.app = web.Application()
        self.app.router.add_post('/v1/chat/completions', self.handle_post)

    async def handle_post(self, request):
        data = await request.json()
        messages = [Message(**msg) for msg in data['messages']]
        chat_request = ChatCompletionRequest(data['model'], messages, data['temperature'])
        prompt = " ".join([msg.content for msg in chat_request.messages if msg.role == "user"])
        shard = shard_mappings.get(chat_request.model)
        if not shard:
            return web.json_response({'detail': f"Invalid model: {chat_request.model}. Supported: {list(shard_mappings.keys())}"}, status=400)
        request_id = str(uuid.uuid4())

        tokenizer = load_tokenizer(get_model_path(shard.model_id))
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
                if result[-1] == tokenizer._tokenizer.eos_token_id:
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
