import uuid
import time
import asyncio
import json
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Literal, Union
from aiohttp import web
import aiohttp_cors
from exo import DEBUG, VERSION
from exo.helpers import terminal_link
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

def generate_completion(
        chat_request: ChatCompletionRequest,
        tokenizer,
        prompt: str,
        request_id: str,
        tokens: List[int],
        stream: bool,
        finish_reason: Union[Literal["length", "stop"], None],
        object_type: Literal["chat.completion", "text_completion"]
    ) -> dict:
    completion = {
        "id": f"chatcmpl-{request_id}",
        "object": object_type,
        "created": int(time.time()),
        "model": chat_request.model,
        "system_fingerprint": f"exo_{VERSION}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tokenizer.decode(tokens)
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]
    }

    if not stream:
        completion["usage"] = {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokens),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokens)
        }

    choice = completion["choices"][0]
    if object_type.startswith("chat.completion"):
        key_name = "delta" if stream else "message"
        choice[key_name] = {"role": "assistant", "content": tokenizer.decode(tokens)}
    elif object_type == "text_completion":
        choice['text'] = tokenizer.decode(tokens)
    else:
        ValueError(f"Unsupported response type: {object_type}")

    return completion

def build_prompt(tokenizer, messages: List[Message]):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class ChatGPTAPI:
    def __init__(self, node: Node, inference_engine_classname: str):
        self.node = node
        self.inference_engine_classname = inference_engine_classname
        self.response_timeout_secs = 90
        self.app = web.Application()
        cors = aiohttp_cors.setup(self.app)
        cors_options = aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
        cors.add(self.app.router.add_post('/v1/chat/completions', self.handle_post_chat_completions), {
            "*": cors_options
        })
        cors.add(self.app.router.add_post('/v1/chat/token/encode', self.handle_post_chat_token_encode), {
            "*": cors_options
        })
        self.static_dir = Path(__file__).parent.parent.parent / 'tinychat/examples/tinychat'
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_static('/', self.static_dir, name='static')

    async def handle_root(self, request):
        return web.FileResponse(self.static_dir / 'index.html')

    async def handle_post_chat_token_encode(self, request):
        data = await request.json()
        shard = shard_mappings.get(data.get('model', 'llama-3-8b'), {}).get(self.inference_engine_classname)
        messages = data.get('messages', [])
        tokenizer = resolve_tokenizer(shard.model_id)
        return web.json_response({'length': len(build_prompt(tokenizer, messages))})

    async def handle_post_chat_completions(self, request):
        data = await request.json()
        stream = data.get('stream', False)
        messages = [Message(**msg) for msg in data['messages']]
        chat_request = ChatCompletionRequest(data.get('model', 'llama-3-8b'), messages, data.get('temperature', 0.0))
        if chat_request.model and chat_request.model.startswith("gpt-"): # to be compatible with ChatGPT tools, point all gpt- model requests to llama instead
            chat_request.model = "llama-3-8b"
        shard = shard_mappings.get(chat_request.model, {}).get(self.inference_engine_classname)
        if not shard:
            return web.json_response({'detail': f"Invalid model: {chat_request.model}. Supported: {list(shard_mappings.keys())}"}, status=400)
        request_id = str(uuid.uuid4())

        tokenizer = resolve_tokenizer(shard.model_id)
        if DEBUG >= 4: print(f"Resolved tokenizer: {tokenizer}")

        prompt = build_prompt(tokenizer, messages)
        callback_id = f"chatgpt-api-wait-response-{request_id}"
        callback = self.node.on_token.register(callback_id)

        if DEBUG >= 2: print(f"Sending prompt from ChatGPT api {request_id=} {shard=} {prompt=}")
        try:
            await self.node.process_prompt(shard, prompt, request_id=request_id)
        except Exception as e:
            if DEBUG >= 2:
                import traceback
                traceback.print_exc()
            return web.json_response({'detail': f"Error processing prompt (see logs): {str(e)}"}, status=500)

        try:
            if DEBUG >= 2: print(f"Waiting for response to finish. timeout={self.response_timeout_secs}s")

            if stream:
                response = web.StreamResponse(
                    status=200,
                    reason="OK",
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                        # "Access-Control-Allow-Origin": "*",
                        # "Access-Control-Allow-Methods": "*",
                        # "Access-Control-Allow-Headers": "*",
                    }
                )
                await response.prepare(request)

                stream_task = None
                last_tokens_len = 0
                async def stream_result(request_id: str, tokens: List[int], is_finished: bool):
                    nonlocal last_tokens_len
                    prev_last_tokens_len = last_tokens_len
                    last_tokens_len = len(tokens)
                    new_tokens = tokens[prev_last_tokens_len:]
                    finish_reason = None
                    eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if isinstance(tokenizer._tokenizer, AutoTokenizer) else tokenizer.eos_token_id
                    if len(new_tokens) > 0 and new_tokens[-1] == eos_token_id:
                        new_tokens = new_tokens[:-1]
                        if is_finished:
                            finish_reason = "stop"
                    if is_finished and not finish_reason:
                        finish_reason = "length"

                    completion = generate_completion(chat_request, tokenizer, prompt, request_id, new_tokens, stream, finish_reason, "chat.completion")
                    if DEBUG >= 2: print(f"Streaming completion: {completion}")
                    await response.write(f"data: {json.dumps(completion)}\n\n".encode())
                def on_result(_request_id: str, tokens: List[int], is_finished: bool):
                    nonlocal stream_task
                    stream_task = asyncio.create_task(stream_result(request_id, tokens, is_finished))

                    return _request_id == request_id and is_finished
                _, tokens, _ = await callback.wait(on_result, timeout=self.response_timeout_secs)
                if stream_task: # in case there is still a stream task running, wait for it to complete
                    if DEBUG >= 2: print(f"Pending stream task. Waiting for stream task to complete.")
                    try:
                        await asyncio.wait_for(stream_task, timeout=30)
                    except asyncio.TimeoutError:
                        print("WARNING: Stream task timed out. This should not happen.")
                await response.write_eof()
                return response
            else:
                _, tokens, _ = await callback.wait(lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished, timeout=self.response_timeout_secs)

                finish_reason = "length"
                eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if isinstance(tokenizer._tokenizer, AutoTokenizer) else tokenizer.eos_token_id
                if DEBUG >= 2: print(f"Checking if end of tokens result {tokens[-1]=} is {eos_token_id=}")
                if tokens[-1] == eos_token_id:
                    tokens = tokens[:-1]
                    finish_reason = "stop"

                return web.json_response(generate_completion(chat_request, tokenizer, prompt, request_id, tokens, stream, finish_reason, "chat.completion"))
        except asyncio.TimeoutError:
            return web.json_response({'detail': "Response generation timed out"}, status=408)
        finally:
            deregistered_callback = self.node.on_token.deregister(callback_id)
            if DEBUG >= 2: print(f"Deregister {callback_id=} {deregistered_callback=}")

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        if DEBUG >= 0:
            print(f"Chat interface started. Open this link in your browser: {terminal_link(f'http://localhost:{port}')}")
            print(f"ChatGPT API endpoint served at {terminal_link(f'http://localhost:{port}/v1/chat/completions')}")
