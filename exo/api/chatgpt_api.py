import uuid
import time
import asyncio
import json
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Literal, Union, Dict
from aiohttp import web
import aiohttp_cors
import traceback
import os
import sys
from exo import DEBUG, VERSION
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import PrefixDict
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.models import model_base_shards
from typing import Callable
import signal

class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]):
    self.role = role
    self.content = content

  def to_dict(self):
    return {"role": self.role, "content": self.content}


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float):
    self.model = model
    self.messages = messages
    self.temperature = temperature

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages], "temperature": self.temperature}


def generate_completion(
  chat_request: ChatCompletionRequest,
  tokenizer,
  prompt: str,
  request_id: str,
  tokens: List[int],
  stream: bool,
  finish_reason: Union[Literal["length", "stop"], None],
  object_type: Literal["chat.completion", "text_completion"],
) -> dict:
  completion = {
    "id": f"chatcmpl-{request_id}",
    "object": object_type,
    "created": int(time.time()),
    "model": chat_request.model,
    "system_fingerprint": f"exo_{VERSION}",
    "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": tokenizer.decode(tokens)},
      "logprobs": None,
      "finish_reason": finish_reason,
    }],
  }

  if not stream:
    completion["usage"] = {
      "prompt_tokens": len(tokenizer.encode(prompt)),
      "completion_tokens": len(tokens),
      "total_tokens": len(tokenizer.encode(prompt)) + len(tokens),
    }

  choice = completion["choices"][0]
  if object_type.startswith("chat.completion"):
    key_name = "delta" if stream else "message"
    choice[key_name] = {"role": "assistant", "content": tokenizer.decode(tokens)}
  elif object_type == "text_completion":
    choice["text"] = tokenizer.decode(tokens)
  else:
    ValueError(f"Unsupported response type: {object_type}")

  return completion


def remap_messages(messages: List[Message]) -> List[Message]:
  remapped_messages = []
  last_image = None
  for message in messages:
    if not isinstance(message.content, list):
      remapped_messages.append(message)
      continue

    remapped_content = []
    for content in message.content:
      if isinstance(content, dict):
        if content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if image_url:
            last_image = {"type": "image", "image": image_url}
            remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
        else:
          remapped_content.append(content)
      else:
        remapped_content.append(content)
    remapped_messages.append(Message(role=message.role, content=remapped_content))

  if last_image:
    # Replace the last image placeholder with the actual image content
    for message in reversed(remapped_messages):
      for i, content in enumerate(message.content):
        if isinstance(content, dict):
          if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
            message.content[i] = last_image
            return remapped_messages

  return remapped_messages


def build_prompt(tokenizer, _messages: List[Message]):
  messages = remap_messages(_messages)
  prompt = tokenizer.apply_chat_template([m.to_dict() for m in messages], tokenize=False, add_generation_prompt=True)
  image_str = None
  for message in messages:
    if not isinstance(message.content, list):
      continue

    for content in message.content:
      # note: we only support one image at a time right now. Multiple is possible. See: https://github.com/huggingface/transformers/blob/e68ec18ce224af879f22d904c7505a765fb77de3/docs/source/en/model_doc/llava.md?plain=1#L41
      # follows the convention in https://platform.openai.com/docs/guides/vision
      if isinstance(content, dict) and content.get("type", None) == "image":
        image_str = content.get("image", None)
        break

  return prompt, image_str


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"])


def parse_chat_request(data: dict):
  return ChatCompletionRequest(
    data.get("model", "llama-3.1-8b"),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt

def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "exo" \
    or ('Contents/MacOS' in str(os.path.dirname(sys.executable)))


class ChatGPTAPI:
  def __init__(self, node: Node, inference_engine_classname: str, response_timeout: int = 90, on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    cors = aiohttp_cors.setup(self.app)
    cors_options = aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
      allow_methods="*",
    )
    cors.add(self.app.router.add_get("/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})

    if not is_frozen():
      self.static_dir = Path(__file__).parent.parent/"tinychat"

    self.app.router.add_get("/", self.handle_root)
    self.app.router.add_static("/", self.static_dir, name="static")

    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)
  
  async def handle_quit(self, request):
    print("Received quit signal")
    from exo.main import shutdown
    response = web.json_response({"detail": "Quit signal received"}, status=200)
    await response.prepare(request)
    await response.write_eof()

    await shutdown(signal.SIGINT, asyncio.get_event_loop())
    return response

  async def timeout_middleware(self, app, handler):
    async def middleware(request):
      try:
        return await asyncio.wait_for(handler(request), timeout=self.response_timeout)
      except asyncio.TimeoutError:
        return web.json_response({"detail": "Request timed out"}, status=408)

    return middleware

  async def log_request(self, app, handler):
    async def middleware(request):
      if DEBUG >= 2: print(f"Received request: {request.method} {request.path}")
      return await handler(request)

    return middleware

  async def handle_root(self, request):
    return web.FileResponse(self.static_dir/"index.html")

  async def handle_get_models(self, request):
    return web.json_response([{"id": model_name, "object": "model", "owned_by": "exo", "ready": True} for model_name, _ in model_base_shards.items()])

  async def handle_post_chat_token_encode(self, request):
    data = await request.json()
    shard = model_base_shards.get(data.get("model", "llama-3.1-8b"), {}).get(self.inference_engine_classname)
    messages = [parse_message(msg) for msg in data.get("messages", [])]
    tokenizer = await resolve_tokenizer(shard.model_id)
    return web.json_response({"length": len(build_prompt(tokenizer, messages)[0])})

  async def handle_get_download_progress(self, request):
    progress_data = {}
    for node_id, progress_event in self.node.node_download_progress.items():
      if isinstance(progress_event, RepoProgressEvent):
        progress_data[node_id] = progress_event.to_dict()
      else:
        print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
    return web.json_response(progress_data)

  async def handle_post_chat_completions(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"Handling chat completions request from {request.remote}: {data}")
    stream = data.get("stream", False)
    chat_request = parse_chat_request(data)
    if chat_request.model and chat_request.model.startswith("gpt-"):  # to be compatible with ChatGPT tools, point all gpt- model requests to llama instead
      chat_request.model = "llama-3.1-8b"
    if not chat_request.model or chat_request.model not in model_base_shards:
      if DEBUG >= 1: print(f"Invalid model: {chat_request.model}. Supported: {list(model_base_shards.keys())}. Defaulting to llama-3.1-8b")
      chat_request.model = "llama-3.1-8b"
    shard = model_base_shards[chat_request.model].get(self.inference_engine_classname, None)
    if not shard:
      supported_models = [model for model, engines in model_base_shards.items() if self.inference_engine_classname in engines]
      return web.json_response(
        {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
        status=400,
      )

    tokenizer = await resolve_tokenizer(shard.model_id)
    if DEBUG >= 4: print(f"Resolved tokenizer: {tokenizer}")

    prompt, image_str = build_prompt(tokenizer, chat_request.messages)
    request_id = str(uuid.uuid4())
    if self.on_chat_completion_request:
      try:
        self.on_chat_completion_request(request_id, chat_request, prompt)
      except Exception as e:
        if DEBUG >= 2: traceback.print_exc()
    # request_id = None
    # match = self.prompts.find_longest_prefix(prompt)
    # if match and len(prompt) > len(match[1].prompt):
    #     if DEBUG >= 2:
    #       print(f"Prompt for request starts with previous prompt {len(match[1].prompt)} of {len(prompt)}: {match[1].prompt}")
    #     request_id = match[1].request_id
    #     self.prompts.add(prompt, PromptSession(request_id=request_id, timestamp=int(time.time()), prompt=prompt))
    #     # remove the matching prefix from the prompt
    #     prompt = prompt[len(match[1].prompt):]
    # else:
    #   request_id = str(uuid.uuid4())
    #   self.prompts.add(prompt, PromptSession(request_id=request_id, timestamp=int(time.time()), prompt=prompt))

    callback_id = f"chatgpt-api-wait-response-{request_id}"
    callback = self.node.on_token.register(callback_id)

    if DEBUG >= 2: print(f"Sending prompt from ChatGPT api {request_id=} {shard=} {prompt=} {image_str=}")

    try:
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, image_str, request_id=request_id))), timeout=self.response_timeout)

      if DEBUG >= 2: print(f"Waiting for response to finish. timeout={self.response_timeout}s")

      if stream:
        response = web.StreamResponse(
          status=200,
          reason="OK",
          headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
          },
        )
        await response.prepare(request)

        async def stream_result(_request_id: str, tokens: List[int], is_finished: bool):
          prev_last_tokens_len = self.prev_token_lens.get(_request_id, 0)
          self.prev_token_lens[_request_id] = max(prev_last_tokens_len, len(tokens))
          new_tokens = tokens[prev_last_tokens_len:]
          finish_reason = None
          eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if hasattr(tokenizer, "_tokenizer") and isinstance(tokenizer._tokenizer,
                                                                                                                             AutoTokenizer) else getattr(tokenizer, "eos_token_id", None)
          if len(new_tokens) > 0 and new_tokens[-1] == eos_token_id:
            new_tokens = new_tokens[:-1]
            if is_finished:
              finish_reason = "stop"
          if is_finished and not finish_reason:
            finish_reason = "length"

          completion = generate_completion(
            chat_request,
            tokenizer,
            prompt,
            request_id,
            new_tokens,
            stream,
            finish_reason,
            "chat.completion",
          )
          if DEBUG >= 2: print(f"Streaming completion: {completion}")
          try:
            await response.write(f"data: {json.dumps(completion)}\n\n".encode())
          except Exception as e:
            if DEBUG >= 2: print(f"Error streaming completion: {e}")
            if DEBUG >= 2: traceback.print_exc()

        def on_result(_request_id: str, tokens: List[int], is_finished: bool):
          if _request_id == request_id: self.stream_tasks[_request_id] = asyncio.create_task(stream_result(_request_id, tokens, is_finished))

          return _request_id == request_id and is_finished

        _, tokens, _ = await callback.wait(on_result, timeout=self.response_timeout)
        if request_id in self.stream_tasks:  # in case there is still a stream task running, wait for it to complete
          if DEBUG >= 2: print("Pending stream task. Waiting for stream task to complete.")
          try:
            await asyncio.wait_for(self.stream_tasks[request_id], timeout=30)
          except asyncio.TimeoutError:
            print("WARNING: Stream task timed out. This should not happen.")
        await response.write_eof()
        return response
      else:
        _, tokens, _ = await callback.wait(
          lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished,
          timeout=self.response_timeout,
        )

        finish_reason = "length"
        eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if isinstance(getattr(tokenizer, "_tokenizer", None), AutoTokenizer) else tokenizer.eos_token_id
        if DEBUG >= 2: print(f"Checking if end of tokens result {tokens[-1]=} is {eos_token_id=}")
        if tokens[-1] == eos_token_id:
          tokens = tokens[:-1]
          finish_reason = "stop"

        return web.json_response(generate_completion(chat_request, tokenizer, prompt, request_id, tokens, stream, finish_reason, "chat.completion"))
    except asyncio.TimeoutError:
      return web.json_response({"detail": "Response generation timed out"}, status=408)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)
    finally:
      deregistered_callback = self.node.on_token.deregister(callback_id)
      if DEBUG >= 2: print(f"Deregister {callback_id=} {deregistered_callback=}")

  async def run(self, host: str = "0.0.0.0", port: int = 8000):
    runner = web.AppRunner(self.app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
