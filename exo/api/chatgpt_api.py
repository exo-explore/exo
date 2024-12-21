import uuid
import time
import asyncio
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Literal, Union, Dict
from aiohttp import web
import aiohttp_cors
import traceback
import signal
from exo import DEBUG, VERSION
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import PrefixDict, shutdown
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.models import build_base_shard, model_cards, get_repo, pretty_name
from typing import Callable, Optional
from exo.download.hf.hf_shard_download import HFShardDownloader
import shutil
from exo.download.hf.hf_helpers import get_hf_home, get_repo_root
from exo.apputil import create_animation_mp4

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
  for message in messages:
    if not isinstance(message.content, list):
      continue

  return prompt


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"])


def parse_chat_request(data: dict, default_model: str):
  return ChatCompletionRequest(
    data.get("model", default_model),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt

class ChatGPTAPI:
  def __init__(self, node: Node, inference_engine_classname: str, response_timeout: int = 90, on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None, default_model: Optional[str] = None):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "llama-3.2-1b"

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
    cors.add(self.app.router.add_get("/modelpool", self.handle_model_support), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_delete("/models/{model_name}", self.handle_delete_model), {"*": cors_options})
    cors.add(self.app.router.add_get("/initial_models", self.handle_get_initial_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/create_animation", self.handle_create_animation), {"*": cors_options})
    cors.add(self.app.router.add_post("/download", self.handle_post_download), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})

    if "__compiled__" not in globals():
      self.static_dir = Path(__file__).parent.parent/"tinychat"
      self.app.router.add_get("/", self.handle_root)
      self.app.router.add_static("/", self.static_dir, name="static")

    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)

  async def handle_quit(self, request):
    if DEBUG>=1: print("Received quit signal")
    response = web.json_response({"detail": "Quit signal received"}, status=200)
    await response.prepare(request)
    await response.write_eof()
    await shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server)

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

  async def handle_healthcheck(self, request):
    return web.json_response({"status": "ok"})

  async def handle_model_support(self, request):
    try:
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        await response.prepare(request)

        for model_name, pretty in pretty_name.items():
            if model_name in model_cards:
                model_info = model_cards[model_name]

                if self.inference_engine_classname in model_info.get("repo", {}):
                    shard = build_base_shard(model_name, self.inference_engine_classname)
                    if shard:
                        downloader = HFShardDownloader(quick_check=True)
                        downloader.current_shard = shard
                        downloader.current_repo_id = get_repo(shard.model_id, self.inference_engine_classname)
                        status = await downloader.get_shard_download_status()

                        download_percentage = status.get("overall") if status else None
                        total_size = status.get("total_size") if status else None
                        total_downloaded = status.get("total_downloaded") if status else False

                        model_data = {
                            model_name: {
                                "name": pretty,
                                "downloaded": download_percentage == 100 if download_percentage is not None else False,
                                "download_percentage": download_percentage,
                                "total_size": total_size,
                                "total_downloaded": total_downloaded
                            }
                        }

                        await response.write(f"data: {json.dumps(model_data)}\n\n".encode())

        await response.write(b"data: [DONE]\n\n")
        return response

    except Exception as e:
        print(f"Error in handle_model_support: {str(e)}")
        traceback.print_exc()
        return web.json_response(
            {"detail": f"Server error: {str(e)}"},
            status=500
        )

  async def handle_get_models(self, request):
    return web.json_response([{"id": model_name, "object": "model", "owned_by": "exo", "ready": True} for model_name, _ in model_cards.items()])

  async def handle_post_chat_token_encode(self, request):
    data = await request.json()
    model = data.get("model", self.default_model)
    if model and model.startswith("gpt-"):  # Handle gpt- model requests
      model = self.default_model
    if not model or model not in model_cards:
      if DEBUG >= 1: print(f"Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      model = self.default_model
    shard = build_base_shard(model, self.inference_engine_classname)
    messages = [parse_message(msg) for msg in data.get("messages", [])]
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    prompt = build_prompt(tokenizer, messages)
    tokens = tokenizer.encode(prompt)
    return web.json_response({
      "length": len(prompt),
      "num_tokens": len(tokens),
      "encoded_tokens": tokens,
      "encoded_prompt": prompt,
    })

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
    chat_request = parse_chat_request(data, self.default_model)
    if chat_request.model and chat_request.model.startswith("gpt-"):  # to be compatible with ChatGPT tools, point all gpt- model requests to default model
      chat_request.model = self.default_model
    if not chat_request.model or chat_request.model not in model_cards:
      if DEBUG >= 1: print(f"Invalid model: {chat_request.model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      chat_request.model = self.default_model
    shard = build_base_shard(chat_request.model, self.inference_engine_classname)
    if not shard:
      supported_models = [model for model, info in model_cards.items() if self.inference_engine_classname in info.get("repo", {})]
      return web.json_response(
        {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
        status=400,
      )

    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    if DEBUG >= 4: print(f"Resolved tokenizer: {tokenizer}")

    prompt = build_prompt(tokenizer, chat_request.messages)
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

    if DEBUG >= 2: print(f"Sending prompt from ChatGPT api {request_id=} {shard=} {prompt=}")

    try:
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id))), timeout=self.response_timeout)

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

  async def handle_delete_model(self, request):
    try:
      model_name = request.match_info.get('model_name')
      if DEBUG >= 2: print(f"Attempting to delete model: {model_name}")

      if not model_name or model_name not in model_cards:
        return web.json_response(
          {"detail": f"Invalid model name: {model_name}"},
          status=400
          )

      shard = build_base_shard(model_name, self.inference_engine_classname)
      if not shard:
        return web.json_response(
          {"detail": "Could not build shard for model"},
          status=400
        )

      repo_id = get_repo(shard.model_id, self.inference_engine_classname)
      if DEBUG >= 2: print(f"Repo ID for model: {repo_id}")

      # Get the HF cache directory using the helper function
      hf_home = get_hf_home()
      cache_dir = get_repo_root(repo_id)

      if DEBUG >= 2: print(f"Looking for model files in: {cache_dir}")

      if os.path.exists(cache_dir):
        if DEBUG >= 2: print(f"Found model files at {cache_dir}, deleting...")
        try:
          shutil.rmtree(cache_dir)
          return web.json_response({
            "status": "success",
            "message": f"Model {model_name} deleted successfully",
            "path": str(cache_dir)
          })
        except Exception as e:
          return web.json_response({
            "detail": f"Failed to delete model files: {str(e)}"
          }, status=500)
      else:
        return web.json_response({
          "detail": f"Model files not found at {cache_dir}"
        }, status=404)

    except Exception as e:
        print(f"Error in handle_delete_model: {str(e)}")
        traceback.print_exc()
        return web.json_response({
            "detail": f"Server error: {str(e)}"
        }, status=500)

  async def handle_get_initial_models(self, request):
    model_data = {}
    for model_name, pretty in pretty_name.items():
        model_data[model_name] = {
            "name": pretty,
            "downloaded": None,  # Initially unknown
            "download_percentage": None,  # Change from 0 to null
            "total_size": None,
            "total_downloaded": None,
            "loading": True  # Add loading state
        }
    return web.json_response(model_data)

  async def handle_create_animation(self, request):
    try:
      data = await request.json()
      replacement_image_path = data.get("replacement_image_path")
      device_name = data.get("device_name", "Local Device")
      prompt_text = data.get("prompt", "")

      if DEBUG >= 2: print(f"Creating animation with params: replacement_image={replacement_image_path}, device={device_name}, prompt={prompt_text}")

      if not replacement_image_path:
        return web.json_response({"error": "replacement_image_path is required"}, status=400)

      # Create temp directory if it doesn't exist
      tmp_dir = Path(tempfile.gettempdir())/"exo_animations"
      tmp_dir.mkdir(parents=True, exist_ok=True)

      # Generate unique output filename in temp directory
      output_filename = f"animation_{uuid.uuid4()}.mp4"
      output_path = str(tmp_dir/output_filename)

      if DEBUG >= 2: print(f"Animation temp directory: {tmp_dir}, output file: {output_path}, directory exists: {tmp_dir.exists()}, directory permissions: {oct(tmp_dir.stat().st_mode)[-3:]}")

      # Create the animation
      create_animation_mp4(
        replacement_image_path,
        output_path,
        device_name,
        prompt_text
      )

      return web.json_response({
        "status": "success",
        "output_path": output_path
      })

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)

  async def handle_post_download(self, request):
    try:
      data = await request.json()
      model_name = data.get("model")
      if not model_name: return web.json_response({"error": "model parameter is required"}, status=400)
      if model_name not in model_cards: return web.json_response({"error": f"Invalid model: {model_name}. Supported models: {list(model_cards.keys())}"}, status=400)
      shard = build_base_shard(model_name, self.inference_engine_classname)
      if not shard: return web.json_response({"error": f"Could not build shard for model {model_name}"}, status=400)
      asyncio.create_task(self.node.inference_engine.ensure_shard(shard))

      return web.json_response({
        "status": "success",
        "message": f"Download started for model: {model_name}"
      })
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)

  async def handle_get_topology(self, request):
    try:
      topology = self.node.current_topology
      if topology:
        return web.json_response(topology.to_json())
      else:
        return web.json_response({})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response(
        {"detail": f"Error getting topology: {str(e)}"},
        status=500
      )

  async def run(self, host: str = "0.0.0.0", port: int = 52415):
    runner = web.AppRunner(self.app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
