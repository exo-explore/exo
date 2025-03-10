import uuid
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional
from aiohttp import web
import aiohttp_cors
import traceback
import signal

from exo import DEBUG
from exo.helpers import PrefixDict, shutdown, get_exo_images_dir
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.models import build_base_shard, build_full_shard, model_cards, get_repo, get_supported_models, get_pretty_name
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import platform

from .helpers import Message, ChatCompletionRequest, generate_completion, build_prompt, parse_message, \
  parse_chat_request, PromptSession
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import delete_model
import tempfile
from exo.apputil import create_animation_mp4
from collections import defaultdict
from .inference_result_manager import InferenceResultManager

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class ChatGPTAPI:
  def __init__(
    self,
    node: Node,
    inference_engine_classname: str,
    response_timeout: int = 90,
    on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None,
    default_model: Optional[str] = None,
    system_prompt: Optional[str] = None
  ):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "llama-3.2-1b"

    # Use InferenceResultManager instead of direct token handling
    self.result_manager = InferenceResultManager(node)
    self.system_prompt = system_prompt

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
    cors.add(self.app.router.add_post("/v1/image/generations", self.handle_post_image_generations), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    cors.add(self.app.router.add_get("/modelpool", self.handle_model_support), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_delete("/models/{model_name}", self.handle_delete_model), {"*": cors_options})
    cors.add(self.app.router.add_get("/initial_models", self.handle_get_initial_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/create_animation", self.handle_create_animation), {"*": cors_options})
    cors.add(self.app.router.add_post("/download", self.handle_post_download), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})

    # Add static routes
    if "__compiled__" not in globals():
      self.static_dir = Path(__file__).parent.parent/"tinychat"
      self.app.router.add_get("/", self.handle_root)
      self.app.router.add_static("/", self.static_dir, name="static")

    # Always add images route, regardless of compilation status
    self.images_dir = get_exo_images_dir()
    self.images_dir.mkdir(parents=True, exist_ok=True)
    self.app.router.add_static('/images/', self.images_dir, name='static_images')

    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)

  async def handle_quit(self, request):
    if DEBUG >= 1: print("Received quit signal")
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
      response = web.StreamResponse(status=200, reason='OK', headers={ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' })
      await response.prepare(request)
      async for path, s in self.node.shard_downloader.get_shard_download_status(self.inference_engine_classname):
        model_data = { s.shard.model_id: { "downloaded": s.downloaded_bytes == s.total_bytes, "download_percentage": 100 if s.downloaded_bytes == s.total_bytes else 100 * float(s.downloaded_bytes) / float(s.total_bytes), "total_size": s.total_bytes, "total_downloaded": s.downloaded_bytes } }
        await response.write(f"data: {json.dumps(model_data)}\n\n".encode())
      await response.write(b"data: [DONE]\n\n")
      return response

    except Exception as e:
      print(f"Error in handle_model_support: {str(e)}")
      traceback.print_exc()
      return web.json_response({"detail": f"Server error: {str(e)}"}, status=500)

  async def handle_get_models(self, request):
    models_list = [{"id": model_name, "object": "model", "owned_by": "exo", "ready": True} for model_name, _ in model_cards.items()]
    return web.json_response({"object": "list", "data": models_list})

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
    prompt = build_prompt(tokenizer, messages, data.get("tools", None))
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
        if progress_event.status != "in_progress": continue
        progress_data[node_id] = progress_event.to_dict()
      else:
        print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
    return web.json_response(progress_data)

  async def handle_post_chat_completions(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"[ChatGPTAPI] Handling chat completions request from {request.remote}: {data}")
    stream = data.get("stream", False)
    chat_request = parse_chat_request(data, self.default_model)
    if chat_request.model and chat_request.model.startswith("gpt-"):  # to be compatible with ChatGPT tools, point all gpt- model requests to default model
      chat_request.model = self.default_model
    if not chat_request.model or chat_request.model not in model_cards:
      if DEBUG >= 1: print(f"[ChatGPTAPI] Invalid model: {chat_request.model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      chat_request.model = self.default_model
    shard = build_base_shard(chat_request.model, self.inference_engine_classname)
    if not shard:
      supported_models = [model for model, info in model_cards.items() if self.inference_engine_classname in info.get("repo", {})]
      return web.json_response(
        {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
        status=400,
      )

    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    if DEBUG >= 4: print(f"[ChatGPTAPI] Resolved tokenizer: {tokenizer}")

    # Add system prompt if set
    if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
      chat_request.messages.insert(0, Message("system", self.system_prompt))

    prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools)
    request_id = str(uuid.uuid4())

    # Register tokenizer and model with the result manager
    self.result_manager.register_tokenizer(request_id, tokenizer)
    self.result_manager.register_model_for_request(request_id, chat_request.model)

    if self.on_chat_completion_request:
      try:
        self.on_chat_completion_request(request_id, chat_request, prompt)
      except Exception as e:
        if DEBUG >= 2: traceback.print_exc()

    if DEBUG >= 2: print(f"[ChatGPTAPI] Processing prompt: {request_id=} {shard=} {prompt=}")

    try:
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(
        shard,
        prompt,
        request_id=request_id,
        generation_options=chat_request.to_generation_options()
      ))), timeout=self.response_timeout)

      if DEBUG >= 2: print(f"[ChatGPTAPI] Waiting for response to finish. timeout={self.response_timeout}s")

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

        try:
          async for chunk in self.result_manager.get_inference_result(request_id, timeout=self.response_timeout):
            if DEBUG >= 2: print(f"[ChatGPTAPI] Got token chunk: {request_id=} {chunk.text=} {chunk.is_finished=} {chunk.finish_reason=}")

            if not chunk.text and not chunk.is_finished:
              continue

            # Generate completion response with tokens for metrics
            completion = generate_completion(
              chat_request,
              tokenizer,
              prompt,
              request_id,
              chunk.tokens,
              chunk.text,
              stream,
              chunk.finish_reason if chunk.is_finished else None,
              "chat.completion",
            )

            await response.write(f"data: {json.dumps(completion)}\n\n".encode())

            if chunk.is_finished:
              # Send a final completion with empty delta to indicate completion
              completion = generate_completion(
                chat_request,
                tokenizer,
                prompt,
                request_id,
                [],
                "",
                stream,
                chunk.finish_reason,
                "chat.completion",
              )
              await response.write(f"data: {json.dumps(completion)}\n\n".encode())
              break

          # Send the DONE event when the stream is finished
          await response.write(b"data: [DONE]\n\n")
          await response.write_eof()
          return response

        except asyncio.TimeoutError:
          if DEBUG >= 2: print(f"[ChatGPTAPI] Timeout waiting for token: {request_id=}")
          return web.json_response({"detail": "Response generation timed out"}, status=408)

        except Exception as e:
          if DEBUG >= 2: traceback.print_exc()
          return web.json_response({"detail": f"Error processing request: {str(e)}"}, status=500)
      else:
        # Non-streaming mode: get complete result
        result = await self.result_manager.get_complete_inference_result(request_id, timeout=self.response_timeout)

        return web.json_response(
          generate_completion(
            chat_request,
            tokenizer,
            prompt,
            request_id,
            result.tokens,
            result.text,
            stream,
            result.finish_reason,
            "chat.completion"
          )
        )

    except asyncio.TimeoutError:
      return web.json_response({"detail": "Response generation timed out"}, status=408)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing request: {str(e)}"}, status=500)

  async def handle_post_image_generations(self, request):
    data = await request.json()

    if DEBUG >= 2: print(f"Handling chat completions request from {request.remote}: {data}")
    stream = data.get("stream", False)
    model = data.get("model", "")
    prompt = data.get("prompt", "")
    image_url = data.get("image_url", "")
    if DEBUG >= 2: print(f"model: {model}, prompt: {prompt}, stream: {stream}")
    shard = build_base_shard(model, self.inference_engine_classname)
    if DEBUG >= 2: print(f"shard: {shard}")
    if not shard:
      return web.json_response({"error": f"Unsupported model: {model} with inference engine {self.inference_engine_classname}"}, status=400)

    request_id = str(uuid.uuid4())
    callback_id = f"chatgpt-api-wait-response-{request_id}"
    callback = self.node.on_token.register(callback_id)
    try:
      if image_url != "" and image_url != None:
        img = self.base64_decode(image_url)
      else:
        img = None
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id, inference_state={"image": img}))), timeout=self.response_timeout)

      response = web.StreamResponse(status=200, reason='OK', headers={
        'Content-Type': 'application/octet-stream',
        "Cache-Control": "no-cache",
      })
      await response.prepare(request)

      def get_progress_bar(current_step, total_steps, bar_length=50):
        # Calculate the percentage of completion
        percent = float(current_step)/total_steps
        # Calculate the number of hashes to display
        arrow = '-'*int(round(percent*bar_length) - 1) + '>'
        spaces = ' '*(bar_length - len(arrow))

        # Create the progress bar string
        progress_bar = f'Progress: [{arrow}{spaces}] {int(percent * 100)}% ({current_step}/{total_steps})'
        return progress_bar

      async def stream_image(_request_id: str, result, is_finished: bool):
        if isinstance(result, list):
          await response.write(json.dumps({'progress': get_progress_bar((result[0]), (result[1]))}).encode('utf-8') + b'\n')

        elif isinstance(result, np.ndarray):
          try:
            im = Image.fromarray(np.array(result))
            # Save the image to a file
            image_filename = f"{_request_id}.png"
            image_path = self.images_dir/image_filename
            im.save(image_path)

            # Get URL for the saved image
            try:
              image_url = request.app.router['static_images'].url_for(filename=image_filename)
              base_url = f"{request.scheme}://{request.host}"
              full_image_url = base_url + str(image_url)

              await response.write(json.dumps({'images': [{'url': str(full_image_url), 'content_type': 'image/png'}]}).encode('utf-8') + b'\n')
            except KeyError as e:
              if DEBUG >= 2: print(f"Error getting image URL: {e}")
              # Fallback to direct file path if URL generation fails
              await response.write(json.dumps({'images': [{'url': str(image_path), 'content_type': 'image/png'}]}).encode('utf-8') + b'\n')

            if is_finished:
              await response.write_eof()

          except Exception as e:
            if DEBUG >= 2: print(f"Error processing image: {e}")
            if DEBUG >= 2: traceback.print_exc()
            await response.write(json.dumps({'error': str(e)}).encode('utf-8') + b'\n')

      stream_task = None

      def on_result(_request_id: str, result, is_finished: bool):
        nonlocal stream_task
        stream_task = asyncio.create_task(stream_image(_request_id, result, is_finished))
        return _request_id == request_id and is_finished

      await callback.wait(on_result, timeout=self.response_timeout*10)

      if stream_task:
        # Wait for the stream task to complete before returning
        await stream_task

      return response

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)

  async def handle_delete_model(self, request):
    model_id = request.match_info.get('model_name')
    try:
      if await delete_model(model_id, self.inference_engine_classname): return web.json_response({"status": "success", "message": f"Model {model_id} deleted successfully"})
      else: return web.json_response({"detail": f"Model {model_id} files not found"}, status=404)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error deleting model: {str(e)}"}, status=500)

  async def handle_get_initial_models(self, request):
    model_data = {}
    for model_id in get_supported_models([[self.inference_engine_classname]]):
      model_data[model_id] = {
        "name": get_pretty_name(model_id),
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
      create_animation_mp4(replacement_image_path, output_path, device_name, prompt_text)

      return web.json_response({"status": "success", "output_path": output_path})

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)

  async def handle_post_download(self, request):
    try:
      data = await request.json()
      model_name = data.get("model")
      if not model_name: return web.json_response({"error": "model parameter is required"}, status=400)
      if model_name not in model_cards: return web.json_response({"error": f"Invalid model: {model_name}. Supported models: {list(model_cards.keys())}"}, status=400)
      shard = build_full_shard(model_name, self.inference_engine_classname)
      if not shard: return web.json_response({"error": f"Could not build shard for model {model_name}"}, status=400)
      asyncio.create_task(self.node.inference_engine.shard_downloader.ensure_shard(shard, self.inference_engine_classname))

      return web.json_response({"status": "success", "message": f"Download started for model: {model_name}"})
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
      return web.json_response({"detail": f"Error getting topology: {str(e)}"}, status=500)

  async def run(self, host: str = "0.0.0.0", port: int = 52415):
    runner = web.AppRunner(self.app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

  def base64_decode(self, base64_string):
    #decode and reshape image
    if base64_string.startswith('data:image'):
      base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data))
    W, H = (dim - dim%64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
      if DEBUG >= 2: print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
      img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter
    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32)/255)*2 - 1
    img = img[None]
    return img
