# EXO API – Technical Reference

This document describes the REST API exposed by the **EXO** service, as implemented in:

`src/exo/api/main.py` (request and response types are in `src/exo/api/types/`)

The API is used to manage model instances in the cluster, inspect cluster state, and perform inference using multiple API-compatible interfaces.

Base URL example:

```
http://localhost:52415
```

## 1. General / Meta Endpoints

### Get This Node's ID

**GET** `/node_id`

Returns the ID of the node that serves this API (this device, not necessarily the master), as a bare JSON string. A node picks a new random ID each time EXO starts.

**Response (example):**

```json
"3f9c2a7e1b5d4c8e9a0f6b2d7c4e1a3b"
```

### Get Cluster State

**GET** `/state`

Returns the current state of the cluster, including nodes and active instances.

**Response:**
JSON object describing topology, nodes, and instances.

### Get Events

**GET** `/events`

Returns the most recent internal events this node has applied, up to 10,000 of them (mainly for debugging and observability). Every generated token is an event, so under load this covers the last few seconds to minutes. A node that joined late, or fell far behind, catches up from a state snapshot, so it only has the events after that point.

**Response:**
Array of event objects.

## 2. Model Instance Management

### Create Instance

**POST** `/instance`

Creates a new model instance in the cluster.

**Request body (example):**

```json
{
  "instance": { ... }
}
```

The `instance` is a placement as returned by `/instance/previews` (each preview's
`instance` field), for example the first preview without an error.

**Response:**
Command acknowledgement. Instance creation is asynchronous; clients should wait
for the model to appear through `/instance/await` before sending inference
requests for that model.

### Delete Instance

**DELETE** `/instance/{instance_id}`

Deletes an existing instance by ID.

**Path parameters:**

* `instance_id`: string, ID of the instance to delete

**Response:**
Status / confirmation JSON.

### Get Instance

**GET** `/instance/{instance_id}`

Returns details of a specific instance.

**Path parameters:**

* `instance_id`: string

**Response:**
JSON description of the instance.

### Await Instance

**GET** `/instance/await?model_id=...&timeout_seconds=0`

Waits until API state contains an instance for the requested model. The response
is an SSE stream so clients receive keep-alive comments while waiting.

**Query parameters:**

* `model_id`: string, required
* `timeout_seconds`: float, optional, default `0`. `0` waits indefinitely;
  positive values time out after that many seconds. Maximum positive value:
  `300`.

**Stream messages:**

```text
data: {"type": "ready", "instance": {...}}

data: {"type": "timeout", "message": "No instance found for model ..."}
```

The HTTP status is `200` for both messages because the stream starts before the
final result is known. The `type` field disambiguates the terminal message.

### Preview Placements

**GET** `/instance/previews?model_id=...`

Returns possible placement previews for a given model.

**Query parameters:**

* `model_id`: string, required

**Response:**
Array of placement preview objects.

### Compute Placement

**GET** `/instance/placement`

Computes a placement for a potential instance without creating it.

**Query parameters (typical):**

* `model_id`: string
* `sharding`: string or config
* `instance_meta`: JSON-encoded metadata
* `min_nodes`: integer

**Response:**
JSON object describing the proposed placement / instance configuration.

### Place Instance

**POST** `/place_instance`

Places an instance for a model using the server's placement logic.

**Request body:**
JSON describing the instance to be placed.

**Response:**
Command acknowledgement, sent once the instance is in the cluster's state. The
model may still be downloading or loading; wait for it through `/instance/await`
before sending inference requests.

Requests that change the cluster (`POST /instance`, `POST /place_instance`,
`DELETE /instance/{instance_id}`, the instance-link endpoints, `POST /models/add`
and `DELETE /models/custom/{model_id}`) are answered once the change shows in the
cluster's state, or at once if the state already shows it (removing a link that
is already gone, say). If the device coordinating the cluster is lost meanwhile, the
change is sent again to the device that takes over. A change not seen within
about 45 s is answered with `503`:

- a launch (`POST /instance`, `POST /place_instance`) or a new link
  (`POST /v1/instance-links`) is then called off: if it still reaches the cluster
  later, the node that answered removes it again (unless it has restarted since),
  so trying again doesn't leave two running. Trying the same instance again takes
  over from the failed request;
- any other change may still apply later; trying it again is safe.

## 3. Models

### List Models

**GET** `/models`
**GET** `/v1/models` (alias)

Returns the list of available models and their metadata.

**Query parameters:**

* `status`: string (optional) - Filter by `downloaded` to show only downloaded models

**Response:**
Array of model descriptors including `is_custom` field for custom HuggingFace models.

### Add Custom Model

**POST** `/models/add`

Add a custom model from HuggingFace hub.

**Request body (example):**

```json
{
  "model_id": "mlx-community/my-custom-model"
}
```

**Response:**
Model descriptor for the added model.

**Security note:**
Models with `trust_remote_code` enabled in their configuration require explicit opt-in (default is false) for security.

### Delete Custom Model

**DELETE** `/models/custom/{model_id}`

Delete a user-added custom model card.

**Path parameters:**

* `model_id`: string, ID of the custom model to delete

**Response:**
Confirmation JSON with deleted model ID.

### Search Models

**GET** `/models/search`

Search HuggingFace Hub for mlx-community models.

**Query parameters:**

* `query`: string (optional) - Search query
* `limit`: integer (default: 20) - Maximum number of results

**Response:**
Array of HuggingFace model search results.

## 4. Inference / Chat Completions

### OpenAI-Compatible Chat Completions

**POST** `/v1/chat/completions`

Executes a chat completion request using an OpenAI-compatible schema. Supports streaming and non-streaming modes.

**Request body (example):**

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**Request parameters:**

* `model`: string, required - Model ID to use
* `messages`: array, required - Conversation messages
* `stream`: boolean (default: false) - Enable streaming responses
* `max_tokens`: integer (optional) - Maximum tokens to generate
* `temperature`: float (optional) - Sampling temperature
* `top_p`: float (optional) - Nucleus sampling parameter
* `top_k`: integer (optional) - Top-k sampling parameter
* `stop`: string or array (optional) - Stop sequences
* `seed`: integer (optional) - Random seed for reproducibility
* `enable_thinking`: boolean (optional) - Enable thinking mode for capable models (DeepSeek V3.1, Qwen3, GLM-4.7)
* `tools`: array (optional) - Tool definitions for function calling
* `logprobs`: boolean (optional) - Return log probabilities
* `top_logprobs`: integer (optional) - Number of top log probabilities to return

**Response:**
OpenAI-compatible chat completion response.

**Streaming response format:**
When `stream=true`, returns Server-Sent Events (SSE) with format:

```
data: {"id":"...","object":"chat.completion","created":...,"model":"...","choices":[...]}

data: [DONE]
```

**Non-streaming response includes usage statistics:**

```json
{
  "id": "...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

**Cancellation:**
You can cancel an active generation by closing the HTTP connection. The server detects the disconnection and stops processing.

**Errors during generation:**
A generation that fails after it started ends with an error event in a stream, or a `500` response otherwise. The message is one short sentence for the user (the details are in the node's log), and `code` says why. The Claude Messages and Responses endpoints report it their own way: a stream ends with an `error` event (Claude, with the code in `error.code`) or a `response.failed` event (Responses, with the code in `response.error.code`).

```
data: {"error":{"message":"A device serving this model disconnected before the reply finished.","type":"InternalServerError","code":"device_left"}}

data: [DONE]
```

| `code` | Meaning |
|---|---|
| `device_left` | A device serving the model disconnected or left the cluster |
| `model_stopped` | The model was stopped while the request was in flight |
| `out_of_memory` | A device serving the model ran out of GPU memory |
| `gpu_timeout` | The GPU of a device serving the model timed out |
| `model_stalled` | The model stopped making progress and is being restarted |
| `model_failed` | The model's process failed in some other way |
| `interrupted` | The request ended for another reason, such as a change in the cluster |
| `model_not_ready` | The model isn't ready to serve yet (with a `503`; see below) |
| `context_length_exceeded` | The prompt and `max_tokens` don't fit in the model's context (see below) |
| `invalid_request` | The request can't be run on this model, such as one its chat template can't render |

Errors without a known cause have the HTTP status (`500`) as their `code`.

**A request too long for the model:**
A request whose prompt, plus `max_tokens` if given, doesn't fit in the model's context window is refused before any work is done on it, with OpenAI's error:

```json
{"error": {"message": "This model's maximum context length is 131072 tokens. However, your messages resulted in 160000 tokens. Please reduce the length of the messages.", "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

A whole response gets it with a `400`. A stream gets it as its error event (with a `200`, as its response has started by then: only the model's runner can count the prompt's tokens), and so do the Claude (`invalid_request_error`) and Responses streams. The limit is the most that the model's card (`context_length` in `/models`) or its `config.json` says the model takes.

**A model that isn't ready yet:**
A request for a model with no instance gets `404`. A request for a model whose instance can't serve yet (it is downloading, loading or warming up) waits for it for up to 30 s and goes ahead as soon as it is ready. If it still isn't, the request gets `503 Service Unavailable`, with a `Retry-After` header (seconds) and a message saying what the model is doing:

```json
{"error": {"message": "The model is not ready yet: downloading (42%, about 3 min left). Try again in 60 s.", "type": "Service Unavailable", "code": "model_not_ready"}}
```

A model that failed to start, or whose download failed, gets its `503` at once. The wait is set with the `EXO_MODEL_READY_WAIT_SECONDS` environment variable: `0` answers at once, and a larger value lets clients that would rather wait for a whole download do so (the OpenAI SDKs wait up to 600 s for a response by default).

This applies to every inference endpoint (OpenAI, Claude, Responses, Ollama and image generation).

### Claude Messages API

**POST** `/v1/messages`

Executes a chat completion request using the Claude Messages API format. Supports streaming and non-streaming modes.

**Request body (example):**

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "max_tokens": 1024,
  "stream": false
}
```

**Streaming response format:**
When `stream=true`, returns Server-Sent Events with Claude-specific event types:

* `message_start` - Message generation started
* `content_block_start` - Content block started
* `content_block_delta` - Incremental content chunk
* `content_block_stop` - Content block completed
* `message_delta` - Message metadata updates
* `message_stop` - Message generation completed

**Response:**
Claude-compatible messages response.

### OpenAI Responses API

**POST** `/v1/responses`

Executes a chat completion request using the OpenAI Responses API format. Supports streaming and non-streaming modes.

**Request body (example):**

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "input": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**Request parameters:**

* `model`: string, required - Model ID to use
* `input`: string or array, required - The prompt, or a list of input items such as messages
* `instructions`: string (optional) - System instructions
* `max_output_tokens`: integer (optional) - Maximum tokens to generate

**Streaming response format:**
When `stream=true`, returns Server-Sent Events with response-specific event types:

* `response.created` - Response generation started
* `response.in_progress` - Response is being generated
* `response.output_item.added` - New output item added
* `response.output_text.delta` - Incremental text chunk
* `response.output_item.done` - Output item completed
* `response.completed` - Response generation completed

**Response:**
OpenAI Responses API-compatible response.

### Benchmarked Chat Completions

**POST** `/bench/chat/completions`

Same as `/v1/chat/completions`, but also returns performance and generation statistics.

**Request body:**
Same schema as `/v1/chat/completions`.

**Response:**
Chat completion plus benchmarking metrics including:

* `prompt_tps` - Tokens per second during prompt processing
* `generation_tps` - Tokens per second during generation
* `prompt_tokens` - Number of prompt tokens
* `generation_tokens` - Number of generated tokens
* `peak_memory_usage` - Peak memory used during generation

### Cancel Command

**POST** `/v1/cancel/{command_id}`

Cancels an active generation command (text or image). Notifies workers and closes the stream.

**Path parameters:**

* `command_id`: string, ID of the command to cancel

**Response (example):**

```json
{
  "message": "Command cancelled.",
  "command_id": "cmd-abc-123"
}
```

Returns 404 if the command is not found or already completed.

## 5. Ollama API Compatibility

EXO provides Ollama API compatibility for tools like OpenWebUI.

### Ollama Chat

**POST** `/ollama/api/chat`
**POST** `/ollama/api/api/chat` (alias)
**POST** `/ollama/api/v1/chat` (alias)

Execute a chat request using Ollama API format.

**Request body (example):**

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**Response:**
Ollama-compatible chat response.

### Ollama Generate

**POST** `/ollama/api/generate`

Execute a text generation request using Ollama API format.

**Request body (example):**

```json
{
  "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "prompt": "Hello",
  "stream": false
}
```

**Response:**
Ollama-compatible generation response.

### Ollama Tags

**GET** `/ollama/api/tags`
**GET** `/ollama/api/api/tags` (alias)
**GET** `/ollama/api/v1/tags` (alias)

Returns list of downloaded models in Ollama tags format.

**Response:**
Array of model tags with metadata.

### Ollama Show

**POST** `/ollama/api/show`

Returns model information in Ollama show format.

**Request body:**

```json
{
  "name": "mlx-community/Llama-3.2-1B-Instruct-4bit"
}
```

**Response:**
Model details including modelfile and family.

### Ollama PS

**GET** `/ollama/api/ps`

Returns list of running models (active instances).

**Response:**
Array of active model instances.

### Ollama Version

**GET** `/ollama/api/version`
**HEAD** `/ollama/` (alias)
**HEAD** `/ollama/api/version` (alias)

Returns version information for Ollama API compatibility.

**Response:**

```json
{
  "version": "exo v1.0"
}
```

## 6. Image Generation & Editing

### Image Generation

**POST** `/v1/images/generations`

Executes an image generation request using an OpenAI-compatible schema with additional advanced_params. Supports both streaming and non-streaming modes.

**Request body (example):**

```json
{
  "prompt": "a robot playing chess",
  "model": "exolabs/FLUX.1-dev",
  "n": 1,
  "size": "1024x1024",
  "stream": false,
  "response_format": "b64_json"
}
```

**Request parameters:**

* `prompt`: string, required - Text description of the image
* `model`: string, required - Image model ID
* `n`: integer (default: 1) - Number of images to generate
* `size`: string (default: "auto") - Image dimensions. Supported sizes:
  - `512x512`
  - `768x768`
  - `1024x768`
  - `768x1024`
  - `1024x1024`
  - `1024x1536`
  - `1536x1024`
  - `1024x1365`
  - `1365x1024`
* `stream`: boolean (default: false) - Enable streaming for partial images
* `partial_images`: integer (default: 0) - Number of partial images to stream during generation
* `response_format`: string (default: "b64_json") - Either `url` or `b64_json`
* `quality`: string (default: "medium") - Either `high`, `medium`, or `low`
* `output_format`: string (default: "png") - Either `png`, `jpeg`, or `webp`
* `advanced_params`: object (optional) - Advanced generation parameters

**Advanced Parameters (`advanced_params`):**

| Parameter | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `seed` | int | >= 0 | Random seed for reproducible generation |
| `num_inference_steps` | int | 1-100 | Number of denoising steps |
| `guidance` | float | 1.0-20.0 | Classifier-free guidance scale |
| `negative_prompt` | string | - | Text describing what to avoid in the image |

**Non-streaming response:**

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
      "url": null
    }
  ]
}
```

**Streaming response format:**
When `stream=true` and `partial_images > 0`, returns Server-Sent Events:

```
data: {"type":"partial","image_index":0,"partial_index":1,"total_partials":5,"format":"png","data":{"b64_json":"..."}}

data: {"type":"final","image_index":0,"format":"png","data":{"b64_json":"..."}}

data: [DONE]
```

### Image Editing

**POST** `/v1/images/edits`

Executes an image editing request (img2img) using FLUX.1-Kontext-dev or similar models.

**Request (multipart/form-data):**

* `image`: file, required - Input image to edit
* `prompt`: string, required - Text description of desired changes
* `model`: string, required - Image editing model ID (e.g., `exolabs/FLUX.1-Kontext-dev`)
* `n`: integer (default: 1) - Number of edited images to generate
* `size`: string (optional) - Output image dimensions
* `response_format`: string (default: "b64_json") - Either `url` or `b64_json`
* `input_fidelity`: string (default: "low") - Either `low` or `high` - Controls how closely the output follows the input image
* `stream`: string (default: "false") - Enable streaming
* `partial_images`: string (default: "0") - Number of partial images to stream
* `quality`: string (default: "medium") - Either `high`, `medium`, or `low`
* `output_format`: string (default: "png") - Either `png`, `jpeg`, or `webp`
* `advanced_params`: string (optional) - JSON-encoded advanced parameters

**Response:**
Same format as `/v1/images/generations`.

### Benchmarked Image Generation

**POST** `/bench/images/generations`

Same as `/v1/images/generations`, but also returns generation statistics.

**Request body:**
Same schema as `/v1/images/generations`.

**Response:**
Image generation plus benchmarking metrics including:

* `seconds_per_step` - Average time per denoising step
* `total_generation_time` - Total generation time
* `num_inference_steps` - Number of inference steps used
* `num_images` - Number of images generated
* `image_width` - Output image width
* `image_height` - Output image height
* `peak_memory_usage` - Peak memory used during generation

### Benchmarked Image Editing

**POST** `/bench/images/edits`

Same as `/v1/images/edits`, but also returns generation statistics.

**Request:**
Same schema as `/v1/images/edits`.

**Response:**
Same format as `/bench/images/generations`, including `generation_stats`.

### List Images

**GET** `/images`

List all stored images.

**Response:**
Array of image metadata including URLs and expiration times.

### Get Image

**GET** `/images/{image_id}`

Retrieve a stored image by ID.

**Path parameters:**

* `image_id`: string, ID of the image

**Response:**
Image file with appropriate content type.

## 7. Complete Endpoint Summary

```
# General
GET     /node_id
GET     /state
GET     /events

# Instance Management
POST    /instance
GET     /instance/await
GET     /instance/previews
GET     /instance/placement
GET     /instance/{instance_id}
DELETE  /instance/{instance_id}
POST    /place_instance

# Models
GET     /models
GET     /v1/models
POST    /models/add
DELETE  /models/custom/{model_id}
GET     /models/search

# Text Generation (OpenAI Chat Completions)
POST    /v1/chat/completions
POST    /bench/chat/completions

# Text Generation (Claude Messages API)
POST    /v1/messages

# Text Generation (OpenAI Responses API)
POST    /v1/responses

# Text Generation (Ollama API)
POST    /ollama/api/chat
POST    /ollama/api/api/chat
POST    /ollama/api/v1/chat
POST    /ollama/api/generate
GET     /ollama/api/tags
GET     /ollama/api/api/tags
GET     /ollama/api/v1/tags
POST    /ollama/api/show
GET     /ollama/api/ps
GET     /ollama/api/version
HEAD    /ollama/
HEAD    /ollama/api/version

# Command Control
POST    /v1/cancel/{command_id}

# Image Generation
POST    /v1/images/generations
POST    /bench/images/generations
POST    /v1/images/edits
POST    /bench/images/edits
GET     /images
GET     /images/{image_id}
```

## 8. Notes

### API Compatibility

EXO provides multiple API-compatible interfaces:

* **OpenAI Chat Completions API** - Compatible with OpenAI clients and tools
* **Claude Messages API** - Compatible with Anthropic's Claude API format
* **OpenAI Responses API** - Compatible with OpenAI's Responses API format
* **Ollama API** - Compatible with Ollama and tools like OpenWebUI

Existing OpenAI, Claude, or Ollama clients can be pointed to EXO by changing the base URL.

### Custom Models

You can add custom models from HuggingFace using the `/models/add` endpoint. Custom models are identified by the `is_custom` field in model list responses.

**Security:** Models requiring `trust_remote_code` must be explicitly enabled (default is false) for security. Only enable this if you trust the model's remote code.

### Usage Statistics

Chat completion responses include usage statistics with:

* `prompt_tokens` - Number of tokens in the prompt
* `completion_tokens` - Number of tokens generated
* `total_tokens` - Sum of prompt and completion tokens

### Request Cancellation

You can cancel active requests by:

1. Closing the HTTP connection (for streaming requests)
2. Calling `/v1/cancel/{command_id}` (for any request)

The server detects cancellation and stops processing immediately.

### Instance Placement

The instance placement endpoints allow you to plan and preview cluster allocations before creating instances. This helps optimize resource usage across nodes.

### Observability

The `/events` and `/state` endpoints are primarily intended for operational visibility and debugging.
