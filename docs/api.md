# EXO API – Technical Reference

This document describes the REST API exposed by the **EXO** service, as implemented in:

`src/exo/master/api.py`

The API is used to manage model instances in the cluster, inspect cluster state, and perform inference using multiple API-compatible interfaces.

Base URL example:

```
http://localhost:52415
```

## 1. General / Meta Endpoints

### Get Master Node ID

**GET** `/node_id`

Returns the identifier of the current master node.

**Response (example):**

```json
{
  "node_id": "node-1234"
}
```

### Get Cluster State

**GET** `/state`

Returns the current state of the cluster, including nodes and active instances.

**Response:**
JSON object describing topology, nodes, and instances.

### Get Events

**GET** `/events`

Returns the list of internal events recorded by the master (mainly for debugging and observability).

**Response:**
Array of event objects.

## 2. Model Instance Management

### Create Instance

**POST** `/instance`

Creates a new model instance in the cluster.

**Request body (example):**

```json
{
  "instance": {
    "model_id": "llama-3.2-1b",
    "placement": { }
  }
}
```

**Response:**
JSON description of the created instance.

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

### Place Instance (Dry Operation)

**POST** `/place_instance`

Performs a placement operation for an instance (planning step), without necessarily creating it.

**Request body:**
JSON describing the instance to be placed.

**Response:**
Placement result.

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
  "model": "llama-3.2-1b",
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
  "model": "llama-3.2-1b",
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

### Claude Messages API

**POST** `/v1/messages`

Executes a chat completion request using the Claude Messages API format. Supports streaming and non-streaming modes.

**Request body (example):**

```json
{
  "model": "llama-3.2-1b",
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
  "model": "llama-3.2-1b",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": false
}
```

**Streaming response format:**
When `stream=true`, returns Server-Sent Events with response-specific event types:

* `response.created` - Response generation started
* `response.in_progress` - Response is being generated
* `response.output_item.added` - New output item added
* `response.output_item.done` - Output item completed
* `response.done` - Response generation completed

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
  "model": "llama-3.2-1b",
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
  "model": "llama-3.2-1b",
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
  "name": "llama-3.2-1b"
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
GET     /instance/{instance_id}
DELETE  /instance/{instance_id}
GET     /instance/previews
GET     /instance/placement
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
