# EXO API – Technical Reference

This document describes the REST API exposed by the **EXO** service, as implemented in:

`src/exo/master/api.py`

The API is used to manage model instances in the cluster, inspect cluster state, and perform inference using an OpenAI-compatible interface.

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

**Response:**
Array of model descriptors.

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

**Response:**
OpenAI-compatible chat completion response.

### Benchmarked Chat Completions

**POST** `/bench/chat/completions`

Same as `/v1/chat/completions`, but also returns performance and generation statistics.

**Request body:**
Same schema as `/v1/chat/completions`.

**Response:**
Chat completion plus benchmarking metrics.

## 5. Image Generation & Editing

### Image Generation

**POST** `/v1/images/generations`

Executes an image generation request using an OpenAI-compatible schema with additional advanced_params.

**Request body (example):**

```json
{
  "prompt": "a robot playing chess",
  "model": "flux-dev",
  "stream": false,
}
```

**Advanced Parameters (`advanced_params`):**

| Parameter | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `seed` | int | >= 0 | Random seed for reproducible generation |
| `num_inference_steps` | int | 1-100 | Number of denoising steps |
| `guidance` | float | 1.0-20.0 | Classifier-free guidance scale |
| `negative_prompt` | string | - | Text describing what to avoid in the image |

**Response:**
OpenAI-compatible image generation response.

### Benchmarked Image Generation

**POST** `/bench/images/generations`

Same as `/v1/images/generations`, but also returns generation statistics.

**Request body:**
Same schema as `/v1/images/generations`.

**Response:**
Image generation plus benchmarking metrics.

### Image Editing

**POST** `/v1/images/edits`

Executes an image editing request using an OpenAI-compatible schema with additional advanced_params (same as `/v1/images/generations`).

**Response:**
Same format as `/v1/images/generations`.

### Benchmarked Image Editing

**POST** `/bench/images/edits`

Same as `/v1/images/edits`, but also returns generation statistics.

**Request:**
Same schema as `/v1/images/edits`.

**Response:**
Same format as `/bench/images/generations`, including `generation_stats`.

## 6. Complete Endpoint Summary

```
GET     /node_id
GET     /state
GET     /events

POST    /instance
GET     /instance/{instance_id}
DELETE  /instance/{instance_id}

GET     /instance/previews
GET     /instance/placement
POST    /place_instance

GET     /models
GET     /v1/models

POST    /v1/chat/completions
POST    /bench/chat/completions

POST    /v1/images/generations
POST    /bench/images/generations
POST    /v1/images/edits
POST    /bench/images/edits
```

## 7. Notes

* The `/v1/chat/completions` endpoint is compatible with the OpenAI Chat API format, so existing OpenAI clients can be pointed to EXO by changing the base URL.
* The `/v1/images/generations` and `/v1/images/edits` endpoints are compatible with the OpenAI Images API format.
* The instance placement endpoints allow you to plan and preview cluster allocations before actually creating instances.
* The `/events` and `/state` endpoints are primarily intended for operational visibility and debugging.
