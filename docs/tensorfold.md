# TensorFold runtime

EXO can run a complete text model through a locally installed TensorFold native
executable. EXO owns model downloads, placement, request routing and runner
supervision. TensorFold owns tokenization, chat templates, generation, batching
and its native prompt cache.

The first integration supports one complete model on one Metal or CUDA device.
It does not split a TensorFold model across EXO nodes. A cluster can contain
both MLX and TensorFold instances, and the dashboard lets you choose the runtime
when launching a model.

Run this EXO build on every node in the cluster: TensorFold introduces new
instance and capability fields in EXO's shared state. Restart EXO after changing
the native executable so capability discovery reflects the new binary.

Source-build identity also includes the installed `mlx` and `mlx-lm` versions.
All nodes need matching build identities and dependencies. A TensorFold-only
cluster can use `mlx-none` consistently; it will not join a source-built cluster
whose identity includes different MLX packages. A mixed runtime cluster must
use the same supported dependency setup on every node. The existing version
gate remains in force.

## Requirements

Use a TensorFold build with token stream protocol version 1, added by
[ashhart/TensorFold#3285](https://github.com/ashhart/TensorFold/pull/3285).
TensorFold 1.2.8 and earlier lack it: EXO needs the exact committed token IDs,
where they fall against the semantic streaming events, and the raw token
position of each log probability row. Build the native executable using
TensorFold's instructions for your platform, then check its capabilities:

```bash
export EXO_TENSORFOLD_BINARY=/absolute/path/to/tensorfold-native
"$EXO_TENSORFOLD_BINARY" capabilities --json
```

The capability response must advertise schema 1, engine `zig`,
`token_stream_versions: [1]`, a supported Metal or CUDA backend, and the
`--unix-socket` and `--name` serve flags. Placement also checks the actual chip,
available GPU memory, model family and weight format. An executable that cannot
pass discovery does not qualify a node for TensorFold placement.

The initial model card is `mlx-community/Qwen3.5-2B-MLX-4bit`, with model type
`qwen3_5` and native weight format `mlx-q4g64`. Further models need a card with
the exact `model_type`, `tensorfold_format` and appropriate TensorFold backend
entries. A model's architecture name alone does not establish compatibility.

TensorFold loads the complete model directory resolved by EXO's existing model
store. EXO's existing download workflow and configured model directories still
apply. The adapter does not fetch native executables or convert weights.

## Run and select TensorFold

From the EXO checkout, install its Python and Rust dependencies and build the
dashboard. The existing `mlx-none` extra supports a TensorFold-only node without
installing MLX:

```bash
uv sync --extra mlx-none
(cd dashboard && npm ci && npm run build)
uv run --extra mlx-none exo
```

Open `http://localhost:52415`. If the first-run model picker appears, choose
**Skip for now** to reach the launch panel. Choose the model, expand Advanced
Options and select **TensorFold** under Runtime. The launch panel then uses one device.
The default launch settings use an 8,192-token context and native automatic
batch capacity. The native loader makes the final capacity decision; placement
reserves memory but cannot guarantee that every requested context will load.

The placement API exposes the same choice and optional context and parallel
settings:

```bash
curl --fail --get http://localhost:52415/instance/placement \
  --data-urlencode 'model_id=mlx-community/Qwen3.5-2B-MLX-4bit' \
  --data 'instance_meta=TensorFold' \
  --data 'sharding=Pipeline' \
  --data 'min_nodes=1' \
  --data 'tensorfold_context_length=8192' > placement.json

python -c 'import json; print(json.dumps({"instance": json.load(open("placement.json"))}))' \
  > launch.json
curl --fail http://localhost:52415/instance \
  -H 'Content-Type: application/json' --data-binary @launch.json
```

`tensorfold_parallel` is optional. When omitted, TensorFold chooses its batch
capacity. When supplied, it must be positive and becomes the native `--parallel`
argument. A TensorFold placement rejects tensor sharding and multiple nodes.

After the dashboard reports the model ready, use EXO's existing API:

```bash
curl --fail --no-buffer http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"mlx-community/Qwen3.5-2B-MLX-4bit","messages":[{"role":"user","content":"Explain a Unix socket in one sentence."}],"max_tokens":128,"stream":true}'
```

Add the usual authorization header if the EXO API is configured to require one.

## Request behavior

The adapter forwards the canonical conversation, including developer messages,
tool calls and tool results, without retokenizing it. TensorFold applies its
own model template. Native reasoning text and tool-call identifiers are
preserved. Log probability rows preserve their native text and byte sequences.
Each row carries its raw token position, which EXO checks against the committed
token order, so a token shown in parts keeps every row at its own position.

Sampling fields, stop strings, seeds, tools, reasoning effort and thinking
switches are forwarded. The loaded model's `/v1/models` capabilities gate log
probabilities, nonzero presence and frequency penalties, explicit thinking
requests and greedy-only generation. Unsupported nondefault repetition
penalties and repetition context sizes fail explicitly. Image inputs and EXO
remote prefill are outside this integration. Requests must allow at least one
output token.

`use_prefix_cache=true` requires the loaded native model to advertise an enabled
`prompt_cache`. It does not turn on a disabled native cache. Setting
`use_prefix_cache=false` leaves caching to TensorFold's native policy; it does
not disable TensorFold's own cache. Usage reports native cached prompt tokens.
EXO does not exchange TensorFold KV state with MLX instances, and TensorFold
instances cannot be selected for prefill/decode disaggregation.

Streaming text becomes available as native semantic events arrive. Final
success is withheld until committed token IDs, native usage, final metadata
and the end-of-stream marker agree. Repeated semantic anchors do not increase
token counts. A valid empty native completion retains zero usage without an
invented token ID. Heartbeat comments keep the runner watchdog informed without
adding text, tokens or usage. Malformed or truncated streams return an error
instead of a successful finish.

## Process lifecycle and configuration

Each instance starts an owned native process on a private Unix socket. EXO
requires healthy, nonwarming `/health` status and the requested loaded model
before running a one-token warmup. Warmup must complete before the instance is
marked ready. HTTP proxies are disabled for this local transport.

Inherited native rank settings must describe one rank. The adapter rejects
noncanonical or distributed `TF_GLM_*`, `TF_NEMOTRON_*`, `TF_MINIMAX_*`, `TF_TP_*`
and `TF_STAGE_*` rank/world values before starting a child. Leave those variables
unset, or use exactly `WORLD=1` and `RANK=0`. Device-selection variables such as
`TF_CUDA_DEVICE` retain their native meaning.

A small guardian watches the runner's parent pipe. Normal shutdown, a closed
pipe or a killed runner stops the owned native process group, escalates after
the shutdown grace period and removes its socket. This also covers descendants
left after the native leader exits. Cancelling one generation closes that
request's HTTP stream so TensorFold can release its work while other requests
continue.

The runner checks native and transport health while idle as well as while
generating. An exited native process therefore reaches EXO's existing runner
failure and restart path without waiting for another user request.

| Setting | Default | Meaning |
| --- | --- | --- |
| `EXO_TENSORFOLD_BINARY` | `tensorfold-native` on `PATH` | Native executable to probe and launch. |
| `EXO_TENSORFOLD_STARTUP_TIMEOUT` | 900 seconds | Deadline for native model loading and readiness. |
| `EXO_TENSORFOLD_REQUEST_TIMEOUT` | 300 seconds | Longest wait for the next output event, including queueing and prefill before the first. |
| `EXO_TENSORFOLD_SHUTDOWN_TIMEOUT` | 5 seconds | Native process-group shutdown grace period. |

Timeout values must be finite positive numbers. The request deadline restarts
with every output event, so a long reply that keeps producing finishes, while a
native process that stops producing is cut off. Heartbeat comments keep EXO's
liveness checks fed but never extend the deadline. A request the loaded model
can't run as sent (unsupported logprobs, penalties or images, say) gets EXO's
`invalid_request` error, an HTTP 400 for a whole response, and so does one the
native server refuses with a 400. The adapter limits active requests,
buffers streaming events with backpressure and bounds individual SSE frames,
tool arguments and retained token IDs.

## Validation

The adapter, transport, guardian and idle-health tests run with the default
suite. The two-node scenarios are cluster tests (`exo.testkit`), which CI runs
with the other fast cluster scenarios:

```bash
uv run pytest src/exo/worker/engines/tensorfold/tests
uv run pytest src/exo/testkit/tests/test_tensorfold_scenarios.py -m "cluster and not slow"
```

The native wire tests also start TensorFold's own Zig HTTP server with its
scripted engine when `EXO_TENSORFOLD_TEST_NATIVE_BINARY` names a `fake_serve`
build and `EXO_TENSORFOLD_TEST_NATIVE_FIXTURES` its fixtures directory;
otherwise they skip. See `src/exo/worker/engines/tensorfold/tests/README.md`.

The cluster scenarios' native executable is a protocol fixture. It exercises actual EXO
routing, runner supervision, subprocess ownership and Unix HTTP where the host
permits sockets, but it does not run GPU inference. Passing these tests is not
a hardware performance or model-quality result.

For an actual Metal or CUDA deployment, collect a qualification receipt:

```bash
uv run --extra mlx-none python tools/tensorfold_validate.py \
  --exo-url http://127.0.0.1:52415 \
  --model mlx-community/Qwen3.5-2B-MLX-4bit \
  --model-revision 93760be4f1f69842a46bc13dbdc0f19e291392a3 \
  --hardware 'actual GPU and memory configuration' \
  --binary "$EXO_TENSORFOLD_BINARY" \
  --repeats 3 --concurrency 2 --max-tokens 128 \
  --output tensorfold-validation.json
```

The hardware description and model revision supplied here are declarations;
verify them against the deployment before collecting the receipt. The tool can
also compare a separately running native server with `--native-url` and
`--native-model`. Record the real binary, model revision and launch settings
when reporting results. See the tests' README for the fault matrix and the
qualification tool's `--help` for additional options.
