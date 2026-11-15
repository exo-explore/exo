# TensorFold backend tests

These tests check the protocol, transport and process handling. Their native
executables are scripted and never load weights, so passing them does not
qualify Metal or CUDA kernels, model output or speed on a real device.

## In the default suite

```sh
uv run pytest src/exo/worker/engines/tensorfold/tests
```

- `test_requests.py`, `test_protocol.py` and `test_config.py` cover request
  mapping, the token stream protocol and settings.
- `test_transport.py` runs the production background transport, SSE decoder,
  token protocol, bounded queue, cancellation and idle deadline over an
  in-memory byte stream. It fragments UTF-8 and injects invalid JSON, offsets,
  usage and truncation, then requires the next request to succeed. A reply that
  keeps producing past the idle deadline must finish; one that stalls must not.
- `test_guardian.py` starts the production guardian with real child processes
  that ignore SIGTERM. The guardian must kill its whole process group and remove
  the socket folder when its owner's pipe closes or the native leader exits.
- `test_idle_health.py` drives the real Runner through loading and warmup with a
  small test engine, then makes the engine unhealthy while no request is open.
  The runner must notice without a new request, report the error, close the
  engine and stop its task reader.
- `test_native_wire.py` starts TensorFold's own Zig HTTP server with its scripted
  engine and feeds the bytes it sends through EXO's `EventDecoder` and
  `StreamProtocol`. It skips unless both variables below are set:

```sh
export EXO_TENSORFOLD_TEST_NATIVE_BINARY=/path/to/fake_serve
export EXO_TENSORFOLD_TEST_NATIVE_FIXTURES=/path/to/tensorfold/zig/tests/server/fixtures
```

  Build `fake_serve` in a TensorFold checkout with `zig build server-cpu`. The
  tests check exact token IDs, both usage arrangements, Unicode split across
  tokens, reasoning, cached prompt counts, tool calls, length limits, refusal of
  an unknown protocol version, and that closing the HTTP connection after the
  first output frees the native request within a second.

## Cluster scenarios

```sh
uv run pytest src/exo/testkit/tests/test_tensorfold_scenarios.py -m "cluster and not slow"
```

`exo.testkit.tensorfold` starts two real EXO nodes (networking, discovery,
election, placement, API forwarding, worker, runner and supervisor). A request
to node zero is served by the model placed on node one. The production builder
starts `fixture_server.py` through the production guardian on a private Unix
socket. The fixture follows the host: Metal on macOS, CUDA elsewhere, where the
node wrapper supplies an 8 GiB CUDA memory reading. A tiny dummy model and a
fixture-only model card are installed locally; nothing is downloaded.

The scenarios cover cross-node routing, request mapping, streamed and whole
responses, usage, Unicode, reasoning, tool calls, concurrent requests,
disconnect cancellation, malformed native replies, native HTTP errors, a native
crash and the supervisor's restart. Closing the cluster must not need the
testkit's forced cleanup.

## On real hardware

Start EXO with a TensorFold build that advertises token stream version 1, place a
supported model with the TensorFold runtime, then run the standard-library
qualification tool. It doesn't start servers, download weights or change
deployments:

```sh
python tools/tensorfold_validate.py \
  --exo-url http://127.0.0.1:52415 \
  --model YOUR_MODEL_ID \
  --model-revision YOUR_PINNED_WEIGHTS_REVISION \
  --hardware 'GPU model, memory, operating system and driver' \
  --binary /path/to/tensorfold-native \
  --repeats 3 --concurrency 2 --max-tokens 128 \
  --output tensorfold-validation.json
```

`--api-key-env EXO_API_TOKEN` reads an API token from that variable. The report
keeps no keys, URLs, local paths or generated text; hardware and revision are
recorded as declared. The tool checks that every instance of the model is a
`TensorFoldInstance`, then checks complete streams, usage, limits, repeated and
concurrent requests and a request after a client disconnect. Confirm that the
disconnect freed the native request with native `/health` or process monitoring.

`--native-url` adds a separately running TensorFold server on the same weights
as a direct baseline (`--native-model` and `--native-api-key-env` name its model
and key). First-output times include transport and prompt work, and the token
rate covers the whole request, not decode alone. The repeated prompt may hit the
prompt cache; it is not a cold-cache benchmark.

For each new model or device, also check real-model stop strings, reasoning,
tools and supported logprobs, cancellation under load, crash recovery, that a
killed runner leaves no native process behind, memory use, and output quality
for the model and quantization before claiming support.
