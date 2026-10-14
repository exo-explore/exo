# Qwen3.8-27B Reproduction

This contribution targets Q8 and BF16 text generation on Apple Silicon. The cards
do not advertise vision, CUDA or CPU qualification. No weights are included.
The processor and cache corrections address separate failures; the Nix override
applies the cache correction to EXO's actual MLX-LM dependency. `uv run exo` does
not apply this Nix-only patch. Dependency pins are unchanged in `uv.lock`.

The cache patch adapts only `ArraysCache.advance()` from Pierre Lamy's
[bb615eb patch](https://github.com/pierre427/mlx-lm/commit/bb615ebdb5aff33eb931ac0627cae142bd7adbaa),
not its `extract()` change. Related upstream work includes
[MLX-LM #1632](https://github.com/ml-explore/mlx-lm/pull/1632) and
[#1845](https://github.com/ml-explore/mlx-lm/issues/1845). A companion MLX-LM PR
has not yet been published. These references do not imply endorsement.

## Component Checks

Run from this checkout on macOS with EXO's documented Nix prerequisites.
Use Python without `-O` or `PYTHONOPTIMIZE`; the scripts include assertions.

```bash
set -e
EXO_TEST_PYTHON_ENV="$(nix build --no-link --print-out-paths path:.#exo.venv)"
EXO_TEST_DASHBOARD="$(nix build --no-link --print-out-paths path:.#dashboard)"
"$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/arrays-cache-metadata-regression.py
```

Expected: `arrays-cache metadata regression: PASS`. No weights or EXO service
are needed. This checks metadata graph bounds after 256 cache advances and the
final numerical values, not distributed inference or long-run resource usage.

Download the complete model repositories at these revisions using Hugging Face:

| Model | Revision |
| --- | --- |
| `mlx-community/Qwen3.8-27B-8bit` | `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` |
| `mlx-community/Qwen3.8-27B-bf16` | `6f265714824f3c38d4452baa1628aef3d9b9aae9` |

Set `MODEL_ROOT` to a directory containing the complete downloads named
`mlx-community--Qwen3.8-27B-8bit` and `mlx-community--Qwen3.8-27B-bf16`.

```bash
MODEL_ROOT="/absolute/path/to/local-models"
(
  set -e
  export HOME="$(mktemp -d)"
  unset EXO_HOME EXO_MODELS_READ_ONLY_DIRS PYTHONPATH PYTHONOPTIMIZE
  export EXO_DEFAULT_MODELS_DIR="$HOME/models"
  export EXO_MODELS_DIRS="$MODEL_ROOT"
  export EXO_RESOURCES_DIR="$PWD/resources"
  export EXO_DASHBOARD_DIR="$EXO_TEST_DASHBOARD"
  "$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/qwen-vision-regression.py mlx-community/Qwen3.8-27B-8bit
  "$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/qwen-vision-regression.py mlx-community/Qwen3.8-27B-bf16
)
```

Each invocation must print `Qwen vision loader regression: PASS (MODEL_ID)`.
It loads real vision weights but does not perform image understanding. The
temporary home is retained for inspection; no services or model files are changed.

Registration and native source tests, following the macOS CI approach:

```bash
TEST_ENV="$(nix build --no-link --print-out-paths path:.#exo-test-env)"
(
  export HOME="$(mktemp -d)"
  unset EXO_HOME
  export PYTHONPATH="$PWD/src"
  export EXO_RESOURCES_DIR="$PWD/resources"
  export EXO_DASHBOARD_DIR="$PWD/dashboard"
  "$TEST_ENV/bin/python" -m pytest src -m "not slow" --import-mode=importlib
)
"$TEST_ENV/bin/basedpyright" --pythonpath "$TEST_ENV/bin/python"
"$TEST_ENV/bin/ruff" check
nix fmt
```

## Distributed Request Replay

This is a manual, opt-in workload, not part of pytest. Build `nix build path:.#exo`
and run that build on all four nodes using EXO's documented JACCL/RDMA setup.
Do not mix builds. Place one variant at a time. Inspect `/state`: require one
matching `MlxJacclInstance`, four Tensor shards with `worldSize` four, and all
four assigned runners in `RunnerReady`. Recheck before and after each request.
Neither the generator nor curl places models, restarts services or verifies this
topology for you. Other serving modes are not equivalent to this qualification.

```bash
MODEL="mlx-community/Qwen3.8-27B-8bit"
EXO_URL="http://localhost:52415"
"$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/qwen-requests.py "$MODEL" sustained > sustained.json
curl --fail-with-body --max-time 1800 -N -H 'Content-Type: application/json' \
  --data-binary @sustained.json "$EXO_URL/v1/chat/completions" > sustained.sse
"$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/qwen-requests.py "$MODEL" recall > recall.json
curl --fail-with-body --max-time 1800 -N -H 'Content-Type: application/json' \
  --data-binary @recall.json "$EXO_URL/v1/chat/completions" > recall.sse
```

For the cached extension, concatenate `choices[].delta.content` from the recall
SSE JSON messages, preserving the answer exactly, into `recall-answer.txt`.
Do not include reasoning, SSE framing, or an added newline. Then:

```bash
"$EXO_TEST_PYTHON_ENV/bin/python" nix/tests/qwen-requests.py "$MODEL" generation \
  --previous-answer recall-answer.txt > generation.json
curl --fail-with-body --max-time 1800 -N -H 'Content-Type: application/json' \
  --data-binary @generation.json "$EXO_URL/v1/chat/completions" > generation.sse
```

Repeat separately with the BF16 ID. HTTP success alone is insufficient: inspect
SSE for errors, usage, nonempty output, `finish_reason: stop` and `[DONE]`. Recall
must return `copper-41, harbor-72, spruce-93`; generation must begin with the
updated middle value `anchor-84`. Check actual prompt/cached token counts, not
the nominal workload name. The original requests produced 210,668 prompt tokens
for recall and 243,192 for generation with 210,666 cached; token counts can change
with dependencies or answers. A short output does not test sustained decoding.
If any rank fails or curl times out, stop the workload and inspect the server;
client timeout does not prove the server stopped generation. Do not treat errors,
truncation or a missing marker as passing. Generated C# has not been compiled.

## Evidence Limits

On September 9, 2026, a development build with both corrections completed Q8
(17,992 output tokens) and BF16 (19,211 output tokens) sustained workloads and
the cached 243k conversations on four M3 Ultra nodes. These are historical
compatibility observations, not a four-node validation of this upstream port.
The public-source component checks ran on macOS 26.6.1 (25G76). Full-window cold
prefill, concurrency, overnight stability and multimodal inference remain unqualified.