# Contributing to EXO

Thank you for your interest in contributing to EXO!

## Getting Started

To run EXO from source:

**Prerequisites:**
- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
  ```bash
  brew install uv
  ```
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- [macmon](https://github.com/vladkens/macmon) (for hardware monitoring on Apple Silicon)
  Use the pinned fork revision used by this repo instead of Homebrew `macmon`.
  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon \
    --force
  ```

```bash
git clone https://github.com/exo-explore/exo.git
cd exo/dashboard
npm install && npm run build && cd ..
uv sync --extra mlx
uv run exo
```

## Development

EXO is built with a mix of Rust, Python, and TypeScript (Svelte for the dashboard), and the codebase is actively evolving. Before starting work:

- Pull the latest source to ensure you're working with the most recent code
- Keep your changes focused - implement one feature or fix per pull request
- Avoid combining unrelated changes, even if they seem small

This makes reviews faster and helps us maintain code quality as the project evolves.

## Code Style

Write pure functions where possible. When adding new code, prefer Rust unless there's a good reason otherwise. Leverage the type systems available to you - Rust's type system, Python type hints, and TypeScript types. Comments should explain why you're doing something, not what the code does - especially for non-obvious decisions.

Run `nix fmt` to auto-format your code before submitting.

## Model Cards

EXO uses TOML-based model cards to define model metadata and capabilities. Model cards are stored in:
- `resources/inference_model_cards/<family>/` for text generation models, one folder per model family (see [Adding a Model](#adding-a-model))
- `resources/image_model_cards/` for image generation models
- `~/.exo/custom_model_cards/` for user-added custom models

### Adding a Model Card

To add a new model, create a TOML file with the following structure:

```toml
model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
n_layers = 16
hidden_size = 2048
supports_tensor = true
tasks = ["TextGeneration"]
family = "llama"
quantization = "4bit"
base_model = "Llama 3.2 1B"
capabilities = ["text"]

[storage_size]
in_bytes = 729808896
```

### Required Fields

- `model_id`: Hugging Face model identifier
- `n_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `supports_tensor`: Whether the model supports tensor parallelism
- `tasks`: List of supported tasks (`TextGeneration`, `TextToImage`, `ImageToImage`)
- `family`: Model family (e.g., "llama", "deepseek", "qwen")
- `quantization`: Quantization level (e.g., "4bit", "8bit", "bf16")
- `base_model`: Human-readable base model name
- `capabilities`: List of capabilities (e.g., `["text"]`, `["text", "thinking"]`)

### Optional Fields

- `model_type`: The `model_type` from the checkpoint's `config.json` (e.g. `"llama"`, `"deepseek_v4"`). The master uses it to find the model's family and its placement rules
- `components`: For multi-component models (like image models with separate text encoders and transformers)
- `uses_cfg`: Whether the model uses classifier-free guidance (for image models)
- `trust_remote_code`: Whether to allow remote code execution (defaults to `false` for security)

### Capabilities

The `capabilities` field defines what the model can do:
- `text`: Standard text generation
- `thinking`: Model supports chain-of-thought reasoning
- `thinking_toggle`: Thinking can be enabled/disabled via `enable_thinking` parameter
- `image_edit`: Model supports image-to-image editing (FLUX.1-Kontext)

### Security Note

By default, `trust_remote_code` is set to `false` for security. Only enable it if the model explicitly requires remote code execution from the Hugging Face hub.

## Adding a Model

Everything exo needs for a family of models beyond what mlx-lm provides lives in one folder,
`src/exo/worker/engines/mlx/models/<family>/`, next to its cards in
`resources/inference_model_cards/<family>/`. Shared mechanisms (pipeline layers, the
tensor-parallel strategy base, MoE wrappers) are in `src/exo/worker/engines/mlx/sharding/` and
never name a model. A family is found by the `model_type` in the checkpoint's `config.json`.

**A new checkpoint of an architecture exo already supports** is a card: add it to the family's
card folder with `model_type` set. If its tokenizer config doesn't list all of its EOS tokens,
add an `EosRule` to the family's `spec.py`.

**A new architecture or family** gets a folder:

| file | contents |
|---|---|
| `__init__.py` | a docstring only: the master imports `spec.py` through this package |
| `spec.py` | `SPEC = FamilySpec(...)`: the `model_type`s, the Hugging Face `architectures` that support tensor parallelism, `PlacementRules`, EOS rules. Must not import MLX: the master runs without it |
| `sharding.py` | a `TensorParallelShardingStrategy` subclass, and a `fix_pipeline_shard` function if the model needs adjusting after its layers are split (e.g. cache indices of hybrid models) |
| `family.py` | `FAMILY = ModelFamily(...)`: one `Architecture` per mlx-lm model class (its strategy, pipeline fixup, layer counter, output parser), plus an optional prompt encoder, chat-template hook, tokenizer loader and `ModelClasses` |
| optional | `prompt.py` (models without a Jinja chat template), `parser.py` (output formats the generic parsers don't handle), `tokenizer.py`, `cache.py` (prefix-cache copies of a custom cache class), `model.py` (a model definition exo's mlx-lm doesn't have, loaded through `ModelClasses`) |

Then add `SPEC` to `FAMILY_SPECS` in `models/spec_registry.py` and `FAMILY` to `FAMILIES` in
`models/registry.py`, at the same position, and add a `.typings/mlx_lm/models/<model_type>.pyi`
stub for the mlx-lm classes you use.

Code copied from elsewhere (a model definition, a reference prompt encoder) starts with a
provenance header: source URL, revision, licence (with its notice when the licence asks for
it), the date it was copied, and every local change.

The registry tests (`src/exo/worker/engines/mlx/models/tests/`) check that every built-in card
belongs to a family, that each family's `model_type`s and classes line up, and that the master
side imports without MLX. Add your family's own tests (sharding equivalence, parser and encoder
goldens) next to its code.

## API Adapters

EXO supports multiple API formats through an adapter pattern. Adapters convert API-specific request formats to the internal `TextGenerationTaskParams` format and convert internal token chunks back to API-specific responses.

### Adapter Architecture

All adapters live in `src/exo/master/adapters/` and follow the same pattern:

1. Convert API-specific requests to `TextGenerationTaskParams`
2. Handle both streaming and non-streaming response generation
3. Convert internal `TokenChunk` objects to API-specific formats
4. Manage error handling and edge cases

### Existing Adapters

- `chat_completions.py`: OpenAI Chat Completions API
- `claude.py`: Anthropic Claude Messages API
- `responses.py`: OpenAI Responses API
- `ollama.py`: Ollama API (for OpenWebUI compatibility)

### Adding a New API Adapter

To add support for a new API format:

1. Create a new adapter file in `src/exo/master/adapters/`
2. Implement a request conversion function:
   ```python
   def your_api_request_to_text_generation(
       request: YourAPIRequest,
   ) -> TextGenerationTaskParams:
       # Convert API request to internal format
       pass
   ```
3. Implement streaming response generation:
   ```python
   async def generate_your_api_stream(
       command_id: CommandId,
       chunk_stream: AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk, None],
   ) -> AsyncGenerator[str, None]:
       # Convert internal chunks to API-specific streaming format
       pass
   ```
4. Implement non-streaming response collection:
   ```python
   async def collect_your_api_response(
       command_id: CommandId,
       chunk_stream: AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk, None],
   ) -> AsyncGenerator[str]:
       # Collect all chunks and return single response
       pass
   ```
5. Register the adapter endpoints in `src/exo/master/api.py`

The adapter pattern keeps API-specific logic isolated from core inference systems. Internal systems (worker, runner, event sourcing) only see `TextGenerationTaskParams` and `TokenChunk` objects - no API-specific types cross the adapter boundary.

For detailed API documentation, see [docs/api.md](docs/api.md).

## Testing

EXO relies heavily on manual testing at this point in the project, but this is evolving. Before submitting a change, test both before and after to demonstrate how your change improves behavior. Do the best you can with the hardware you have available - if you need help testing, ask and we'll do our best to assist. Add automated tests where possible - we're actively working to substantially improve our automated testing story.

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request and follow the PR template

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (macOS version, hardware, etc.)

## Questions?

Join our community:
- [X](https://x.com/exolabs)
