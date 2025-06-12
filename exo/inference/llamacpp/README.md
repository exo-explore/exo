# LlamaCpp Inference Engine for Exo

This inference engine provides full support for GGUF models using the llama-cpp-python library, making it the default inference engine for the Exo framework.

## Features

- **Full GGUF Support**: Native support for all GGUF quantization formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32)
- **Automatic Quantization Detection**: Automatically detects quantization level from filename and optimizes parameters
- **GPU Acceleration**: Supports CUDA, Metal (macOS), and other GPU backends
- **Memory Optimization**: Automatically configures memory settings based on model size and system resources
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Supported Quantization Formats

| Format | Description | Memory Usage | Performance |
|--------|-------------|--------------|-------------|
| Q2_K   | 2-bit quantization | Lowest | Fast |
| Q3_K   | 3-bit quantization | Very Low | Fast |
| Q4_K   | 4-bit quantization | Low | Balanced |
| Q5_K   | 5-bit quantization | Medium | Good |
| Q6_K   | 6-bit quantization | Medium-High | Very Good |
| Q8_0   | 8-bit quantization | High | Excellent |
| F16    | 16-bit floating point | Very High | Best |
| F32    | 32-bit floating point | Highest | Best |

## Installation

```bash
pip install llama-cpp-python
```

For GPU support:
```bash
# CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## Usage

The llamacpp engine is now the default inference engine. To use it explicitly:

```bash
# Use llamacpp engine (default)
python -m exo

# Explicitly specify llamacpp engine
python -m exo --inference-engine llamacpp

# Run with a specific model
python -m exo run llama3.2-1b --prompt "Hello, world!"
```

## Configuration

The engine automatically configures itself based on:

1. **Model Quantization**: Detected from GGUF filename
2. **GPU Availability**: Auto-detects CUDA, Metal, etc.
3. **System Memory**: Adjusts parameters based on available RAM
4. **CPU Cores**: Optimizes threading for your system

### Manual Configuration

You can override default settings by modifying the `default_params` in `llama_inference_engine.py`:

```python
self.default_params = {
    "n_ctx": 4096,          # Context length
    "n_batch": 512,         # Batch size
    "n_threads": 8,         # CPU threads
    "n_gpu_layers": -1,     # GPU layers (-1 = all)
    "use_mmap": True,       # Memory mapping
    "f16_kv": True,         # Half-precision cache
}
```

## Performance Tips

1. **For CPU-only inference**: Use Q4_K or Q5_K quantizations for best balance
2. **For GPU inference**: Use Q6_K, Q8_0, or F16 for best quality
3. **For low memory systems**: Use Q2_K or Q3_K quantizations
4. **For maximum quality**: Use F16 or F32 (requires more memory)

## Troubleshooting

### Common Issues

1. **"llama-cpp-python not available"**
   ```bash
   pip install llama-cpp-python
   ```

2. **GPU not detected**
   - Ensure CUDA/Metal support is compiled in
   - Check GPU drivers are installed

3. **Out of memory errors**
   - Use smaller quantization (Q2_K, Q3_K)
   - Reduce `n_ctx` (context length)
   - Reduce `n_batch` size

4. **Slow inference**
   - Enable GPU acceleration
   - Use larger quantization if memory allows
   - Increase `n_batch` size

### Debug Mode

Enable debug output to see detailed configuration:

```bash
DEBUG=2 python -m exo --inference-engine llamacpp
```

This will show:
- Quantization detection
- GPU configuration
- Memory optimization
- Model loading details

## Model Compatibility

The engine works with any GGUF model from:
- Hugging Face (models ending in `.gguf`)
- Local GGUF files
- Any llama.cpp compatible model

Popular model series supported:
- Llama 2/3/3.1/3.2
- Mistral 7B/8x7B
- CodeLlama
- Phi-3
- Qwen
- And many more!

## Architecture

The LlamaCpp inference engine:
- Implements the `InferenceEngine` abstract base class
- Uses thread pools for async operations
- Manages model state and caching
- Provides automatic optimization
- Supports concurrent requests

## Contributing

To extend the llamacpp engine:
1. Modify `llama_inference_engine.py`
2. Add new quantization detection in `_optimize_for_quantization()`
3. Update GPU detection in `_detect_gpu_support()`
4. Test with various GGUF models